use std::{
    borrow::BorrowMut,
    cell::RefCell,
    collections::{HashMap, VecDeque},
    hash::Hash,
    rc::Rc,
};

use ash::{extensions::nv, prelude::VkResult, vk};

use super::TBuffer;

use super::Context;
use super::Fence;
pub struct KernelArgs<T> {
    pub sets: Vec<Set>,
    pub push_constants: Option<T>,
}
/*
let kernel = create_kernel!("xxx.spv", Argument::StorageBuffer(xxx), Argument::UniformBuffer(xxx))

*/
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Layout {
    pub sets: Vec<Set>,
    pub push_constants: Option<usize>,
}
#[derive(Clone, PartialEq, Eq, Hash)]

pub enum Set {
    Bindings(Vec<Binding>),
    StorageBufferArray(Vec<vk::Buffer>),
    // UniformBufferArray(&'a [vk::Buffer]),
    // StorageImageArray(Vec<>),
    SampledImageArray(Vec<(vk::ImageView, vk::ImageLayout)>),
}
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum Binding {
    AccelerationStructure(vk::AccelerationStructureNV),
    StorageBuffer(vk::Buffer),
    UniformBuffer(vk::Buffer),
    Sampler(vk::Sampler),
    // StorageImage(vk::ImageView)
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum BindingType {
    StorageBuffer,
    UniformBuffer,
    AccelerationStructure,
    SampledImage,
    Sampler,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum SetLayout {
    Bindings(Vec<BindingType>),
    BindlessArray(BindingType),
}
fn get_layout(set: &Set) -> SetLayout {
    match set {
        Set::Bindings(bindings) => SetLayout::Bindings(
            bindings
                .iter()
                .map(|binding| match binding {
                    Binding::StorageBuffer(_x) => BindingType::StorageBuffer,
                    Binding::UniformBuffer(_x) => BindingType::UniformBuffer,
                    Binding::AccelerationStructure(_x) => BindingType::AccelerationStructure,
                    Binding::Sampler(_) => BindingType::Sampler,
                })
                .collect(),
        ),
        Set::SampledImageArray(_) => SetLayout::BindlessArray(BindingType::SampledImage),
        Set::StorageBufferArray(_) => SetLayout::BindlessArray(BindingType::StorageBuffer),
    }
}
struct DescriptorCache {
    descriptor_pools: Vec<vk::DescriptorPool>,
    cache: HashMap<Set, vk::DescriptorSet>,
    ctx: Context,
    descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    fences: Vec<VecDeque<Rc<Fence>>>,
    stage: vk::ShaderStageFlags,
    layouts: Vec<SetLayout>,
}
impl Drop for DescriptorCache {
    fn drop(&mut self) {
        unsafe {
            self.ctx.device.destroy_descriptor_pool(
                self.descriptor_pools[0],
                self.ctx.allocation_callbacks.as_ref(),
            );
            self.ctx.device.destroy_descriptor_pool(
                self.descriptor_pools[1],
                self.ctx.allocation_callbacks.as_ref(),
            );
        }
    }
}

impl DescriptorCache {
    const MAX_FENCE: usize = 1024;
    fn add_fence(&mut self, fence: Rc<Fence>) {
        if self.fences[0].len() > Self::MAX_FENCE {
            while self.fences[0].len() > Self::MAX_FENCE / 2 {
                let fence = self.fences[0].pop_front().unwrap();
                fence.wait();
            }
        }
        self.fences[0].push_back(fence);
    }
    fn allocate(&mut self, idx: usize, set: &Set) -> vk::DescriptorSet {
        {
            let layout = get_layout(set);
            if layout != self.layouts[idx] {
                panic!("set differs in layout!");
            }
        }
        if let Some(descriptor_set) = self.cache.get(set) {
            *descriptor_set
        } else {
            unsafe {
                let descriptor_set = create_descriptor_set(
                    &self.ctx.device,
                    self.ctx.allocation_callbacks.as_ref(),
                    set,
                    self.stage,
                    self.descriptor_pools[0],
                    self.descriptor_set_layouts[idx],
                );
                match descriptor_set {
                    Ok(descriptor_set) => {
                        self.cache.insert(set.clone(), descriptor_set);
                        descriptor_set
                    }
                    Err(err) => {
                        if err == vk::Result::ERROR_OUT_OF_POOL_MEMORY {
                            let mut need_wait = vec![];
                            self.fences.swap(0, 1);
                            self.descriptor_pools.swap(0, 1);
                            for fence in &self.fences[0] {
                                if !self.ctx.device.get_fence_status(fence.inner).unwrap() {
                                    need_wait.push(fence.inner);
                                }
                            }
                            if !need_wait.is_empty() {
                                self.ctx
                                    .device
                                    .wait_for_fences(&need_wait, true, u64::MAX)
                                    .unwrap();
                            }
                            self.fences[0].clear();
                            // now switch pool
                            self.cache.clear();
                            // retry
                            let descriptor_set = create_descriptor_set(
                                &self.ctx.device,
                                self.ctx.allocation_callbacks.as_ref(),
                                set,
                                self.stage,
                                self.descriptor_pools[0],
                                self.descriptor_set_layouts[idx],
                            )
                            .unwrap();
                            self.cache.insert(set.clone(), descriptor_set);
                            descriptor_set
                        } else {
                            panic!("allocate_descriptor_sets: {:?} ", err);
                        }
                    }
                }
            }
        }
    }
    fn new(
        ctx: &Context,
        sets: &Vec<Set>,
        descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
        stage: vk::ShaderStageFlags,
    ) -> Self {
        let layouts = sets.iter().map(|x| get_layout(x)).collect();
        unsafe {
            let descriptor_sizes = [
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::ACCELERATION_STRUCTURE_NV,
                    descriptor_count: 3,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: 1024,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 1024,
                },
            ];
            let pools = vec![
                ctx.device
                    .create_descriptor_pool(
                        &vk::DescriptorPoolCreateInfo::builder()
                            .pool_sizes(&descriptor_sizes)
                            .max_sets(128)
                            .build(),
                        ctx.allocation_callbacks.as_ref(),
                    )
                    .unwrap(),
                ctx.device
                    .create_descriptor_pool(
                        &vk::DescriptorPoolCreateInfo::builder()
                            .pool_sizes(&descriptor_sizes)
                            .max_sets(128)
                            .build(),
                        ctx.allocation_callbacks.as_ref(),
                    )
                    .unwrap(),
            ];
            Self {
                descriptor_pools: pools,
                ctx: ctx.clone(),
                cache: HashMap::new(),
                descriptor_set_layouts,
                fences: vec![VecDeque::new(), VecDeque::new()],
                stage,
                layouts,
            }
        }
    }
}

pub struct Kernel {
    shader_modules: Vec<vk::ShaderModule>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    pc_range: Option<u32>,
    ctx: Context,
    stage: vk::ShaderStageFlags,
    sets: Vec<Set>,
    descriptor_sets: Vec<vk::DescriptorSet>,
    descriptor_cache: RefCell<DescriptorCache>,
    descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    rtx: Option<Rc<nv::RayTracing>>,
    pub sbt: Option<TBuffer<u8>>,
}

fn create_descriptor_set_layout(
    device: &ash::Device,
    allocation_callbacks: Option<&vk::AllocationCallbacks>,
    set: &Set,
    stage: vk::ShaderStageFlags,
) -> vk::DescriptorSetLayout {
    match set {
        Set::Bindings(bindings) => {
            let flags: Vec<vk::DescriptorBindingFlags> =
                vec![vk::DescriptorBindingFlagsEXT::empty(); bindings.len()];
            let mut binding_flags = vk::DescriptorSetLayoutBindingFlagsCreateInfoEXT::builder()
                .binding_flags(&flags)
                .build();
            let real_bindings: Vec<_> = bindings
                .iter()
                .enumerate()
                .map(|(i, binding)| {
                    let ty = match binding {
                        Binding::StorageBuffer(_x) => vk::DescriptorType::STORAGE_BUFFER,
                        Binding::UniformBuffer(_x) => vk::DescriptorType::UNIFORM_BUFFER,
                        Binding::AccelerationStructure(_x) => {
                            vk::DescriptorType::ACCELERATION_STRUCTURE_NV
                        }
                        Binding::Sampler(_) => vk::DescriptorType::SAMPLER,
                    };
                    vk::DescriptorSetLayoutBinding::builder()
                        .binding(i as u32)
                        .descriptor_count(1)
                        .descriptor_type(ty)
                        .stage_flags(stage)
                        .build()
                })
                .collect();
            unsafe {
                let descriptor_set_layout = device
                    .create_descriptor_set_layout(
                        &vk::DescriptorSetLayoutCreateInfo::builder()
                            .bindings(&real_bindings)
                            .push_next(&mut binding_flags)
                            .build(),
                        allocation_callbacks,
                    )
                    .unwrap();
                descriptor_set_layout
            }
        }
        Set::SampledImageArray(images) => {
            let flags: Vec<vk::DescriptorBindingFlags> =
                vec![vk::DescriptorBindingFlagsEXT::VARIABLE_DESCRIPTOR_COUNT];
            let mut binding_flags = vk::DescriptorSetLayoutBindingFlagsCreateInfoEXT::builder()
                .binding_flags(&flags)
                .build();
            let real_binding = [vk::DescriptorSetLayoutBinding::builder() // Instances
                .binding(0)
                .descriptor_count((images.len() as u32).max(1))
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .stage_flags(stage)
                .build()];
            let create_info = vk::DescriptorSetLayoutCreateInfo::builder()
                .bindings(&real_binding)
                .push_next(&mut binding_flags)
                .build();
            unsafe {
                let descriptor_set_layout = device
                    .create_descriptor_set_layout(&create_info, allocation_callbacks)
                    .unwrap();
                descriptor_set_layout
            }
        }
        Set::StorageBufferArray(buffers) => {
            let flags: Vec<vk::DescriptorBindingFlags> =
                vec![vk::DescriptorBindingFlagsEXT::VARIABLE_DESCRIPTOR_COUNT];
            let mut binding_flags = vk::DescriptorSetLayoutBindingFlagsCreateInfoEXT::builder()
                .binding_flags(&flags)
                .build();
            let real_binding = [vk::DescriptorSetLayoutBinding::builder() // Instances
                .binding(0)
                .descriptor_count((buffers.len() as u32).max(1))
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .stage_flags(stage)
                .build()];
            let create_info = vk::DescriptorSetLayoutCreateInfo::builder()
                .bindings(&real_binding)
                .push_next(&mut binding_flags)
                .build();
            unsafe {
                let descriptor_set_layout = device
                    .create_descriptor_set_layout(&create_info, allocation_callbacks)
                    .unwrap();
                descriptor_set_layout
            }
        }
    }
}

fn create_descriptor_set(
    device: &ash::Device,
    _allocation_callbacks: Option<&vk::AllocationCallbacks>,
    set: &Set,
    _stage: vk::ShaderStageFlags,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
) -> VkResult<vk::DescriptorSet> {
    unsafe {
        let descriptor_set_layout = [descriptor_set_layout];
        match set {
            Set::Bindings(bindings) => {
                let descriptor_set = device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::builder()
                        .set_layouts(&descriptor_set_layout)
                        .descriptor_pool(descriptor_pool)
                        .build(),
                )?[0];
                for i in 0..bindings.len() {
                    match bindings[i] {
                        Binding::Sampler(sampler) => {
                            let info =
                                [vk::DescriptorImageInfo::builder().sampler(sampler).build()];
                            let write = vk::WriteDescriptorSet::builder()
                                .dst_set(descriptor_set)
                                .dst_binding(i as u32)
                                .dst_array_element(0)
                                .descriptor_type(vk::DescriptorType::SAMPLER)
                                .image_info(&info)
                                .build();
                            device.update_descriptor_sets(&[write], &[]);
                        }
                        Binding::StorageBuffer(_) | Binding::UniformBuffer(_) => {
                            let (buf, ty) = match bindings[i] {
                                Binding::StorageBuffer(buf) => {
                                    (buf, vk::DescriptorType::STORAGE_BUFFER)
                                }
                                Binding::UniformBuffer(buf) => {
                                    (buf, vk::DescriptorType::UNIFORM_BUFFER)
                                }
                                _ => unreachable!(),
                            };
                            let info = [vk::DescriptorBufferInfo::builder()
                                .buffer(buf)
                                .range(vk::WHOLE_SIZE)
                                .build()];
                            let write = vk::WriteDescriptorSet::builder()
                                .dst_set(descriptor_set)
                                .dst_binding(i as u32)
                                .dst_array_element(0)
                                .descriptor_type(ty)
                                .buffer_info(&info)
                                .build();
                            device.update_descriptor_sets(&[write], &[]);
                        }
                        Binding::AccelerationStructure(accel) => {
                            let accel_structs = [accel];
                            let mut accel_info =
                                vk::WriteDescriptorSetAccelerationStructureNV::builder()
                                    .acceleration_structures(&accel_structs)
                                    .build();
                            let mut accel_write = vk::WriteDescriptorSet::builder()
                                .dst_set(descriptor_set)
                                .dst_binding(i as u32)
                                .dst_array_element(0)
                                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_NV)
                                .push_next(&mut accel_info)
                                .build();
                            accel_write.descriptor_count = 1;
                            device.update_descriptor_sets(&[accel_write], &[]);
                        }
                    }
                }
                Ok(descriptor_set)
            }
            Set::SampledImageArray(images) => {
                let counts = [images.len() as u32];
                let mut ext = vk::DescriptorSetVariableDescriptorCountAllocateInfo::builder()
                    .descriptor_counts(&counts);
                let descriptor_set = device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::builder()
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(&descriptor_set_layout)
                        .push_next(&mut ext)
                        .build(),
                )?[0];
                if images.len() > 0 {
                    let image_infos: Vec<_> = images
                        .iter()
                        .map(|(view, layout)| {
                            vk::DescriptorImageInfo::builder()
                                .image_layout(*layout)
                                .image_view(*view)
                                .build()
                        })
                        .collect();
                    let write = vk::WriteDescriptorSet::builder()
                        .dst_set(descriptor_set)
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(&image_infos)
                        .build();
                    device.update_descriptor_sets(&[write], &[]);
                }
                // device.update_descriptor_sets(&[write], &[]);
                Ok(descriptor_set)
            }
            Set::StorageBufferArray(buffers) => {
                let counts = [buffers.len() as u32];
                let mut ext = vk::DescriptorSetVariableDescriptorCountAllocateInfo::builder()
                    .descriptor_counts(&counts);
                let descriptor_set = device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::builder()
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(&descriptor_set_layout)
                        .push_next(&mut ext)
                        .build(),
                )?[0];
                if buffers.len() > 0 {
                    let buffer_infos: Vec<_> = buffers
                        .iter()
                        .map(|buffer| {
                            vk::DescriptorBufferInfo::builder()
                                .buffer(*buffer)
                                .range(vk::WHOLE_SIZE)
                                .build()
                        })
                        .collect();
                    let write = vk::WriteDescriptorSet::builder()
                        .dst_set(descriptor_set)
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&buffer_infos)
                        .build();
                    device.update_descriptor_sets(&[write], &[]);
                }
                Ok(descriptor_set)
            }
        }
    }
}

pub struct CommandEncoder{
    pub command_buffer: vk::CommandBuffer,
    queue: vk::Queue,
    fence: RefCell<Option<Rc<Fence>>>,
    ctx:Context,
    wait_semaphores: RefCell<Vec<vk::Semaphore>>,
    signal_semaphores: RefCell<Vec<vk::Semaphore>>,
}
impl CommandEncoder {
    pub fn new(
        ctx:&Context,
        command_buffer: vk::CommandBuffer,
        queue: vk::Queue,
        begin_info: &vk::CommandBufferBeginInfo,
    ) -> Self {
        unsafe {
            ctx.device
                .begin_command_buffer(command_buffer, begin_info)
                .unwrap();
        }
        Self {
            command_buffer,
            queue,
            fence: RefCell::new(None),
            ctx:ctx.clone(),
            signal_semaphores: RefCell::new(vec![]),
            wait_semaphores: RefCell::new(vec![]),
        }
    }
    pub fn get_fence(&self) -> Rc<Fence> {
        let mut fence = self.fence.borrow_mut();
        if fence.is_some() {
            (*fence).clone().unwrap().clone()
        } else {
            unsafe {
                *fence = Some(Rc::new(Fence::new(
                    self.ctx.device
                        .create_fence(&vk::FenceCreateInfo::default(), None)
                        .unwrap(),
                    &self.ctx.device,
                )));
                (*fence).clone().unwrap().clone()
            }
        }
    }
    pub fn wait_semaphore(&self, semaphore:vk::Semaphore){
        let mut wait_semaphores = self.wait_semaphores.borrow_mut();
        wait_semaphores.push(semaphore);
    }
    pub fn signal_semaphore(&self, semaphore:vk::Semaphore){
        let mut signal_semaphores = self.wait_semaphores.borrow_mut();
        signal_semaphores.push(semaphore);
    }
}
impl Drop for CommandEncoder {
    fn drop(&mut self) {
        unsafe {
            let cbs = [self.command_buffer];
            let wait_semaphores = self.wait_semaphores.borrow();
            let signal_semaphores = self.signal_semaphores.borrow();
            let submit_info = vk::SubmitInfo::builder()
                .command_buffers(&cbs)
                .wait_semaphores(&wait_semaphores)
                .signal_semaphores(&signal_semaphores)
                .build();
            self.ctx.device.end_command_buffer(self.command_buffer).unwrap();
            let fence = self.fence.borrow();
            self.ctx.device
                .queue_submit(
                    self.queue,
                    &[submit_info],
                    (*fence).as_ref().map_or(vk::Fence::null(), |x| x.inner),
                )
                .unwrap();
        }
    }
}
impl std::ops::Deref for CommandEncoder {
    type Target = vk::CommandBuffer;
    fn deref(&self) -> &Self::Target {
        &self.command_buffer
    }
}
#[derive(Clone, Copy, Default)]
pub struct SbtRecord {
    pub buffer: vk::Buffer,
    pub offset: u64,
    pub stride: u64,
}
impl Kernel {
    pub fn new_rchit(
        ctx: &Context,
        rtx: &Rc<nv::RayTracing>,
        ray_gen: &[u32],
        hit: &[u32],
        miss: &[u32],
        layout: &Layout,
    ) -> Self {
        Self::new_rt_kernel(ctx, rtx, ray_gen, hit, miss, layout, false)
    }
    pub fn new_rahit(
        ctx: &Context,
        rtx: &Rc<nv::RayTracing>,
        ray_gen: &[u32],
        hit: &[u32],
        miss: &[u32],
        layout: &Layout,
    ) -> Self {
        Self::new_rt_kernel(ctx, rtx, ray_gen, hit, miss, layout, true)
    }
    fn new_rt_kernel(
        ctx: &Context,
        rtx: &Rc<nv::RayTracing>,
        ray_gen: &[u32],
        hit: &[u32],
        miss: &[u32],
        layout: &Layout,
        is_anyhit: bool,
    ) -> Self {
        unsafe {
            let allocation_callbacks = ctx.allocation_callbacks.as_ref();
            let raygen_module = ctx
                .device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::builder().code(ray_gen).build(),
                    ctx.allocation_callbacks.as_ref(),
                )
                .unwrap();

            let hit_module = ctx
                .device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::builder().code(hit).build(),
                    ctx.allocation_callbacks.as_ref(),
                )
                .unwrap();
            let miss_module = ctx
                .device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::builder().code(miss).build(),
                    ctx.allocation_callbacks.as_ref(),
                )
                .unwrap();
            let stage = vk::ShaderStageFlags::RAYGEN_NV
                | vk::ShaderStageFlags::CLOSEST_HIT_NV
                | vk::ShaderStageFlags::MISS_NV
                | vk::ShaderStageFlags::ANY_HIT_NV;
            let mut descriptor_set_layouts = vec![];
            let mut descriptor_sets = vec![];
            for set in &layout.sets {
                let descriptor_set_layout =
                    create_descriptor_set_layout(&ctx.device, allocation_callbacks, set, stage);
                descriptor_set_layouts.push(descriptor_set_layout);
            }
            let mut cache =
                DescriptorCache::new(ctx, &layout.sets, descriptor_set_layouts.clone(), stage);
            for (i, set) in layout.sets.iter().enumerate() {
                let descriptor_set = cache.allocate(i, set);
                descriptor_sets.push(descriptor_set);
            }

            let mut create_info =
                vk::PipelineLayoutCreateInfo::builder().set_layouts(&descriptor_set_layouts);
            let mut pc_ranges = vec![];
            let mut pc_range = None;
            if let Some(range) = layout.push_constants {
                pc_ranges.push(
                    vk::PushConstantRange::builder()
                        .offset(0)
                        .size(range as u32)
                        .stage_flags(stage)
                        .build(),
                );
                pc_range = Some(range as u32);

                create_info = create_info.push_constant_ranges(&pc_ranges);
            }
            let pipeline_layout = ctx
                .device
                .create_pipeline_layout(&create_info.build(), allocation_callbacks)
                .unwrap();
            let shader_groups = vec![
                // group0 = [ raygen ]
                vk::RayTracingShaderGroupCreateInfoNV::builder()
                    .ty(vk::RayTracingShaderGroupTypeNV::GENERAL)
                    .general_shader(0)
                    .closest_hit_shader(vk::SHADER_UNUSED_NV)
                    .any_hit_shader(vk::SHADER_UNUSED_NV)
                    .intersection_shader(vk::SHADER_UNUSED_NV)
                    .build(),
                // group1 = [ chit ]
                if is_anyhit {
                    vk::RayTracingShaderGroupCreateInfoNV::builder()
                        .ty(vk::RayTracingShaderGroupTypeNV::TRIANGLES_HIT_GROUP)
                        .general_shader(vk::SHADER_UNUSED_NV)
                        .any_hit_shader(1)
                        .closest_hit_shader(vk::SHADER_UNUSED_NV)
                        .intersection_shader(vk::SHADER_UNUSED_NV)
                        .build()
                } else {
                    vk::RayTracingShaderGroupCreateInfoNV::builder()
                        .ty(vk::RayTracingShaderGroupTypeNV::TRIANGLES_HIT_GROUP)
                        .general_shader(vk::SHADER_UNUSED_NV)
                        .closest_hit_shader(1)
                        .any_hit_shader(vk::SHADER_UNUSED_NV)
                        .intersection_shader(vk::SHADER_UNUSED_NV)
                        .build()
                },
                // group2 = [ miss ]
                vk::RayTracingShaderGroupCreateInfoNV::builder()
                    .ty(vk::RayTracingShaderGroupTypeNV::GENERAL)
                    .general_shader(2)
                    .closest_hit_shader(vk::SHADER_UNUSED_NV)
                    .any_hit_shader(vk::SHADER_UNUSED_NV)
                    .intersection_shader(vk::SHADER_UNUSED_NV)
                    .build(),
            ];
            let shader_stages = vec![
                vk::PipelineShaderStageCreateInfo::builder()
                    .stage(vk::ShaderStageFlags::RAYGEN_NV)
                    .module(raygen_module)
                    .name(std::ffi::CStr::from_bytes_with_nul(b"main\0").unwrap())
                    .build(),
                vk::PipelineShaderStageCreateInfo::builder()
                    .stage(if is_anyhit {
                        vk::ShaderStageFlags::ANY_HIT_NV
                    } else {
                        vk::ShaderStageFlags::CLOSEST_HIT_NV
                    })
                    .module(hit_module)
                    .name(std::ffi::CStr::from_bytes_with_nul(b"main\0").unwrap())
                    .build(),
                vk::PipelineShaderStageCreateInfo::builder()
                    .stage(vk::ShaderStageFlags::MISS_NV)
                    .module(miss_module)
                    .name(std::ffi::CStr::from_bytes_with_nul(b"main\0").unwrap())
                    .build(),
            ];
            let pipeline = rtx
                .create_ray_tracing_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::RayTracingPipelineCreateInfoNV::builder()
                        .stages(&shader_stages)
                        .groups(&shader_groups)
                        .max_recursion_depth(1)
                        .layout(pipeline_layout)
                        .build()],
                    None,
                )
                .unwrap()[0];
            let sbt = {
                let props_rt = nv::RayTracing::get_properties(&ctx.instance, ctx.pdevice);
                let group_count = 3; // Listed in vk::RayTracingPipelineCreateInfoNV
                let handle_size = props_rt.shader_group_handle_size as usize;
                let alignment = props_rt.shader_group_base_alignment as usize;
                let handle_size_aligned =
                    ((handle_size + alignment - 1) & !(alignment - 1)) as usize;
                let table_size = (handle_size_aligned * group_count) as usize;
                let mut table_data: Vec<u8> = vec![0u8; table_size];

                rtx.get_ray_tracing_shader_group_handles(
                    pipeline,
                    0,
                    group_count as u32,
                    &mut table_data,
                )
                .unwrap();
                // println!("{:?}", table_data);
                let mut table_data_aligned = vec![0u8; table_size];
                for i in 0..3 {
                    (&mut table_data_aligned
                        [handle_size_aligned * i..handle_size_aligned * i + handle_size])
                        .copy_from_slice(
                            &table_data[handle_size * i..handle_size * i + handle_size],
                        );
                }

                let sbt = TBuffer::<u8>::new(
                    ctx,
                    table_size as usize,
                    vk::BufferUsageFlags::TRANSFER_SRC,
                    vk::SharingMode::EXCLUSIVE,
                    vk::MemoryPropertyFlags::HOST_VISIBLE,
                );
                sbt.store(&table_data_aligned);
                sbt
            };
            Self {
                sbt: Some(sbt),
                pipeline,
                pipeline_layout,
                pc_range,
                shader_modules: vec![raygen_module, hit_module, miss_module],
                ctx: ctx.clone(),
                stage,
                descriptor_cache: RefCell::new(cache),
                descriptor_sets,
                descriptor_set_layouts,
                sets: layout.sets.clone(),
                rtx: Some(rtx.clone()),
            }
        }
    }
    pub fn new(ctx: &Context, spv: &[u32], stage: vk::ShaderStageFlags, layout: &Layout) -> Self {
        unsafe {
            let allocation_callbacks = ctx.allocation_callbacks.as_ref();
            let shader_module = ctx
                .device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::builder().code(spv).build(),
                    ctx.allocation_callbacks.as_ref(),
                )
                .unwrap();

            let mut descriptor_set_layouts = vec![];
            let mut descriptor_sets = vec![];
            for set in &layout.sets {
                let descriptor_set_layout =
                    create_descriptor_set_layout(&ctx.device, allocation_callbacks, set, stage);
                descriptor_set_layouts.push(descriptor_set_layout);
            }
            let mut cache =
                DescriptorCache::new(ctx, &layout.sets, descriptor_set_layouts.clone(), stage);
            for (i, set) in layout.sets.iter().enumerate() {
                let descriptor_set = cache.allocate(i, set);
                descriptor_sets.push(descriptor_set);
            }

            let mut create_info =
                vk::PipelineLayoutCreateInfo::builder().set_layouts(&descriptor_set_layouts);
            let mut pc_ranges = vec![];
            let mut pc_range = None;
            if let Some(range) = layout.push_constants {
                pc_ranges.push(
                    vk::PushConstantRange::builder()
                        .offset(0)
                        .size(range as u32)
                        .stage_flags(stage)
                        .build(),
                );
                pc_range = Some(range as u32);

                create_info = create_info.push_constant_ranges(&pc_ranges);
            }
            let pipeline_layout = ctx
                .device
                .create_pipeline_layout(&create_info.build(), allocation_callbacks)
                .unwrap();
            let pipeline = ctx
                .device
                .create_compute_pipelines(
                    ctx.pipeline_cache,
                    &[vk::ComputePipelineCreateInfo::builder()
                        .layout(pipeline_layout)
                        .stage(
                            vk::PipelineShaderStageCreateInfo::builder()
                                .stage(stage)
                                .module(shader_module)
                                .name(std::ffi::CStr::from_bytes_with_nul(b"main\0").unwrap())
                                .build(),
                        )
                        .build()],
                    ctx.allocation_callbacks.as_ref(),
                )
                .unwrap()[0];

            Self {
                pipeline_layout,
                pipeline,
                shader_modules: vec![shader_module],
                pc_range,
                ctx: ctx.clone(),
                stage,
                descriptor_set_layouts,
                descriptor_cache: RefCell::new(cache),
                descriptor_sets,
                sets: layout.sets.clone(),
                rtx: None,
                sbt: None,
            }
        }
    }
    pub fn bind(&mut self, set_idx: usize, set: &Set) {
        if *set != self.sets[set_idx] {
            let mut cache = self.descriptor_cache.borrow_mut();
            self.descriptor_sets[set_idx] = cache.allocate(set_idx, set);
            self.sets[set_idx] = set.clone();
        }
    }
    pub fn cmd_trace_rays<'a, T: bytemuck::Pod>(
        &self,
        command_encoder: &CommandEncoder,
        raygen_sbt: SbtRecord,
        miss_sbt: SbtRecord,
        hit_sbt: SbtRecord,
        callable_sbt: SbtRecord,
        width: u32,
        height: u32,
        depth: u32,
        push_constants: Option<T>,
    ) {
        unsafe {
            let rtx = self.rtx.as_ref().unwrap().as_ref();
            let bind_point = if self.stage.intersects(vk::ShaderStageFlags::RAYGEN_NV) {
                vk::PipelineBindPoint::RAY_TRACING_NV
            } else {
                panic!("")
            };
            self.bind_pipeline(**command_encoder, bind_point, push_constants);
            rtx.cmd_trace_rays(
                **command_encoder,
                raygen_sbt.buffer,
                raygen_sbt.offset,
                miss_sbt.buffer,
                miss_sbt.offset,
                miss_sbt.stride,
                hit_sbt.buffer,
                hit_sbt.offset,
                hit_sbt.stride,
                callable_sbt.buffer,
                callable_sbt.offset,
                callable_sbt.stride,
                width,
                height,
                depth,
            );
            let mut cache = self.descriptor_cache.borrow_mut();
            let fence = command_encoder.get_fence();
            cache.add_fence(fence);
        }
    }
    fn bind_pipeline<T: bytemuck::Pod>(
        &self,
        command_buffer: vk::CommandBuffer,
        bind_point: vk::PipelineBindPoint,
        push_constants: Option<T>,
    ) {
        let device = &self.ctx.device;
        unsafe {
            device.cmd_bind_pipeline(command_buffer, bind_point, self.pipeline);
            device.cmd_bind_descriptor_sets(
                command_buffer,
                bind_point,
                self.pipeline_layout,
                0,
                &self.descriptor_sets,
                &[],
            );
            match (self.pc_range, push_constants) {
                (Some(range), Some(val)) => {
                    if (range as usize) < std::mem::size_of::<T>() {
                        panic!("push constants size > range!!");
                    }
                    device.cmd_push_constants(
                        command_buffer,
                        self.pipeline_layout,
                        self.stage,
                        0,
                        bytemuck::cast_slice(&[val]),
                    );
                }
                _ => panic!("push constants not matching!"),
            }
        }
    }

    pub fn cmd_dispatch<T: bytemuck::Pod>(
        &self,
        command_encoder: &CommandEncoder,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
        push_constants: Option<T>,
    ) {
        let device = &self.ctx.device;
        unsafe {
            let bind_point = match self.stage {
                vk::ShaderStageFlags::COMPUTE => vk::PipelineBindPoint::COMPUTE,
                _ => panic!("stage should be compute but is {:?}", self.stage),
            };
            self.bind_pipeline(**command_encoder, bind_point, push_constants);
            device.cmd_dispatch(
                **command_encoder,
                group_count_x,
                group_count_y,
                group_count_z,
            );
            let mut cache = self.descriptor_cache.borrow_mut();
            let fence = command_encoder.get_fence();
            cache.add_fence(fence);
        }
    }
}
impl Drop for Kernel {
    fn drop(&mut self) {
        unsafe {
            let device = &self.ctx.device;
            let allocation_callbacks = self.ctx.allocation_callbacks.as_ref();
            for sm in &self.shader_modules {
                device.destroy_shader_module(*sm, allocation_callbacks);
            }
            for layout in &self.descriptor_set_layouts {
                device.destroy_descriptor_set_layout(*layout, allocation_callbacks);
            }
            device.destroy_pipeline_layout(self.pipeline_layout, allocation_callbacks);
            device.destroy_pipeline(self.pipeline, allocation_callbacks);
        }
    }
}
pub struct RayTracingKernel {
    ray_gen: Vec<u32>,
    hit: Vec<u32>,
    miss: Vec<u32>,
    imp: RefCell<Option<Kernel>>,
    ctx: Context,
    rtx: Rc<nv::RayTracing>,
}
impl RayTracingKernel {
    pub fn new_rchit(
        ctx: &Context,
        rtx: &Rc<nv::RayTracing>,
        ray_gen: &[u32],
        hit: &[u32],
        miss: &[u32],
    ) -> Self {
        let ray_gen: Vec<_> = ray_gen.iter().map(|x| *x).collect();
        let hit: Vec<_> = hit.iter().map(|x| *x).collect();
        let miss: Vec<_> = miss.iter().map(|x| *x).collect();
        Self {
            ray_gen,
            hit,
            miss,
            ctx: ctx.clone(),
            rtx: rtx.clone(),
            imp: RefCell::new(None),
        }
    }
    pub fn cmd_trace_rays<T: bytemuck::Pod>(
        &self,
        command_encoder: &CommandEncoder,
        raygen_sbt: SbtRecord,
        miss_sbt: SbtRecord,
        hit_sbt: SbtRecord,
        callable_sbt: SbtRecord,
        width: u32,
        height: u32,
        depth: u32,
        args: &KernelArgs<T>,
    ) {
        let mut imp = self.imp.borrow_mut();
        if imp.is_none() {
            *imp = Some(Kernel::new_rchit(
                &self.ctx,
                &self.rtx,
                &self.ray_gen,
                &self.hit,
                &self.miss,
                &Layout {
                    sets: args.sets.clone(),
                    push_constants: args
                        .push_constants
                        .clone()
                        .map_or(None, |_| Some(std::mem::size_of::<T>())),
                },
            ))
        } else {
            let imp = imp.as_mut().unwrap();
            assert!(args.sets.len() == imp.descriptor_sets.len());
            for i in 0..args.sets.len() {
                imp.bind(i, &args.sets[i]);
            }
        }
        imp.as_ref().unwrap().cmd_trace_rays(
            command_encoder,
            raygen_sbt,
            miss_sbt,
            hit_sbt,
            callable_sbt,
            width,
            height,
            depth,
            args.push_constants,
        );
    }
}
pub struct ComputeKernel {
    imp: RefCell<Option<Kernel>>,
    spv: Vec<u32>,
    ctx: Context,
}

impl ComputeKernel {
    pub fn new(ctx: &Context, spv: &[u32]) -> Self {
        let mut spv_v = vec![0; spv.len()];
        spv_v.copy_from_slice(spv);
        Self {
            spv: spv_v,
            imp: RefCell::new(None),
            ctx: ctx.clone(),
        }
    }
    pub fn cmd_dispatch<T: bytemuck::Pod>(
        &self,
        command_encoder: &CommandEncoder,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
        args: &KernelArgs<T>,
    ) {
        let mut imp = self.imp.borrow_mut();
        if imp.is_none() {
            *imp = Some(Kernel::new(
                &self.ctx,
                &self.spv,
                vk::ShaderStageFlags::COMPUTE,
                &Layout {
                    sets: args.sets.clone(),
                    push_constants: args
                        .push_constants
                        .clone()
                        .map_or(None, |_| Some(std::mem::size_of::<T>())),
                },
            ))
        } else {
            let imp = imp.as_mut().unwrap();
            assert!(args.sets.len() == imp.descriptor_sets.len());
            for i in 0..args.sets.len() {
                imp.bind(i, &args.sets[i]);
            }
        }
        imp.as_ref().unwrap().cmd_dispatch(
            command_encoder,
            group_count_x,
            group_count_y,
            group_count_z,
            args.push_constants,
        );
    }
}
