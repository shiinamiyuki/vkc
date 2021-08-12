use std::{
    ffi::c_void,
    marker::PhantomData,
    mem::align_of,
    ops::{Bound, RangeBounds},
};

use ash::{
    vk,
};

use super::Context;

pub struct TBuffer<T: bytemuck::Pod> {
    pub ctx: Context,
    pub handle: vk::Buffer,
    pub size: usize,
    pub memory: vk::DeviceMemory,
    pub mem_req: vk::MemoryRequirements,
    pub usage: vk::BufferUsageFlags,
    phantom: PhantomData<T>,
    is_null: bool,
}

pub fn find_memorytype_index(
    memory_req: &vk::MemoryRequirements,
    memory_prop: &vk::PhysicalDeviceMemoryProperties,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    memory_prop.memory_types[..memory_prop.memory_type_count as _]
        .iter()
        .enumerate()
        .find(|(index, memory_type)| {
            (1 << index) & memory_req.memory_type_bits != 0
                && memory_type.property_flags & flags == flags
        })
        .map(|(index, _memory_type)| index as _)
}
pub struct TBufferMap<'a, T: bytemuck::Pod> {
    parent: &'a TBuffer<T>,
    pub slice: &'a mut [T],
}
impl<'a, T: bytemuck::Pod> Drop for TBufferMap<'a, T> {
    fn drop(&mut self) {
        self.parent.unmap();
    }
}

impl<T> TBuffer<T>
where
    T: bytemuck::Pod,
{
    pub fn is_empty(&self) -> bool {
        self.is_null
    }
    pub fn new(
        ctx: &Context,
        size: usize,
        usage: vk::BufferUsageFlags,
        sharing_mode: vk::SharingMode,
        memory_property_flags: vk::MemoryPropertyFlags,
    ) -> Self {
        if size == 0 {
            return Self {
                handle: vk::Buffer::null(),
                phantom: PhantomData {},
                memory: vk::DeviceMemory::null(),
                size,
                mem_req: vk::MemoryRequirements::default(),
                usage: usage,
                ctx: ctx.clone(),
                is_null: true,
            };
        }
        let create_info = vk::BufferCreateInfo::builder()
            .size((size * std::mem::size_of::<T>()) as u64)
            .usage(usage)
            .sharing_mode(sharing_mode);
        unsafe {
            let handle = ctx
                .device
                .create_buffer(&create_info, ctx.allocation_callbacks.as_ref())
                .unwrap();

            let req = ctx.device.get_buffer_memory_requirements(handle);
            let memory_index =
                find_memorytype_index(&req, &ctx.device_memory_properties, memory_property_flags)
                    .unwrap();
            let index_allocate_info = vk::MemoryAllocateInfo {
                allocation_size: req.size,
                memory_type_index: memory_index,
                ..Default::default()
            };
            let memory = ctx
                .device
                .allocate_memory(&index_allocate_info, ctx.allocation_callbacks.as_ref())
                .unwrap();
            ctx.device.bind_buffer_memory(handle, memory, 0).unwrap();

            Self {
                handle,
                phantom: PhantomData {},
                memory,
                size,
                mem_req: req,
                usage: usage,
                ctx: ctx.clone(),
                is_null: false,
            }
        }
    }
    pub fn store(&self, data: &[T]) {
        assert!(!self.is_empty());
        unsafe {
            assert!(self.size == data.len());
            // let mapped = self.map(0, self.size as u64, vk::MemoryMapFlags::empty());
            // let mut mapped_slice = Align::new(
            //     mapped.slice.as_mut_ptr() as *mut c_void,
            //     align_of::<T>() as u64,
            //     self.size as u64,
            // );
            // mapped_slice.copy_from_slice(&data);
            let mapped = self.map_range(.., vk::MemoryMapFlags::empty());
            mapped.slice.copy_from_slice(data);
        }
    }
    pub fn map_range<'a, S: RangeBounds<vk::DeviceSize>>(
        &'a self,
        range: S,
        memory_map_flags: vk::MemoryMapFlags,
    ) -> TBufferMap<'a, T> {
        assert!(!self.is_empty());
        let start: vk::DeviceSize = match range.start_bound() {
            Bound::Included(x) => *x,
            Bound::Excluded(x) => x + 1,
            Bound::Unbounded => 0,
        };
        let end: vk::DeviceSize = match range.end_bound() {
            Bound::Included(x) => x + 1,
            Bound::Excluded(x) => *x,
            Bound::Unbounded => self.size as vk::DeviceSize,
        };
        self.map(start, end - start, memory_map_flags)
    }
    pub fn map<'a>(
        &'a self,
        offset: u64,
        len: u64,
        memory_map_flags: vk::MemoryMapFlags,
    ) -> TBufferMap<'a, T> {
        assert!(!self.is_empty());
        unsafe {
            let ptr = self
                .ctx
                .device
                .map_memory(
                    self.memory,
                    offset,
                    std::mem::size_of::<T>() as u64 * len,
                    memory_map_flags,
                )
                .unwrap();
            let slice = std::slice::from_raw_parts_mut(ptr as *mut T, len as usize);
            TBufferMap::<'a> {
                slice,
                parent: &self,
            }
        }
    }
    fn unmap(&self) {
        assert!(!self.is_empty());
        unsafe {
            self.ctx.device.unmap_memory(self.memory);
        }
    }
}
impl<T: bytemuck::Pod> Drop for TBuffer<T> {
    fn drop(&mut self) {
        if !self.is_null {
            unsafe {
                self.ctx
                    .device
                    .destroy_buffer(self.handle, self.ctx.allocation_callbacks.as_ref());
                self.ctx
                    .device
                    .free_memory(self.memory, self.ctx.allocation_callbacks.as_ref());
            }
        }
    }
}
// pub struct TBufferView<T: bytemuck::Pod> {
//     pub handle: vk::BufferView,
//     phantom: PhantomData<T>,
// }
fn begin_command_buffer(command_buffer: vk::CommandBuffer, ctx: &Context) {
    unsafe {
        ctx.device
            .begin_command_buffer(
                command_buffer,
                &vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .unwrap();
    }
}
fn end_command_buffer(command_buffer: vk::CommandBuffer, ctx: &Context) {
    unsafe {
        ctx.device.end_command_buffer(command_buffer).unwrap();
        ctx.device
            .queue_submit(
                ctx.queue,
                &[vk::SubmitInfo::builder()
                    .command_buffers(&[command_buffer])
                    .build()],
                vk::Fence::null(),
            )
            .unwrap();
        match ctx.device.queue_wait_idle(ctx.queue) {
            Ok(_) => {}
            Err(err) => {
                println!("GPU ERROR {:?}", err);
                panic!("GPU ERROR");
            }
        }
    }
}
fn transition_image_layout(
    image: vk::Image,
    format: vk::Format,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    command_buffer: vk::CommandBuffer,
    ctx: &Context,
) {
    unsafe {
        begin_command_buffer(command_buffer, ctx);
        let (src_access_mask, dst_access_mask, src_stage_mask, dst_stage_mask) = if old_layout
            == vk::ImageLayout::UNDEFINED
            && new_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL
        {
            (
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
            )
        } else if old_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL
            && new_layout == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        {
            (
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER
                    | vk::PipelineStageFlags::RAY_TRACING_SHADER_NV,
            )
        } else {
            panic!("unsupported layout transition");
        };
        let barrier = vk::ImageMemoryBarrier::builder()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(
                vk::ImageSubresourceRange::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_array_layer(0)
                    .layer_count(1)
                    .base_mip_level(0)
                    .level_count(1)
                    .build(),
            )
            .src_access_mask(src_access_mask)
            .dst_access_mask(dst_access_mask)
            .build();

        ctx.device.cmd_pipeline_barrier(
            command_buffer,
            src_stage_mask,
            dst_stage_mask,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
        end_command_buffer(command_buffer, ctx);
    }
}
fn copy_buffer_to_image(
    buffer: vk::Buffer,
    image: vk::Image,
    width: u32,
    height: u32,
    command_buffer: vk::CommandBuffer,
    ctx: &Context,
) {
    begin_command_buffer(command_buffer, ctx);
    let region = vk::BufferImageCopy::builder()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(
            vk::ImageSubresourceLayers::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1)
                .build(),
        )
        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
        .image_extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .build();
    unsafe {
        ctx.device.cmd_copy_buffer_to_image(
            command_buffer,
            buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        )
    };
    end_command_buffer(command_buffer, ctx);
}
pub struct Image {
    pub handle: vk::Image,
    pub view: vk::ImageView,
    pub ctx: Context,
    pub memory: vk::DeviceMemory,
    pub mem_req: vk::MemoryRequirements,
}
impl Image {
    pub fn from_data(ctx: &Context, data: &[u8], extent: vk::Extent2D, format: vk::Format) -> Self {
        unsafe {
            let staging_buffer = TBuffer::<u8>::new(
                ctx,
                data.len(),
                vk::BufferUsageFlags::TRANSFER_SRC,
                vk::SharingMode::EXCLUSIVE,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );
            {
                let mapped = staging_buffer.map_range(.., vk::MemoryMapFlags::empty());
                mapped.slice.copy_from_slice(data);
            }
            let image = ctx
                .device
                .create_image(
                    &vk::ImageCreateInfo::builder()
                        .image_type(vk::ImageType::TYPE_2D)
                        .extent(vk::Extent3D {
                            width: extent.width,
                            height: extent.height,
                            depth: 1,
                        })
                        .mip_levels(1)
                        .array_layers(1)
                        .format(format)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .initial_layout(vk::ImageLayout::UNDEFINED)
                        .usage(
                            vk::ImageUsageFlags::TRANSFER_DST
                                | vk::ImageUsageFlags::SAMPLED
                                | vk::ImageUsageFlags::STORAGE,
                        )
                        .samples(vk::SampleCountFlags::TYPE_1),
                    ctx.allocation_callbacks.as_ref(),
                )
                .unwrap();
            let memory_requirements = ctx.device.get_image_memory_requirements(image);
            let alloc_info = vk::MemoryAllocateInfo::builder()
                .allocation_size(memory_requirements.size)
                .memory_type_index(
                    find_memorytype_index(
                        &memory_requirements,
                        &ctx.device_memory_properties,
                        vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    )
                    .unwrap(),
                )
                .build();
            let memory = ctx
                .device
                .allocate_memory(&alloc_info, ctx.allocation_callbacks.as_ref())
                .unwrap();
            ctx.device.bind_image_memory(image, memory, 0);
            let command_buffer = ctx
                .device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::builder()
                        .command_buffer_count(1)
                        .command_pool(ctx.pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .build(),
                )
                .unwrap()[0];
            transition_image_layout(
                image,
                format,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                command_buffer,
                ctx,
            );
            copy_buffer_to_image(
                staging_buffer.handle,
                image,
                extent.width,
                extent.height,
                command_buffer,
                ctx,
            );
            transition_image_layout(
                image,
                format,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                command_buffer,
                ctx,
            );
            ctx.device.free_command_buffers(ctx.pool, &[command_buffer]);
            let view = ctx
                .device
                .create_image_view(
                    &vk::ImageViewCreateInfo::builder()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(format)
                        .subresource_range(
                            vk::ImageSubresourceRange::builder()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .base_mip_level(0)
                                .level_count(1)
                                .base_array_layer(0)
                                .layer_count(1)
                                .build(),
                        ),
                    ctx.allocation_callbacks.as_ref(),
                )
                .unwrap();
            Self {
                handle: image,
                ctx: ctx.clone(),
                mem_req: memory_requirements,
                memory,
                view,
            }
        }
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        unsafe {
            self.ctx
                .device
                .destroy_image_view(self.view, self.ctx.allocation_callbacks.as_ref());
            self.ctx
                .device
                .destroy_image(self.handle, self.ctx.allocation_callbacks.as_ref());
            self.ctx
                .device
                .free_memory(self.memory, self.ctx.allocation_callbacks.as_ref());
        }
    }
}

pub struct Sampler {
    pub handle: vk::Sampler,
    ctx: Context,
}
impl Sampler {
    pub fn new(ctx: &Context) -> Self {
        let props = unsafe { ctx.instance.get_physical_device_properties(ctx.pdevice) };
        let info = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(true)
            .max_anisotropy(props.limits.max_sampler_anisotropy)
            .border_color(vk::BorderColor::FLOAT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.0)
            .max_lod(0.0)
            .min_lod(0.0)
            .build();
        let sampler = unsafe {
            ctx.device
                .create_sampler(&info, ctx.allocation_callbacks.as_ref())
        }
        .unwrap();
        Self {
            handle: sampler,
            ctx: ctx.clone(),
        }
    }
}
impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe {
            self.ctx
                .device
                .destroy_sampler(self.handle, self.ctx.allocation_callbacks.as_ref());
        }
    }
}
