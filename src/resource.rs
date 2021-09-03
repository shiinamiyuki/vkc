use std::{
    ffi::c_void,
    marker::PhantomData,
    mem::align_of,
    ops::{Bound, RangeBounds},
};

use ash::vk;

use crate::{allocator::MemoryBlock, default_memory_handle_type};

use super::Context;

pub struct TBuffer<T: bytemuck::Pod> {
    pub ctx: Context,
    pub handle: vk::Buffer,
    pub size: usize,
    pub memory: Option<MemoryBlock>,
    pub mem_req: vk::MemoryRequirements,
    pub usage: vk::BufferUsageFlags,
    pub memory_property: vk::MemoryPropertyFlags,
    pub sharing_mode: vk::SharingMode,
    phantom: PhantomData<T>,
}
impl<T: bytemuck::Pod> Clone for TBuffer<T> {
    fn clone(&self) -> Self {
        let clone = TBuffer::<T>::new(
            &self.ctx,
            self.size,
            self.usage,
            self.sharing_mode,
            self.memory_property,
        );
        unsafe {
            let command_buffer = self
                .ctx
                .device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::builder()
                        .command_buffer_count(1)
                        .command_pool(self.ctx.pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .build(),
                )
                .unwrap()[0];
            copy_buffer_to_buffer(
                self.handle,
                0,
                clone.handle,
                0,
                (self.size * std::mem::size_of::<T>()) as vk::DeviceSize,
                command_buffer,
                &self.ctx,
            );
            self.ctx
                .device
                .free_command_buffers(self.ctx.pool, &[command_buffer]);
            clone
        }
    }
}
pub fn copy_buffer_to_buffer(
    src: vk::Buffer,
    src_offset: vk::DeviceSize,
    dst: vk::Buffer,
    dst_offset: vk::DeviceSize,
    size: vk::DeviceSize,
    command_buffer: vk::CommandBuffer,
    ctx: &Context,
) {
    begin_command_buffer(command_buffer, ctx);
    let region = vk::BufferCopy::builder()
        .src_offset(src_offset)
        .dst_offset(dst_offset)
        .size(size)
        .build();
    unsafe {
        ctx.device
            .cmd_copy_buffer(command_buffer, src, dst, &[region]);
    };
    end_command_buffer(command_buffer, ctx);
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
    pub slice: &'a [T],
    staging_buffer: Option<TBuffer<T>>,
    command_buffer: vk::CommandBuffer,
}
pub struct TBufferMapMut<'a, T: bytemuck::Pod> {
    parent: &'a TBuffer<T>,
    pub slice: &'a mut [T],
    staging_buffer: Option<TBuffer<T>>,
    command_buffer: vk::CommandBuffer,
}
struct TBufferMapRaw<'a, T: bytemuck::Pod> {
    parent: &'a TBuffer<T>,
    pub slice: &'a mut [T],
    staging_buffer: Option<TBuffer<T>>,
    command_buffer: vk::CommandBuffer,
}
impl<'a, T: bytemuck::Pod> Drop for TBufferMap<'a, T> {
    fn drop(&mut self) {
        unsafe {
            let ctx = &self.parent.ctx;
            if let Some(staging_buffer) = &self.staging_buffer {
                ctx.device
                    .unmap_memory(staging_buffer.memory.as_ref().unwrap().alloc_obj.memory);
                let command_buffer = self.command_buffer;
                ctx.device.free_command_buffers(ctx.pool, &[command_buffer]);
            } else {
                ctx.device
                    .unmap_memory(self.parent.memory.as_ref().unwrap().alloc_obj.memory);
            }
        }
    }
}

impl<'a, T: bytemuck::Pod> Drop for TBufferMapMut<'a, T> {
    fn drop(&mut self) {
        unsafe {
            let ctx = &self.parent.ctx;
            if let Some(staging_buffer) = &self.staging_buffer {
                ctx.device
                    .unmap_memory(staging_buffer.memory.as_ref().unwrap().alloc_obj.memory);
                let command_buffer = self.command_buffer;

                copy_buffer_to_buffer(
                    staging_buffer.handle,
                    0,
                    self.parent.handle,
                    0,
                    (self.parent.size * std::mem::size_of::<T>()) as vk::DeviceSize,
                    command_buffer,
                    ctx,
                );
                ctx.device.free_command_buffers(ctx.pool, &[command_buffer]);
            } else {
                ctx.device
                    .unmap_memory(self.parent.memory.as_ref().unwrap().alloc_obj.memory);
            }
        }
    }
}

impl<T> TBuffer<T>
where
    T: bytemuck::Pod,
{
    pub fn is_empty(&self) -> bool {
        self.memory.is_none()
    }
    pub fn new_external(
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
                memory: None,
                size,
                mem_req: vk::MemoryRequirements::default(),
                usage: usage,
                ctx: ctx.clone(),
                memory_property: memory_property_flags,
                sharing_mode,
            };
        }

        unsafe {
            let mut create_info = vk::BufferCreateInfo::builder()
                .size((size * std::mem::size_of::<T>()) as u64)
                .usage(usage)
                .sharing_mode(sharing_mode);

            let mut ext = vk::ExternalMemoryBufferCreateInfo::builder()
                .handle_types(default_memory_handle_type())
                .build();
            create_info = create_info.push_next(&mut ext);

            let handle = ctx
                .device
                .create_buffer(&create_info, ctx.allocation_callbacks.as_ref())
                .unwrap();

            let req = ctx.device.get_buffer_memory_requirements(handle);
            let memory_index =
                find_memorytype_index(&req, &ctx.device_memory_properties, memory_property_flags)
                    .unwrap();
            let memory_block = {
                let mut allocator = ctx.allocator.write().unwrap();
                let allocator = allocator.as_mut().unwrap();
                allocator.allocate(req.size, req.alignment, memory_index, true)
            };
            ctx.device
                .bind_buffer_memory(handle, memory_block.alloc_obj.memory, memory_block.offset)
                .unwrap();

            Self {
                handle,
                phantom: PhantomData {},
                memory: Some(memory_block),
                size,
                mem_req: req,
                usage,
                ctx: ctx.clone(),
                memory_property: memory_property_flags,
                sharing_mode,
            }
        }
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
                memory: None,
                size,
                mem_req: vk::MemoryRequirements::default(),
                usage: usage,
                ctx: ctx.clone(),
                memory_property: memory_property_flags,
                sharing_mode,
            };
        }

        unsafe {
            let create_info = vk::BufferCreateInfo::builder()
                .size((size * std::mem::size_of::<T>()) as u64)
                .usage(usage)
                .sharing_mode(sharing_mode);

            let handle = ctx
                .device
                .create_buffer(&create_info, ctx.allocation_callbacks.as_ref())
                .unwrap();

            let req = ctx.device.get_buffer_memory_requirements(handle);
            let memory_index =
                find_memorytype_index(&req, &ctx.device_memory_properties, memory_property_flags)
                    .unwrap();
            let memory_block = {
                let mut allocator = ctx.allocator.write().unwrap();
                let allocator = allocator.as_mut().unwrap();
                allocator.allocate(req.size, req.alignment, memory_index, false)
            };
            ctx.device
                .bind_buffer_memory(handle, memory_block.alloc_obj.memory, memory_block.offset)
                .unwrap();

            Self {
                handle,
                phantom: PhantomData {},
                memory: Some(memory_block),
                size,
                mem_req: req,
                usage,
                ctx: ctx.clone(),
                memory_property: memory_property_flags,
                sharing_mode,
            }
        }
    }
    pub fn store(&self, data: &[T]) {
        assert!(!self.is_empty());
        assert!(self.size == data.len());
        let mapped = self.map_range_mut(.., vk::MemoryMapFlags::empty());
        mapped.slice.copy_from_slice(data);
    }
    pub fn map_range_mut<'a, S: RangeBounds<vk::DeviceSize>>(
        &'a self,
        range: S,
        memory_map_flags: vk::MemoryMapFlags,
    ) -> TBufferMapMut<'a, T> {
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
        self.map_mut(start, end - start, memory_map_flags)
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
    fn map_raw<'a>(
        &'a self,
        offset: u64,
        len: u64,
        memory_map_flags: vk::MemoryMapFlags,
    ) -> TBufferMapRaw<'a, T> {
        assert!(!self.is_empty());
        unsafe {
            if !self
                .memory_property
                .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
            {
                let staging_buffer = TBuffer::<T>::new(
                    &self.ctx,
                    self.size,
                    vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
                    vk::SharingMode::EXCLUSIVE,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                );
                let ptr = self
                    .ctx
                    .device
                    .map_memory(
                        staging_buffer.memory.as_ref().unwrap().alloc_obj.memory,
                        staging_buffer.memory.as_ref().unwrap().offset + offset,
                        std::mem::size_of::<T>() as u64 * len,
                        memory_map_flags,
                    )
                    .unwrap();
                let command_buffer = self
                    .ctx
                    .device
                    .allocate_command_buffers(
                        &vk::CommandBufferAllocateInfo::builder()
                            .command_buffer_count(1)
                            .command_pool(self.ctx.pool)
                            .level(vk::CommandBufferLevel::PRIMARY)
                            .build(),
                    )
                    .unwrap()[0];

                copy_buffer_to_buffer(
                    self.handle,
                    0,
                    staging_buffer.handle,
                    0,
                    (self.size * std::mem::size_of::<T>()) as vk::DeviceSize,
                    command_buffer,
                    &self.ctx,
                );
                let slice = std::slice::from_raw_parts_mut(ptr as *mut T, len as usize);
                TBufferMapRaw::<'a> {
                    slice,
                    parent: &self,
                    staging_buffer: Some(staging_buffer),
                    command_buffer,
                }
            } else {
                let ptr = self
                    .ctx
                    .device
                    .map_memory(
                        self.memory.as_ref().unwrap().alloc_obj.memory,
                        self.memory.as_ref().unwrap().offset + offset,
                        std::mem::size_of::<T>() as u64 * len,
                        memory_map_flags,
                    )
                    .unwrap();
                let slice = std::slice::from_raw_parts_mut(ptr as *mut T, len as usize);
                TBufferMapRaw::<'a> {
                    slice,
                    parent: &self,
                    staging_buffer: None,
                    command_buffer: vk::CommandBuffer::null(),
                }
            }
        }
    }
    pub fn map_mut<'a>(
        &'a self,
        offset: u64,
        len: u64,
        memory_map_flags: vk::MemoryMapFlags,
    ) -> TBufferMapMut<'a, T> {
        let raw = self.map_raw(offset, len, memory_map_flags);
        TBufferMapMut::<'a> {
            slice: raw.slice,
            parent: raw.parent,
            staging_buffer: raw.staging_buffer,
            command_buffer: raw.command_buffer,
        }
    }
    pub fn map<'a>(
        &'a self,
        offset: u64,
        len: u64,
        memory_map_flags: vk::MemoryMapFlags,
    ) -> TBufferMap<'a, T> {
        let raw = self.map_raw(offset, len, memory_map_flags);
        TBufferMap::<'a> {
            slice: raw.slice,
            parent: raw.parent,
            staging_buffer: raw.staging_buffer,
            command_buffer: raw.command_buffer,
        }
    }
    // fn unmap(&self) {
    //     assert!(!self.is_empty());
    //     unsafe {
    //         self.ctx
    //             .device
    //             .unmap_memory(self.memory.as_ref().unwrap().alloc_obj.memory);
    //     }
    // }
}
impl<T: bytemuck::Pod> Drop for TBuffer<T> {
    fn drop(&mut self) {
        if self.memory.is_some() {
            unsafe {
                self.ctx
                    .device
                    .destroy_buffer(self.handle, self.ctx.allocation_callbacks.as_ref());
            }
            let mut allocator = self.ctx.allocator.write().unwrap();
            let allocator = allocator.as_mut().unwrap();
            allocator.free(self.memory.as_ref().unwrap());
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
pub fn transition_image_layout(
    image: vk::Image,
    format: vk::Format,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_stage_mask: vk::PipelineStageFlags,
    dst_stage_mask: vk::PipelineStageFlags,
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
                dst_stage_mask,
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
pub fn copy_buffer_to_image(
    buffer: vk::Buffer,
    buffer_offset: vk::DeviceSize,
    image: vk::Image,
    width: u32,
    height: u32,
    command_buffer: vk::CommandBuffer,
    ctx: &Context,
) {
    begin_command_buffer(command_buffer, ctx);
    let region = vk::BufferImageCopy::builder()
        .buffer_offset(buffer_offset)
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
    pub memory: MemoryBlock,
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
                let mapped = staging_buffer.map_range_mut(.., vk::MemoryMapFlags::empty());
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
            let memory_index = find_memorytype_index(
                &memory_requirements,
                &ctx.device_memory_properties,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .unwrap();
            let memory_block = {
                let mut allocator = ctx.allocator.write().unwrap();
                let allocator = allocator.as_mut().unwrap();
                allocator.allocate(
                    memory_requirements.size,
                    memory_requirements.alignment,
                    memory_index,
                    false,
                )
            };
            ctx.device
                .bind_image_memory(image, memory_block.alloc_obj.memory, memory_block.offset)
                .unwrap();
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
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                command_buffer,
                ctx,
            );
            copy_buffer_to_image(
                staging_buffer.handle,
                0,
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
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
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
                memory: memory_block,
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
        }
        let mut allocator = self.ctx.allocator.write().unwrap();
        let allocator = allocator.as_mut().unwrap();
        allocator.free(&self.memory);
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
