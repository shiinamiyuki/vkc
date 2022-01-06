# vkc
Simple wrapper for Vulkan compute pipeline


## Example

```rust
use vkc::{Context, include_spv,resource::TBuffer};
let Context::new_compute_only(vkc::ContextCreateInfo {
    enable_validation: true,
    enabled_extensions: &[],
};
let buffer: TBuffer<u32> = TBuffer::new(
    ctx,
    1024usize,
    vk::BufferUsageFlags::STORAGE_BUFFER
        | vk::BufferUsageFlags::TRANSFER_SRC
        | vk::BufferUsageFlags::TRANSFER_DST,
    vk::SharingMode::EXCLUSIVE,
    vk::MemoryPropertyFlags::DEVICE_LOCAL,
);
let kernel = vkc::ComputeKernel::new(ctx, &include_spv!("kernel.spv"));
let command_buffer = ...;
let command_encoder = vkc::CommandEncoder::new(
      &ctx.device,
      command_buffer,
      ctx.queue,
      &vk::CommandBufferBeginInfo::builder()
          .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
);

kernel.cmd_dispatch(&command_encoder,1024/256,1,1, &vkc::KernerlArgs{
  sets:vec![
    vkc::Set::Bindings(vec![
      buffer.handle
    ])
  ], 
  push_constants:Some((1,2,3,4))
});
let fence = command_encoder.get_fence();
std::mem::drop(command_encoder);
fence.wait();

```
