use ash::vk;
#[cfg(test)]
mod tests {
    use ash::vk;

    use crate::{Context, TBuffer, align_to};
    #[test]
    fn test_utils() {
        assert_eq!(align_to(123,16), 128);
        assert_eq!(align_to(80, 25), 100);
    }
    #[test]
    fn test_buffer_alloc() {
        let ctx = Context::new(true);
        let mut buffers = vec![];
        for i in 0..1024 {
            let buffer: TBuffer<[f32; 3]> = TBuffer::new(
                &ctx,
                15 * i,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::SharingMode::EXCLUSIVE,
                vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
            );
            buffers.push(buffer);
        }
    }
}

fn align_to(x: vk::DeviceSize, alignment: vk::DeviceSize) -> u64 {
    ((x + alignment - 1) / alignment) * alignment
}
pub mod allocator;
pub mod ctx;
pub mod kernel;
pub mod resource;

pub use ctx::*;
pub use kernel::*;
pub use resource::*;
