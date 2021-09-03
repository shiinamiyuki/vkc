use ash::vk;
#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use ash::vk;

    use crate::{
        align_to, Context, ContextCreateInfo, Extension, Profiler, TBuffer, TaskGraph,
        TaskGraphBuilder,
    };
    #[test]
    fn test_profiler() {
        let ctx = Context::new(ContextCreateInfo {
            enabled_extensions: &[],
            enable_validation: true,
        });
        let profiler = Profiler::new(&ctx, 1024);
    }
    #[test]
    fn test_utils() {
        assert_eq!(align_to(123, 16), 128);
        assert_eq!(align_to(80, 25), 100);
    }
    #[test]
    fn test_buffer_clone() {
        let ctx = Context::new(ContextCreateInfo {
            enabled_extensions: &[],
            enable_validation: true,
        });
        for i in 1..1024 {
            let buffer: TBuffer<usize> = TBuffer::new(
                &ctx,
                15 * i,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::TRANSFER_SRC,
                vk::SharingMode::EXCLUSIVE,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            );
            {
                let mapped = buffer.map_range_mut(.., vk::MemoryMapFlags::empty());
                let _ = mapped.slice.iter_mut().enumerate().map(|(i, x)| {
                    *x = i;
                });
            }
            let another = buffer.clone();
            {
                let mapped = another.map_range(.., vk::MemoryMapFlags::empty());
                let _ = mapped
                    .slice
                    .iter()
                    .enumerate()
                    .map(|(i, x)| assert_eq!(*x, i));
            }
        }
    }

    #[test]
    #[cfg(target_os = "windows")]
    fn test_ext_mem_create() {
        let ctx = Context::new(ContextCreateInfo {
            enabled_extensions: &[Extension::ExternalMemory],
            enable_validation: true,
        });
        let buffer: TBuffer<usize> = TBuffer::new(
            &ctx,
            1024,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC,
            vk::SharingMode::EXCLUSIVE,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
    }

    #[test]
    fn test_buffer_map() {
        let ctx = Context::new(ContextCreateInfo {
            enabled_extensions: &[],
            enable_validation: true,
        });
        for i in 1..1024 {
            let buffer: TBuffer<usize> = TBuffer::new(
                &ctx,
                15 * i,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::TRANSFER_SRC,
                vk::SharingMode::EXCLUSIVE,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            );
            {
                let mapped = buffer.map_range_mut(.., vk::MemoryMapFlags::empty());
                let _ = mapped.slice.iter_mut().enumerate().map(|(i, x)| {
                    *x = i;
                });
            }
            {
                let mapped = buffer.map_range(.., vk::MemoryMapFlags::empty());
                let _ = mapped
                    .slice
                    .iter()
                    .enumerate()
                    .map(|(i, x)| assert_eq!(*x, i));
            }
        }
    }
    #[test]
    fn test_buffer_alloc() {
        let ctx = Context::new(ContextCreateInfo {
            enabled_extensions: &[],
            enable_validation: true,
        });
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
    #[test]
    fn test_task_graph() {
        let ctx = Context::new(ContextCreateInfo {
            enabled_extensions: &[],
            enable_validation: true,
        });
        let graph = {
            let mut builder = TaskGraph::builder();
            let list: Rc<RefCell<Vec<i32>>> = Rc::new(RefCell::new(vec![]));
            let task1 = {
                let list = list.clone();
                builder.add(&[], move |_| {
                    let mut list = (*list).borrow_mut();
                    list.push(0);
                })
            };
            let task2 = {
                let list = list.clone();
                builder.add(&[], move |_| {
                    let mut list = (*list).borrow_mut();
                    list.push(1);
                })
            };
            let task3 = {
                let list = list.clone();
                builder.add(&[&task1], move |_| {
                    let mut list = (*list).borrow_mut();
                    list.push(2);
                })
            };
            let task4 = {
                let list = list.clone();
                builder.add(&[&task2, &task3], move |_| {
                    let mut list = (*list).borrow_mut();
                    list.push(3);
                })
            };
            builder.build()
        };
    }
}

fn align_to(x: vk::DeviceSize, alignment: vk::DeviceSize) -> u64 {
    ((x + alignment - 1) / alignment) * alignment
}

#[cfg(target_os = "windows")]
pub fn default_memory_handle_type() -> vk::ExternalMemoryHandleTypeFlags {
    vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32
}
#[cfg(target_os = "linux")]
pub fn default_memory_handle_type() -> vk::ExternalMemoryHandleTypeFlags {
    vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD
}

#[cfg(target_os = "windows")]
pub fn default_semaphore_handle_type() -> vk::ExternalSemaphoreHandleTypeFlags {
    vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_WIN32
}
#[cfg(target_os = "linux")]
pub fn default_semaphore_handle_type() -> vk::ExternalSemaphoreHandleTypeFlags {
    vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD
}
pub mod allocator;
pub mod ctx;
pub mod kernel;
pub mod profile;
pub mod resource;
pub mod sync;
pub mod task;

pub use ctx::*;
pub use kernel::*;
pub use profile::*;
pub use resource::*;
pub use sync::*;
pub use task::*;
