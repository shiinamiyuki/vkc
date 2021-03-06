use ash::vk;
use std::collections::BTreeSet;
use std::ffi::c_void;
use std::{cell::RefCell, sync::Arc};

use crate::{align_to, default_memory_handle_type, Context};

pub struct MemoryAllocateInfo {
    pub size: vk::DeviceSize,
    pub usage: vk::BufferUsageFlags,
    pub sharing_mode: vk::SharingMode,
    pub memory_property_flags: vk::MemoryPropertyFlags,
}
pub struct MemoryObject {
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
    pub memory_index: u32,
    ctx: Context,
}
impl MemoryObject {
    fn new(
        ctx: &Context,
        size: vk::DeviceSize,
        memory_index: u32,
        external_memory: bool,
        device_address: bool,
    ) -> Self {
        unsafe {
            let mut allocate_info = vk::MemoryAllocateInfo::builder()
                .allocation_size(size)
                .memory_type_index(memory_index);
            let mut export_info =
                vk::ExportMemoryAllocateInfo::builder().handle_types(default_memory_handle_type());
            if external_memory {
                assert!(ctx.extensions.contains(&crate::Extension::ExternalMemory));
                allocate_info = allocate_info.push_next(&mut export_info);
            }
            let mut flags = vk::MemoryAllocateFlagsInfo::builder()
                .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS)
                .build();
            if device_address {
                assert!(ctx
                    .extensions
                    .contains(&crate::Extension::VulkanMemoryModel));
                allocate_info = allocate_info.push_next(&mut flags);
            }
            let memory = ctx
                .device
                .allocate_memory(&allocate_info, ctx.allocation_callbacks.as_ref())
                .unwrap();
            Self {
                memory,
                size,
                memory_index,
                ctx: ctx.clone(),
            }
        }
    }
}
impl Drop for MemoryObject {
    fn drop(&mut self) {
        unsafe {
            self.ctx
                .device
                .free_memory(self.memory, self.ctx.allocation_callbacks.as_ref());
        }
    }
}
pub type GPUAllocatorPtr = Arc<RefCell<GPUAllocator>>;
#[derive(Clone)]
pub struct MemoryBlock {
    pub alloc_obj: Arc<MemoryObject>,
    pub capacity: vk::DeviceSize,
    pub offset: vk::DeviceSize, // [offset..offset+size] is used
    pub start: vk::DeviceSize,  // this is the begining of this block
    pub size: vk::DeviceSize,
}
impl std::cmp::PartialEq for MemoryBlock {
    fn eq(&self, other: &Self) -> bool {
        let p1 = self.alloc_obj.as_ref() as *const MemoryObject;
        let p2 = other.alloc_obj.as_ref() as *const MemoryObject;
        if p1 != p2 {
            return false;
        }
        self.offset == other.offset && self.size == other.size
    }
}
impl std::cmp::Eq for MemoryBlock {}
impl std::cmp::PartialOrd for MemoryBlock {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let ordering = self.size.partial_cmp(&other.size)?;
        let p1 = self.alloc_obj.as_ref() as *const MemoryObject;
        let p2 = other.alloc_obj.as_ref() as *const MemoryObject;
        let ordering = ordering.then(p1.partial_cmp(&p2)?);
        let ordering = ordering.then(self.capacity.partial_cmp(&other.capacity)?);
        let ordering = ordering.then(self.offset.partial_cmp(&other.offset)?);
        if ordering == std::cmp::Ordering::Equal {
            assert!(self.eq(other));
        }
        Some(ordering)
    }
}
impl std::cmp::Ord for MemoryBlock {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}
impl MemoryBlock {}

// very naive allocator
pub struct GPUAllocator {
    availables: BTreeSet<MemoryBlock>,
    used: BTreeSet<MemoryBlock>,
    ctx: Context,
}
impl GPUAllocator {
    pub fn new(ctx: &Context) -> Self {
        Self {
            ctx: ctx.clone(),
            availables: BTreeSet::new(),
            used: BTreeSet::new(),
        }
    }
    pub fn free(&mut self, block: &MemoryBlock) {
        let inside = self.used.remove(block);
        assert!(inside);
        self.availables.insert(block.clone());
    }
    pub fn allocate(
        &mut self,
        size: vk::DeviceSize,
        alignment: vk::DeviceSize,
        memory_type_index: u32,
        external_memory: bool,
        device_address: bool,
    ) -> MemoryBlock {
        let mut chosen: Option<MemoryBlock> = None;
        let mut splitted: Option<MemoryBlock> = None;
        let mut to_remove: Option<MemoryBlock> = None;

        for block in &self.availables {
            let mut good = true;
            let aligned_size = if block.start % alignment == 0 {
                size
            } else {
                size + alignment - (block.start % alignment)
            };
            good &= (block.capacity - block.start) >= aligned_size;
            good &= block.alloc_obj.memory_index == memory_type_index;
            if good {
                // chosen = Some(block.clone());
                to_remove = Some(block.clone());
                let new_offset = block.start + aligned_size;
                chosen = Some(MemoryBlock {
                    alloc_obj: block.alloc_obj.clone(),
                    capacity: block.capacity,
                    start: block.start,
                    offset: align_to(block.start, alignment),
                    size: aligned_size,
                });
                if block.size > aligned_size {
                    splitted = Some(MemoryBlock {
                        alloc_obj: block.alloc_obj.clone(),
                        capacity: block.capacity,
                        offset: new_offset,
                        start: new_offset,
                        size: block.size - aligned_size,
                    })
                }
            }
        }
        if let Some(splitted) = splitted {
            self.availables.insert(splitted);
        }
        if let Some(to_remove) = to_remove {
            let inside = self.availables.remove(&to_remove);
            assert!(inside);
        }
        if chosen.is_none() {
            let obj = MemoryObject::new(
                &self.ctx,
                size.max(65536),
                memory_type_index,
                external_memory,
                device_address
            );
            let size = obj.size;
            let block = MemoryBlock {
                alloc_obj: Arc::new(obj),
                size,
                start: 0,
                offset: 0,
                capacity: size,
            };
            chosen = Some(block);
        }
        self.used.insert(chosen.clone().unwrap());
        chosen.unwrap()
    }
}
