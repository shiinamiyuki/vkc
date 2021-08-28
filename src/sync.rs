use std::sync::{atomic::AtomicU64, Arc};

use ash::vk;

use crate::{CommandEncoder, Context};

pub struct Fence {
    pub(crate) inner: vk::Fence,
    device: *const ash::Device,
}
impl Fence {
    pub fn new(fence: vk::Fence, device: &ash::Device) -> Self {
        Self {
            inner: fence,
            device: device as *const ash::Device,
        }
    }
    pub fn wait(&self) {
        unsafe {
            while !(*self.device).get_fence_status(self.inner).unwrap() {
                (*self.device)
                    .wait_for_fences(&[self.inner], true, u64::MAX)
                    .unwrap();
            }
        }
    }
    pub fn handle(&self) -> vk::Fence {
        self.inner
    }
}
impl Drop for Fence {
    fn drop(&mut self) {
        unsafe {
            self.wait();
            (*self.device).destroy_fence(self.inner, None);
        }
    }
}

// different from VkEvent
// similar to cuda event
pub struct Event {
    inner: Arc<SemaphoreInner>,
    wait_value: u64,
}
impl Event {
    pub fn new(semaphore: &Semaphore) -> Self {
        Self {
            inner: semaphore.inner.clone(),
            wait_value: semaphore
                .inner
                .timeline
                .fetch_add(2, std::sync::atomic::Ordering::SeqCst),
        }
    }
    pub fn signal(&self) {
        unsafe {
            let device = &self.inner.ctx.device;
            let signal_info = vk::SemaphoreSignalInfo::builder()
                .semaphore(self.inner.handle)
                .value(self.wait_value + 1)
                .build();
            device.signal_semaphore(&signal_info).unwrap();
        }
    }
    pub fn wait(&self, timeout: u64) {
        unsafe {
            let device = &self.inner.ctx.device;
            let wait_semaphores = [self.inner.handle];
            let wait_values = [self.wait_value];
            let wait_info = vk::SemaphoreWaitInfo::builder()
                .semaphores(&wait_semaphores)
                .values(&wait_values)
                .build();
            device.wait_semaphores(&wait_info, timeout).unwrap();
        }
    }
}
impl Drop for Event {
    fn drop(&mut self) {
        self.wait(u64::MAX);
    }
}

struct SemaphoreInner {
    handle: vk::Semaphore,
    timeline: AtomicU64,
    ctx: Context,
}
impl Drop for SemaphoreInner {
    fn drop(&mut self) {
        unsafe {
            self.ctx
                .device
                .destroy_semaphore(self.handle, self.ctx.allocation_callbacks.as_ref());
        }
    }
}
pub struct Semaphore {
    inner: Arc<SemaphoreInner>,
}
impl Semaphore {
    pub fn handle(&self) -> vk::Semaphore {
        self.inner.handle
    }
    pub fn new(ctx: &Context) -> Self {
        let mut type_create_info = vk::SemaphoreTypeCreateInfo::builder()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0)
            .build();
        let create_info = vk::SemaphoreCreateInfo::builder()
            .push_next(&mut type_create_info)
            .build();
        unsafe {
            let semaphore = ctx
                .device
                .create_semaphore(&create_info, ctx.allocation_callbacks.as_ref())
                .unwrap();
            Self {
                inner: Arc::new(SemaphoreInner {
                    handle: semaphore,
                    timeline: AtomicU64::new(0),
                    ctx: ctx.clone(),
                }),
            }
        }
    }
}
