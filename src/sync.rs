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
impl SemaphoreInner {
    fn wait(&self, values: &[(vk::Semaphore, u64)], timeout: u64) -> bool {
        unsafe {
            let device = &self.ctx.device;
            let wait_semaphores: Vec<_> = values.iter().map(|x| x.0).collect();
            let wait_values: Vec<_> = values.iter().map(|x| x.1).collect();
            let wait_info = vk::SemaphoreWaitInfo::builder()
                .semaphores(&wait_semaphores)
                .values(&wait_values)
                .build();
            match device.wait_semaphores(&wait_info, timeout) {
                Ok(_) => true,
                Err(e) => match e {
                    vk::Result::TIMEOUT => false,
                    _ => panic!("{}", e),
                },
            }
        }
    }
    fn signal(&self, value: u64) {
        unsafe {
            let device = &self.ctx.device;
            let signal_info = vk::SemaphoreSignalInfo::builder()
                .semaphore(self.handle)
                .value(value)
                .build();
            device.signal_semaphore(&signal_info).unwrap();
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
    pub fn wait(&self, values: &[(vk::Semaphore, u64)], timeout: u64) -> bool {
        self.inner.wait(values, timeout)
    }
    pub fn signal(&self, value: u64) {
        self.inner.signal(value)
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

pub struct TimePoint {
    semaphore: Arc<SemaphoreInner>,
    value: u64,
}
impl TimePoint {
    pub fn wait(&self, timeout: u64) -> bool {
        self.semaphore
            .wait(&[(self.semaphore.handle, self.value)], timeout)
    }
    pub fn signal(&self) {
        self.semaphore.signal(self.value)
    }
}

pub fn create_timeline(semaphore: &Semaphore, n_timepoints: u64) -> Vec<TimePoint> {
    let value = semaphore
        .inner
        .timeline
        .fetch_add(n_timepoints, std::sync::atomic::Ordering::SeqCst);
    (0..value)
        .map(|v| TimePoint {
            semaphore: semaphore.inner.clone(),
            value: v,
        })
        .collect()
}
