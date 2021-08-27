use ash::vk;

use crate::CommandEncoder;

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