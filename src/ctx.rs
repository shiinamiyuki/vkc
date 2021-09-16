use std::{
    borrow::Cow,
    cell::RefCell,
    ffi::{c_void, CStr, CString},
    os::raw::c_char,
    process::abort,
    sync::{Arc, RwLock},
};

use ash::{
    extensions::{
        ext::{DebugReport, DebugUtils},
        khr::{Surface, Swapchain},
        nv::RayTracing,
    },
    vk, Entry,
};
use ash_window::create_surface;

use crate::allocator::GPUAllocator;

// pub struct ContextInner {
//     pub device: ash::Device,
//     pub instance: ash::Instance,
//     pub device_memory_properties: vk::PhysicalDeviceMemoryProperties,
//     pub allocation_callbacks: *const vk::AllocationCallbacks,
//     pub pool: vk::CommandPool,
//     pub queue: vk::Queue,
//     pub pdevice: vk::PhysicalDevice,
//     pub debug_utils_loader: DebugUtils,
//     pub debug_call_back: vk::DebugUtilsMessengerEXT,
// }
#[derive(Clone)]
pub struct Context {
    inner: Arc<ContextInner>,
}
impl std::ops::Deref for Context {
    type Target = ContextInner;
    fn deref(&self) -> &Self::Target {
        self.inner.as_ref()
    }
}
fn extension_names() -> Vec<*const i8> {
    vec![
        DebugUtils::name().as_ptr(),
        // DebugReport::name().as_ptr(),
        vk::KhrGetPhysicalDeviceProperties2Fn::name().as_ptr(),
    ]
}
unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message_id_number: i32 = callback_data.message_id_number as i32;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };
    println!(
        "{:?}:\n{:?} [{} ({})] : {}\n",
        message_severity,
        message_type,
        message_id_name,
        &message_id_number.to_string(),
        message,
    );
    if message_severity == vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
        panic!();
    }
    vk::FALSE
}
// unsafe extern "system" fn vulkan_debug_callback(
//     _: vk::DebugReportFlagsEXT,
//     _: vk::DebugReportObjectTypeEXT,
//     _: u64,
//     _: usize,
//     _: i32,
//     _: *const c_char,
//     p_message: *const c_char,
//     _: *mut c_void,
// ) -> u32 {
//     println!("{:?}", CStr::from_ptr(p_message));
//     vk::FALSE
// }
pub struct ContextInner {
    pub entry: Entry,
    pub instance: ash::Instance,
    pub device: ash::Device,
    pub extensions: Vec<Extension>,
    pub pipeline_cache: vk::PipelineCache,
    pub surface_loader: Option<Surface>,
    // pub swapchain_loader: Swapchain,
    // pub debug_report_loader: DebugReport,
    // pub window: winit::Window,
    // pub events_loop: RefCell<winit::EventsLoop>,
    // pub debug_call_back: vk::DebugReportCallbackEXT,
    pub debug_utils_loader: DebugUtils,
    pub debug_call_back: vk::DebugUtilsMessengerEXT,

    pub pdevice: vk::PhysicalDevice,
    pub device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub queue_family_index: u32,
    pub queue: vk::Queue,

    // pub surface: vk::SurfaceKHR,
    // pub surface_format: vk::SurfaceFormatKHR,
    // pub surface_resolution: vk::Extent2D,

    // pub swapchain: vk::SwapchainKHR,
    // pub present_images: Vec<vk::Image>,
    pub pool: vk::CommandPool,
    // pub command_buffer: vk::CommandBuffer,
    // pub present_complete_semaphore: vk::Semaphore,
    // pub rendering_complete_semaphore: vk::Semaphore,

    // pub window_width: u32,
    // pub window_height: u32,
    pub allocation_callbacks: *const vk::AllocationCallbacks,
    pub limits: vk::PhysicalDeviceLimits,
    pub allocator: RwLock<Option<GPUAllocator>>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Extension {
    ShaderAtomicFloat,
    ExternalMemory,
    // TimelineSemaphore, // Default enabled
}

pub struct ContextCreateInfo<'a> {
    pub enabled_extensions: &'a [Extension],
    pub enable_validation: bool,
}
impl ContextInner {
    fn new_1(
        info: ContextCreateInfo<'_>,
        window: Option<&dyn raw_window_handle::HasRawWindowHandle>,
    ) -> Self {
        unsafe {
            // let hidpi_factor: f64 = window.get_hidpi_factor();
            // let physical_dimensions = logical_dimensions.to_physical(hidpi_factor);

            let entry = Entry::new().unwrap();
            let app_name = CString::new("Rust_VK_RT_HLSL").unwrap();

            let mut layer_names = vec![];
            if info.enable_validation {
                layer_names.push(CString::new("VK_LAYER_KHRONOS_validation").unwrap());
            }
            // let layer_names: Vec<CString> = vec![];
            let layers_names_raw: Vec<*const i8> = layer_names
                .iter()
                .map(|raw_name| raw_name.as_ptr())
                .collect();

            let mut extension_names_raw = extension_names();
            let surface_extensions = if window.is_some() {
                ash_window::enumerate_required_extensions(window.unwrap()).unwrap()
            } else {
                vec![]
            };
            for e in &surface_extensions {
                extension_names_raw.push(e.as_ptr());
            }

            let appinfo = vk::ApplicationInfo::builder()
                .application_name(&app_name)
                .application_version(0)
                .engine_name(&app_name)
                .engine_version(0)
                .api_version(vk::make_api_version(0, 1, 2, 0));
            let mut validation_features = vk::ValidationFeaturesEXT::builder()
                .enabled_validation_features(&[vk::ValidationFeatureEnableEXT::DEBUG_PRINTF])
                .build();
            let mut create_info = vk::InstanceCreateInfo::builder()
                .application_info(&appinfo)
                .enabled_layer_names(&layers_names_raw)
                .enabled_extension_names(&extension_names_raw);
            if info.enable_validation {
                create_info = create_info.push_next(&mut validation_features);
            }
            let instance: ash::Instance = entry
                .create_instance(&create_info, None)
                .expect("Instance creation error");

            // let debug_info = vk::DebugReportCallbackCreateInfoEXT::builder()
            //     .flags(
            //         vk::DebugReportFlagsEXT::ERROR
            //             | vk::DebugReportFlagsEXT::INFORMATION
            //             | vk::DebugReportFlagsEXT::WARNING
            //             | vk::DebugReportFlagsEXT::PERFORMANCE_WARNING,
            //     )
            //     .pfn_callback(Some(vulkan_debug_callback));
            let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
                )
                .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
                .pfn_user_callback(Some(vulkan_debug_callback));
            // let debug_report_loader = DebugReport::new(&entry, &instance);
            // let debug_call_back = debug_report_loader
            //     .create_debug_report_callback(&debug_info, None)
            //     .unwrap();
            let debug_utils_loader = DebugUtils::new(&entry, &instance);
            let debug_call_back = debug_utils_loader
                .create_debug_utils_messenger(&debug_info, None)
                .unwrap();

            let pdevices = instance
                .enumerate_physical_devices()
                .expect("Physical device error");
            let surface =
                window.map(|window| create_surface(&entry, &instance, window, None).unwrap());
            let surface_loader = window.map(|_| Surface::new(&entry, &instance));
            let (pdevice, queue_family_index) = pdevices
                .iter()
                .map(|pdevice| {
                    instance
                        .get_physical_device_queue_family_properties(*pdevice)
                        .iter()
                        .enumerate()
                        .filter_map(|(index, ref info)| {
                            let supports = if window.is_some() {
                                info.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                                    && surface_loader.as_ref().unwrap()
                                        .get_physical_device_surface_support(
                                            *pdevice,
                                            index as u32,
                                            surface.unwrap(),
                                        )
                                        .unwrap()
                            } else {
                                info.queue_flags.contains(vk::QueueFlags::COMPUTE)
                            };
                            match supports {
                                true => Some((*pdevice, index)),
                                _ => None,
                            }
                        })
                        .nth(0)
                })
                .filter_map(|v| v)
                .nth(0)
                .expect("Couldn't find suitable device.");
            let queue_family_index = queue_family_index as u32;

            let mut device_extension_names_raw = vec![
                // Swapchain::name().as_ptr(),
                RayTracing::name().as_ptr(),
                vk::ExtDescriptorIndexingFn::name().as_ptr(),
                vk::ExtScalarBlockLayoutFn::name().as_ptr(),
                vk::KhrGetMemoryRequirements2Fn::name().as_ptr(),
                vk::KhrShaderNonSemanticInfoFn::name().as_ptr(),
                // vk::ExtShaderAtomicFloatFn::name().as_ptr(),
                // vk::KhrExternalMemoryWin32Fn::name().as_ptr(),
            ];
            if window.is_some() {
                device_extension_names_raw.push(Swapchain::name().as_ptr());
            }
            if info
                .enabled_extensions
                .contains(&Extension::ShaderAtomicFloat)
            {
                device_extension_names_raw.push(vk::ExtShaderAtomicFloatFn::name().as_ptr());
            }
            if info.enabled_extensions.contains(&Extension::ExternalMemory) {
                device_extension_names_raw.push(vk::KhrExternalMemoryWin32Fn::name().as_ptr());
            }
            let priorities = [1.0];

            let queue_info = [vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(queue_family_index)
                .queue_priorities(&priorities)
                .build()];

            let mut descriptor_indexing =
                vk::PhysicalDeviceDescriptorIndexingFeaturesEXT::builder()
                    .descriptor_binding_variable_descriptor_count(true)
                    .runtime_descriptor_array(true)
                    .build();

            let mut scalar_block = vk::PhysicalDeviceScalarBlockLayoutFeaturesEXT::builder()
                .scalar_block_layout(true)
                .build();

            let mut features2 = vk::PhysicalDeviceFeatures2::default();
            instance
                .fp_v1_1()
                .get_physical_device_features2(pdevice, &mut features2);
            let mut robustness2features = vk::PhysicalDeviceRobustness2FeaturesEXT::builder()
                .null_descriptor(true)
                .build();
            let mut query_reset_features = vk::PhysicalDeviceHostQueryResetFeatures::builder()
                .host_query_reset(true)
                .build();
            let mut shader_atomic_float = vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT::builder()
                .shader_buffer_float32_atomic_add(true)
                .shader_buffer_float32_atomics(true)
                .build();
            let mut timeline_semaphore = vk::PhysicalDeviceTimelineSemaphoreFeatures::builder()
                .timeline_semaphore(true)
                .build();
            let device_create_info = {
                let mut builder = vk::DeviceCreateInfo::builder()
                    .queue_create_infos(&queue_info)
                    .enabled_extension_names(&device_extension_names_raw)
                    .enabled_features(&features2.features)
                    .push_next(&mut scalar_block)
                    .push_next(&mut descriptor_indexing)
                    .push_next(&mut robustness2features)
                    .push_next(&mut query_reset_features);
                if info
                    .enabled_extensions
                    .contains(&Extension::ShaderAtomicFloat)
                {
                    builder = builder.push_next(&mut shader_atomic_float);
                }
                // if info
                //     .enabled_extensions
                //     .contains(&Extension::TimelineSemaphore)
                // {
                builder = builder.push_next(&mut timeline_semaphore);
                // }
                builder.build()
            };

            let device: ash::Device = instance
                .create_device(pdevice, &device_create_info, None)
                .unwrap();

            let present_queue = device.get_device_queue(queue_family_index as u32, 0);

            // let surface_formats = surface_loader
            //     .get_physical_device_surface_formats(pdevice, surface)
            //     .unwrap();
            // let surface_format = surface_formats
            //     .iter()
            //     .map(|sfmt| match sfmt.format {
            //         vk::Format::UNDEFINED => vk::SurfaceFormatKHR {
            //             format: vk::Format::B8G8R8_UNORM,
            //             color_space: sfmt.color_space,
            //         },
            //         _ => sfmt.clone(),
            //     })
            //     .nth(0)
            //     .expect("Unable to find suitable surface format.");
            // let surface_capabilities = surface_loader
            //     .get_physical_device_surface_capabilities(pdevice, surface)
            //     .unwrap();
            // let mut desired_image_count = surface_capabilities.min_image_count + 1;
            // if surface_capabilities.max_image_count > 0
            //     && desired_image_count > surface_capabilities.max_image_count
            // {
            //     desired_image_count = surface_capabilities.max_image_count;
            // }
            // let surface_resolution = match surface_capabilities.current_extent.width {
            //     std::u32::MAX => vk::Extent2D {
            //         width: window_width,
            //         height: window_height,
            //     },
            //     _ => surface_capabilities.current_extent,
            // };
            // let pre_transform = if surface_capabilities
            //     .supported_transforms
            //     .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
            // {
            //     vk::SurfaceTransformFlagsKHR::IDENTITY
            // } else {
            //     surface_capabilities.current_transform
            // };
            // let present_modes = surface_loader
            //     .get_physical_device_surface_present_modes(pdevice, surface)
            //     .unwrap();
            // let present_mode = present_modes
            //     .iter()
            //     .cloned()
            //     .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
            //     .unwrap_or(vk::PresentModeKHR::FIFO);
            // let swapchain_loader = Swapchain::new(&instance, &device);

            // let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
            //     .surface(surface)
            //     .min_image_count(desired_image_count)
            //     .image_color_space(surface_format.color_space)
            //     .image_format(surface_format.format)
            //     .image_extent(surface_resolution.clone())
            //     .image_usage(vk::ImageUsageFlags::TRANSFER_DST)
            //     .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            //     .pre_transform(pre_transform)
            //     .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            //     .present_mode(present_mode)
            //     .clipped(true)
            //     .image_array_layers(1);

            // let swapchain = swapchain_loader
            //     .create_swapchain(&swapchain_create_info, None)
            //     .unwrap();

            let pool_create_info = vk::CommandPoolCreateInfo::builder()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(queue_family_index);

            let pool = device.create_command_pool(&pool_create_info, None).unwrap();

            // let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
            //     .command_buffer_count(1)
            //     .command_pool(pool)
            //     .level(vk::CommandBufferLevel::PRIMARY);

            // let command_buffer = device
            //     .allocate_command_buffers(&command_buffer_allocate_info)
            //     .unwrap()[0];

            // let present_images = swapchain_loader.get_swapchain_images(swapchain).unwrap();
            let device_memory_properties = instance.get_physical_device_memory_properties(pdevice);

            // let semaphore_create_info = vk::SemaphoreCreateInfo::default();

            // let present_complete_semaphore = device
            //     .create_semaphore(&semaphore_create_info, None)
            //     .unwrap();
            // let rendering_complete_semaphore = device
            //     .create_semaphore(&semaphore_create_info, None)
            //     .unwrap();
            let limits = {
                let props = instance.get_physical_device_properties(pdevice);
                props.limits
                // println!("{:?}", limits);
            };
            let pipeline_cache = device
                .create_pipeline_cache(&vk::PipelineCacheCreateInfo::builder().build(), None)
                .unwrap();
            ContextInner {
                // events_loop: RefCell::new(events_loop),
                entry,
                instance,
                device,
                queue_family_index,
                pdevice,
                device_memory_properties,
                extensions: info.enabled_extensions.iter().map(|x| *x).collect(),
                // window,
                surface_loader,
                // surface_format,
                // present_queue,
                // surface_resolution,
                // swapchain_loader,
                // swapchain,
                // present_images,
                queue: present_queue,
                pool,
                // command_buffer,
                // present_complete_semaphore,
                // rendering_complete_semaphore,
                // surface,
                debug_call_back,
                debug_utils_loader,
                pipeline_cache,
                allocation_callbacks: std::ptr::null(),
                limits,
                allocator: RwLock::new(None),
                // window_width: physical_dimensions.width as u32,
                // window_height: physical_dimensions.height as u32,
            }
        }
    }
}

impl Drop for ContextInner {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.device
                .destroy_pipeline_cache(self.pipeline_cache, None);
            // self.device
            //     .destroy_semaphore(self.present_complete_semaphore, None);
            // self.device
            //     .destroy_semaphore(self.rendering_complete_semaphore, None);
            self.device.destroy_command_pool(self.pool, None);
            // self.swapchain_loader
            // .destroy_swapchain(self.swapchain, None);
            self.device.destroy_device(None);
            self.debug_utils_loader
                .destroy_debug_utils_messenger(self.debug_call_back, None);
            // self.surface_loader.destroy_surface(self.surface, None);
            // self.debug_report_loader
            // .destroy_debug_report_callback(self.debug_call_back, None);
            self.instance.destroy_instance(None);
        }
    }
}

// impl Drop for ContextInner {
//     fn drop(&mut self) {
//         unsafe {
//             self.device.device_wait_idle().unwrap();
//             self.device.destroy_device(None);
//             self.debug_utils_loader
//                 .destroy_debug_utils_messenger(self.debug_call_back, None);
//             self.instance.destroy_instance(None);
//         }
//     }
// }
impl Context {
    pub fn new_compute_only(info: ContextCreateInfo<'_>) -> Self {
        let ctx = Self {
            inner: Arc::new(ContextInner::new_1(info, None)),
        };
        let allocator = GPUAllocator::new(&ctx);
        {
            let mut a = ctx.inner.allocator.write().unwrap();
            *a = Some(allocator);
        }
        ctx
    }
    pub fn new_graphics(
        info: ContextCreateInfo<'_>,
        window: &dyn raw_window_handle::HasRawWindowHandle,
    ) -> Self {
        let ctx = Self {
            inner: Arc::new(ContextInner::new_1(info, Some(window))),
        };
        let allocator = GPUAllocator::new(&ctx);
        {
            let mut a = ctx.inner.allocator.write().unwrap();
            *a = Some(allocator);
        }
        ctx
    }
}

#[macro_export]
macro_rules! include_spv {
    ($path:literal) => {{
        let bytes = include_bytes!($path);
        ash::util::read_spv(&mut std::io::Cursor::new(&bytes[..])).unwrap()
    }};
}

pub fn create_descriptor_set_from_storage_buffers(
    device: &ash::Device,
    descriptor_pool: vk::DescriptorPool,
    buffers: &[vk::Buffer],
    stage: vk::ShaderStageFlags,
    allocation_callbacks: *const vk::AllocationCallbacks,
) -> (vk::DescriptorSetLayout, vk::DescriptorSet) {
    unsafe {
        let flags = vec![vk::DescriptorBindingFlagsEXT::empty(); buffers.len()];
        let mut binding_flags = vk::DescriptorSetLayoutBindingFlagsCreateInfoEXT::builder()
            .binding_flags(&flags)
            .build();
        let bindings: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, _buffer)| {
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(i as u32)
                    .descriptor_count(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .stage_flags(stage)
                    .build()
            })
            .collect();
        let descriptor_set_layout = device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::builder()
                    .bindings(&bindings)
                    .push_next(&mut binding_flags)
                    .build(),
                allocation_callbacks.as_ref(),
            )
            .unwrap();
        let descriptor_set = device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::builder()
                    .set_layouts(&[descriptor_set_layout])
                    .descriptor_pool(descriptor_pool)
                    .build(),
            )
            .unwrap()[0];
        let buffer_infos: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(_i, buffer)| {
                vk::DescriptorBufferInfo::builder()
                    .buffer(*buffer)
                    .range(vk::WHOLE_SIZE)
                    .build()
            })
            .collect();
        for i in 0..buffers.len() {
            let info = [buffer_infos[i]];
            let write = vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .dst_binding(i as u32)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&info)
                .build();
            device.update_descriptor_sets(&[write], &[]);
        }

        (descriptor_set_layout, descriptor_set)
    }
}
