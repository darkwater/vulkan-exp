use std::collections::HashSet;
use std::iter::FromIterator;
use std::sync::Arc;
use vulkano::device::Device;
use vulkano::device::DeviceExtensions;
use vulkano::device::Features;
use vulkano::device::Queue;
use vulkano::instance::ApplicationInfo;
use vulkano::instance::Instance;
use vulkano::instance::InstanceExtensions;
use vulkano::instance::PhysicalDevice;
use vulkano::instance::Version;
use vulkano::instance::debug::DebugCallback;
use vulkano::instance::debug::MessageTypes;
use vulkano::swapchain::Surface;
use vulkano_win::VkSurfaceBuild;
use winit::Event;
use winit::EventsLoop;
use winit::Window;
use winit::WindowBuilder;
use winit::WindowEvent;
use winit::dpi::LogicalSize;
use winit::os::unix::WindowBuilderExt;
use { vulkano, vulkano_win };

const WIDTH:  u32 = 800;
const HEIGHT: u32 = 600;

const VALIDATION_LAYERS: &[&str] = &[
    "VK_LAYER_LUNARG_standard_validation",
];

#[cfg(not(release))]
const ENABLE_VALIDATION_LAYERS: bool = true;
#[cfg(release)]
const ENABLE_VALIDATION_LAYERS: bool = false;

struct QueueFamilyIndices {
    graphics_family: i32,
    present_family:  i32,
}

impl QueueFamilyIndices {
    fn new() -> Self {
        Self {
            graphics_family: -1,
            present_family:  -1,
        }
    }

    fn is_complete(&self) -> bool {
        self.graphics_family >= 0
            && self.present_family >= 0
    }
}

struct HelloTriangleApplication {
    instance: Arc<Instance>,
    debug_callback: Option<DebugCallback>,

    events_loop: EventsLoop,
    surface: Arc<Surface<Window>>,

    physical_device_index: usize,
    device: Arc<Device>,

    graphics_queue: Arc<Queue>,
    present_queue: Arc<Queue>,
}

impl HelloTriangleApplication {
    pub fn initialize() -> Self {
        let instance       = Self::create_instance();
        let debug_callback = Self::setup_debug_callback(&instance);

        let (events_loop, surface) = Self::create_surface(&instance);

        let physical_device_index                   = Self::pick_physical_device(&instance, &surface);
        let (device, graphics_queue, present_queue) = Self::create_logical_device(&instance, &surface, physical_device_index);

        Self {
            instance,
            debug_callback,

            events_loop,
            surface,

            physical_device_index,
            device,

            graphics_queue,
            present_queue,
        }
    }

    fn create_instance() -> Arc<Instance> {
        if ENABLE_VALIDATION_LAYERS && !Self::check_validation_layer_support() {
            eprintln!("Validation layers requested, but not available!")
        }

        let supported_extensions = InstanceExtensions::supported_by_core().expect("supported extensions");
        dbg!(supported_extensions);

        let version = Version {
            major: env!("CARGO_PKG_VERSION_MAJOR").parse().unwrap(),
            minor: env!("CARGO_PKG_VERSION_MINOR").parse().unwrap(),
            patch: env!("CARGO_PKG_VERSION_PATCH").parse().unwrap(),
        };

        let app_info = ApplicationInfo {
            application_name:    Some("Hello Vulkan".into()),
            application_version: Some(version),
            engine_name:         Some("Engineless".into()),
            engine_version:      Some(version),
        };

        let required_extensions = Self::get_required_extensions();
        if ENABLE_VALIDATION_LAYERS && Self::check_validation_layer_support() {
            Instance::new(Some(&app_info), &required_extensions, VALIDATION_LAYERS.iter().cloned()).expect("create instance")
        } else {
            Instance::new(Some(&app_info), &required_extensions, None).expect("create instance")
        }
    }

    fn check_validation_layer_support() -> bool {
        let layers: Vec<_> = vulkano::instance::layers_list().unwrap()
            .map(|l| l.name().to_string()).collect();

        VALIDATION_LAYERS.iter().all(|l| layers.contains(&l.to_string()))
    }

    fn get_required_extensions() -> InstanceExtensions {
        let mut extensions = vulkano_win::required_extensions();

        if ENABLE_VALIDATION_LAYERS {
            // TODO: Should be ext_debug_utils but vulkano doesn't support that yet
            extensions.ext_debug_report = true;
        }

        extensions
    }

    fn setup_debug_callback(instance: &Arc<Instance>) -> Option<DebugCallback> {
        if !ENABLE_VALIDATION_LAYERS {
            return None;
        }

        let msg_types = MessageTypes {
            error: true,
            warning: true,
            performance_warning: true,
            information: true,
            debug: true,
        };

        DebugCallback::new(&instance, msg_types, |msg| {
            eprintln!("[VAL] {}", msg.description);
        }).ok()
    }

    fn create_surface(instance: &Arc<Instance>) -> (EventsLoop, Arc<Surface<Window>>) {
        let events_loop = EventsLoop::new();
        let surface = WindowBuilder::new()
            .with_class("vulkan".to_string(), "vulkan".to_string())
            .with_title("Vulkan")
            .with_dimensions(LogicalSize::new(WIDTH as f64, HEIGHT as f64));

        let surface = surface
            .build_vk_surface(&events_loop, instance.clone())
            .expect("window surface");

        (events_loop, surface)
    }

    fn pick_physical_device(instance: &Arc<Instance>, surface: &Arc<Surface<Window>>) -> usize {
        PhysicalDevice::enumerate(&instance)
            .position(|device| Self::is_device_suitable(surface, &device))
            .expect("suitable GPU")
    }

    fn is_device_suitable(surface: &Arc<Surface<Window>>, device: &PhysicalDevice) -> bool {
        let indices = Self::find_queue_families(surface, device);
        indices.is_complete()
    }

    fn find_queue_families(surface: &Arc<Surface<Window>>, device: &PhysicalDevice) -> QueueFamilyIndices {
        let mut indices = QueueFamilyIndices::new();

        for (id, queue_family) in device.queue_families().enumerate() {
            if queue_family.supports_graphics() {
                indices.graphics_family = id as i32;
            }

            if surface.is_supported(queue_family).unwrap() {
                indices.present_family = id as i32;
            }

            if indices.is_complete() {
                break;
            }
        }

        indices
    }

    fn create_logical_device(
        instance: &Arc<Instance>,
        surface: &Arc<Surface<Window>>,
        physical_device_index: usize,
    ) -> (Arc<Device>, Arc<Queue>, Arc<Queue>) {
        let physical_device = PhysicalDevice::from_index(&instance, physical_device_index).unwrap();
        let indices = Self::find_queue_families(&surface, &physical_device);

        let families = [ indices.graphics_family, indices.present_family ];
        let unique_queue_families: HashSet<&i32> = HashSet::from_iter(families.iter());

        let queue_priority = 1.0;
        let queue_families = unique_queue_families.iter().map(|i| {
            (physical_device.queue_families().nth(**i as usize).unwrap(), queue_priority)
        });

        let (device, mut queues) = Device::new(
            physical_device,
            &Features::none(),
            &DeviceExtensions::none(),
            queue_families
        )
        .expect("failed to create logical device!");

        let graphics_queue = queues.next().unwrap();
        let present_queue = queues.next().unwrap_or_else(|| graphics_queue.clone());

        (device, graphics_queue, present_queue)
    }

    fn main_loop(&mut self) {
        loop {
            let mut done = false;

            self.events_loop.poll_events(|ev| {
                match ev {
                    Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                        done = true;
                    },
                    _ => (),
                }
            });

            if done {
                return;
            }
        }
    }
}

fn main() {
    let mut app = HelloTriangleApplication::initialize();
    app.main_loop();
}
