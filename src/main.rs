// use vulkano::pipeline::GraphicsPipelineAbstract;
use std::collections::HashSet;
use std::ffi::CString;
use std::iter::FromIterator;
use std::ops::Deref;
use std::sync::Arc;
use vulkano::SynchronizedVulkanObject;
use vulkano::VulkanObject;
use vulkano::command_buffer::AutoCommandBuffer;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::DynamicState;
use vulkano::descriptor::PipelineLayoutAbstract;
use vulkano::device::Device;
use vulkano::device::DeviceExtensions;
use vulkano::device::Features;
use vulkano::device::Queue;
use vulkano::device::RawDeviceExtensions;
use vulkano::format::Format;
use vulkano::framebuffer::Framebuffer;
use vulkano::framebuffer::FramebufferAbstract;
use vulkano::framebuffer::RenderPassAbstract;
use vulkano::framebuffer::Subpass;
use vulkano::image::ImageUsage;
use vulkano::image::swapchain::SwapchainImage;
use vulkano::image::traits::ImageAccess;
use vulkano::instance::ApplicationInfo;
use vulkano::instance::Instance;
use vulkano::instance::InstanceExtensions;
use vulkano::instance::PhysicalDevice;
use vulkano::instance::RawInstanceExtensions;
use vulkano::instance::Version;
use vulkano::instance::debug::DebugCallback;
use vulkano::instance::debug::MessageTypes;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::vertex::BufferlessDefinition;
use vulkano::pipeline::vertex::BufferlessVertices;
use vulkano::pipeline::viewport::Viewport;
use vulkano::swapchain::AcquireError;
use vulkano::swapchain::Capabilities;
use vulkano::swapchain::ColorSpace;
use vulkano::swapchain::CompositeAlpha;
use vulkano::swapchain::PresentMode;
use vulkano::swapchain::SupportedPresentModes;
use vulkano::swapchain::Surface;
use vulkano::swapchain::Swapchain;
use vulkano::sync::GpuFuture;
use vulkano::sync::SharingMode;
use vulkano_win::VkSurfaceBuild;
use winit::Event;
use winit::EventsLoop;
use winit::Window;
use winit::WindowBuilder;
use winit::WindowEvent;
use winit::dpi::LogicalSize;
use winit::os::unix::WindowBuilderExt;
use { openvr, vulkano, vulkano_win };

mod vertex_shader;
mod fragment_shader;

const WIDTH:  u32 = 800;
const HEIGHT: u32 = 600;

const VALIDATION_LAYERS: &[&str] = &[
    "VK_LAYER_LUNARG_standard_validation",
];

#[cfg(not(release))]
const ENABLE_VALIDATION_LAYERS: bool = true;
#[cfg(release)]
const ENABLE_VALIDATION_LAYERS: bool = false;

/// Required device extensions
fn device_extensions() -> RawDeviceExtensions {
    let extensions = DeviceExtensions {
        khr_swapchain: true,
        ..vulkano::device::DeviceExtensions::none()
    };

    let mut extensions: RawDeviceExtensions = (&extensions).into();
    for n in vec![
        "VK_KHR_external_memory_fd",
        "VK_KHR_external_semaphore_fd",
        "VK_KHR_external_memory",
        "VK_KHR_external_semaphore",
        "VK_KHR_dedicated_allocation",
        "VK_KHR_get_memory_requirements2"
    ] {
        extensions.insert(CString::new(n).unwrap());
    }

    extensions
}

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

type ConcreteGraphicsPipeline = GraphicsPipeline<BufferlessDefinition,
                                                 Box<dyn PipelineLayoutAbstract + Send + Sync + 'static>,
                                                 Arc<dyn RenderPassAbstract + Send + Sync + 'static>>;

struct HelloTriangleApplication {
    instance: Arc<Instance>,
    debug_callback: Option<DebugCallback>,

    events_loop: EventsLoop,
    surface: Arc<Surface<Window>>,

    physical_device_index: usize,
    device: Arc<Device>,

    graphics_queue: Arc<Queue>,
    present_queue: Arc<Queue>,

    swap_chain: Arc<Swapchain<Window>>,
    swap_chain_images: Vec<Arc<SwapchainImage<Window>>>,

    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    graphics_pipeline: Arc<ConcreteGraphicsPipeline>,
    // graphics_pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,

    swap_chain_framebuffers: Vec<Arc<dyn FramebufferAbstract + Send + Sync>>,

    command_buffers: Vec<Arc<AutoCommandBuffer>>,

    previous_frame_end: Option<Box<GpuFuture>>,
    recreate_swap_chain: bool,

    vr: VrSubsystem,
}

struct VrSubsystem {
    context:    openvr::Context,
    system:     openvr::System,
    compositor: openvr::Compositor,
}

impl HelloTriangleApplication {
    pub fn new() -> Self {
        let instance       = Self::create_instance();
        let debug_callback = Self::setup_debug_callback(&instance);

        let (events_loop, surface) = Self::create_surface(&instance);

        let physical_device_index                   = Self::pick_physical_device(&instance, &surface);
        let (device, graphics_queue, present_queue) = Self::create_logical_device(&instance, &surface, physical_device_index);

        let vr = Self::create_vr_context();

        let (swap_chain, swap_chain_images) = Self::create_swap_chain(&instance, &surface, physical_device_index,
                                                                      &device, &graphics_queue, &present_queue, None);

        let render_pass       = Self::create_render_pass(&device, swap_chain.format());
        let graphics_pipeline = Self::create_graphics_pipeline(&device, swap_chain.dimensions(), &render_pass);

        let swap_chain_framebuffers = Self::create_framebuffers(&swap_chain_images, &render_pass);

        let previous_frame_end = Some(Self::create_sync_objects(&device));

        let mut app = Self {
            instance,
            debug_callback,

            events_loop,
            surface,

            physical_device_index,
            device,

            graphics_queue,
            present_queue,

            swap_chain,
            swap_chain_images,

            render_pass,
            graphics_pipeline,

            swap_chain_framebuffers,

            command_buffers: vec![],

            previous_frame_end,
            recreate_swap_chain: false,

            vr,
        };

        app.create_command_buffers();

        app
    }

    fn create_vr_context() -> VrSubsystem {
        let context    = unsafe { openvr::init(openvr::ApplicationType::Scene).expect("openvr context") };
        let system     = context.system().expect("openvr system");
        let compositor = context.compositor().expect("openvr compositor");

        VrSubsystem {
            context,
            system,
            compositor,
        }
    }

    fn create_instance() -> Arc<Instance> {
        if ENABLE_VALIDATION_LAYERS && !Self::check_validation_layer_support() {
            eprintln!("Validation layers requested, but not available!")
        }

        // let supported_extensions = InstanceExtensions::supported_by_core().expect("supported extensions");

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
            Instance::new(Some(&app_info), required_extensions, VALIDATION_LAYERS.iter().cloned()).expect("create instance")
        } else {
            Instance::new(Some(&app_info), required_extensions, None).expect("create instance")
        }
    }

    fn check_validation_layer_support() -> bool {
        let layers: Vec<_> = vulkano::instance::layers_list().unwrap()
            .map(|l| l.name().to_string()).collect();

        VALIDATION_LAYERS.iter().all(|l| layers.contains(&l.to_string()))
    }

    fn get_required_extensions() -> RawInstanceExtensions {
        let mut extensions = vulkano_win::required_extensions();

        if ENABLE_VALIDATION_LAYERS {
            // TODO: Should be ext_debug_utils but vulkano doesn't support that yet
            extensions.ext_debug_report = true;
        }

        let mut extensions: RawInstanceExtensions = (&extensions).into();
        for n in vec![
            "VK_KHR_external_memory_capabilities",
            "VK_KHR_external_semaphore_capabilities",
            "VK_KHR_get_physical_device_properties2",
        ] {
            extensions.insert(CString::new(n).unwrap());
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
            performance_warning: false,
            information: false,
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
        let extensions_supported = Self::check_device_extension_support(device);

        let swap_chain_adequate = if extensions_supported {
            let capabilities = surface.capabilities(*device).expect("surface capabilities");

            !capabilities.supported_formats.is_empty() && capabilities.present_modes.iter().next().is_some()
        } else {
            false
        };

        indices.is_complete() && extensions_supported && swap_chain_adequate
    }

    fn check_device_extension_support(device: &PhysicalDevice) -> bool {
        let available_extensions = RawDeviceExtensions::supported_by_device(*device);
        let device_extensions = device_extensions();

        available_extensions.intersection(&device_extensions) == device_extensions
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
            device_extensions(),
            queue_families
        )
        .expect("logical device");

        let graphics_queue = queues.next().unwrap();
        let present_queue = queues.next().unwrap_or_else(|| graphics_queue.clone());

        (device, graphics_queue, present_queue)
    }

    fn choose_swap_surface_format(available_formats: &[ (Format, ColorSpace) ]) -> (Format, ColorSpace) {
        // NOTE: the 'preferred format' mentioned in the tutorial doesn't seem to be
        // queryable in Vulkano (no VK_FORMAT_UNDEFINED enum)

        *available_formats.iter().find(|(format, color_space)| {
            *format == Format::B8G8R8A8Unorm && *color_space == ColorSpace::SrgbNonLinear
        })
        .unwrap_or_else(|| &available_formats[0])
    }

    fn choose_swap_present_mode(available_present_modes: SupportedPresentModes) -> PresentMode {
        if available_present_modes.mailbox {
            PresentMode::Mailbox
        } else if available_present_modes.immediate {
            PresentMode::Immediate
        } else {
            PresentMode::Fifo
        }
    }

    fn choose_swap_extent(capabilities: &Capabilities) -> [u32; 2] {
        if let Some(current_extent) = capabilities.current_extent {
            current_extent
        } else {
            let mut actual_extent = [WIDTH, HEIGHT];
            actual_extent[0] = capabilities.min_image_extent[0]
                          .max(capabilities.max_image_extent[0].min(actual_extent[0]));
            actual_extent[1] = capabilities.min_image_extent[1]
                          .max(capabilities.max_image_extent[1].min(actual_extent[1]));
            actual_extent
        }
    }

    fn create_swap_chain(
        instance: &Arc<Instance>,
        surface: &Arc<Surface<Window>>,
        physical_device_index: usize,
        device: &Arc<Device>,
        graphics_queue: &Arc<Queue>,
        present_queue: &Arc<Queue>,
        old_swapchain: Option<Arc<Swapchain<Window>>>,
    ) -> (Arc<Swapchain<Window>>, Vec<Arc<SwapchainImage<Window>>>) {
        let physical_device = PhysicalDevice::from_index(&instance, physical_device_index).unwrap();
        let capabilities = surface.capabilities(physical_device).expect("surface capabilities");

        let surface_format = Self::choose_swap_surface_format(&capabilities.supported_formats);
        let present_mode = Self::choose_swap_present_mode(capabilities.present_modes);
        let extent = Self::choose_swap_extent(&capabilities);

        let mut image_count = capabilities.min_image_count + 1;
        if capabilities.max_image_count.map(|max| image_count > max).unwrap_or(false) {
            image_count = capabilities.max_image_count.unwrap();
        }

        let image_usage = ImageUsage {
            color_attachment: true,
            transfer_source: true,
            ..ImageUsage::none()
        };

        let indices = Self::find_queue_families(&surface, &physical_device);

        let sharing: SharingMode = if indices.graphics_family != indices.present_family {
            vec![graphics_queue, present_queue].as_slice().into()
        } else {
            graphics_queue.into()
        };

        let (swap_chain, images) = Swapchain::new(
            device.clone(),
            surface.clone(),
            image_count,
            surface_format.0, // TODO: color space?
            extent,
            1, // layers
            image_usage,
            sharing,
            capabilities.current_transform,
            CompositeAlpha::Opaque,
            present_mode,
            true, // clipped
            old_swapchain.as_ref(),
        ).expect("swap chain");

        (swap_chain, images)
    }

    fn recreate_swap_chain(&mut self) {
        let (swap_chain, images) = Self::create_swap_chain(&self.instance, &self.surface, self.physical_device_index,
                                                           &self.device, &self.graphics_queue, &self.present_queue, Some(self.swap_chain.clone()));

        self.swap_chain        = swap_chain;
        self.swap_chain_images = images;

        self.render_pass             = Self::create_render_pass(&self.device, self.swap_chain.format());
        self.graphics_pipeline       = Self::create_graphics_pipeline(&self.device, self.swap_chain.dimensions(), &self.render_pass);
        self.swap_chain_framebuffers = Self::create_framebuffers(&self.swap_chain_images, &self.render_pass);

        self.create_command_buffers();
    }

    fn create_sync_objects(device: &Arc<Device>) -> Box<GpuFuture> {
        Box::new(vulkano::sync::now(device.clone())) as Box<GpuFuture>
    }

    fn create_render_pass(device: &Arc<Device>, color_format: Format) -> Arc<dyn RenderPassAbstract + Send + Sync> {
        Arc::new(vulkano::single_pass_renderpass!(device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: color_format,
                    samples: 1,
                }
            },
            pass: {
                color: [ color ],
                depth_stencil: {}
            }
        ).unwrap())
    }

    fn create_graphics_pipeline(
        device: &Arc<Device>,
        swap_chain_extent: [u32; 2],
        render_pass: &Arc<dyn RenderPassAbstract + Send + Sync>,
    // ) -> Arc<dyn GraphicsPipelineAbstract + Send + Sync> {
    ) -> Arc<ConcreteGraphicsPipeline> {
        let vert_shader_module = vertex_shader::Shader::load(device.clone()).expect("vertex shader module");
        let frag_shader_module = fragment_shader::Shader::load(device.clone()).expect("fragment shader module");

        let dimensions = [ swap_chain_extent[0] as f32, swap_chain_extent[1] as f32 ];
        let viewport = Viewport {
            origin: [ 0.0, 0.0 ],
            dimensions,
            depth_range: 0.0 .. 1.0,
        };

        Arc::new(GraphicsPipeline::start()
            .vertex_input(BufferlessDefinition {})
            .vertex_shader(vert_shader_module.main_entry_point(), ())
            .triangle_list()
            .primitive_restart(false)
            .viewports(vec![ viewport ]) // NOTE: also sets scissor to cover whole viewport
            .fragment_shader(frag_shader_module.main_entry_point(), ())
            .depth_clamp(false)
            // NOTE: there's an outcommented .rasterizer_discard() in Vulkano...
            .polygon_mode_fill()
            .line_width(1.0)
            .cull_mode_back()
            .front_face_clockwise()
            // NOTE: no depth_bias here, but on pipeline::raster::Rasterization
            .blend_pass_through()
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap())
    }

    fn create_framebuffers(
        swap_chain_images: &[Arc<SwapchainImage<Window>>],
        render_pass: &Arc<dyn RenderPassAbstract + Send + Sync>
    ) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
        swap_chain_images.iter()
            .map(|image| -> Arc<dyn FramebufferAbstract + Send + Sync> {
                Arc::new(Framebuffer::start(render_pass.clone())
                         .add(image.clone()).unwrap()
                         .build().unwrap())
            })
            .collect::<Vec<_>>()
    }

    fn create_command_buffers(&mut self) {
        let queue_family = self.graphics_queue.family();

        self.command_buffers = self.swap_chain_framebuffers.iter()
            .map(|framebuffer| {
                let vertices = BufferlessVertices { vertices: 3, instances: 1 };

                Arc::new(AutoCommandBufferBuilder::primary_simultaneous_use(self.device.clone(), queue_family).unwrap()
                         .begin_render_pass(framebuffer.clone(), false, vec![ [ 0.0, 0.0, 0.0, 1.0 ].into() ]).unwrap()
                         .draw(self.graphics_pipeline.clone(), &DynamicState::none(), vertices, (), ()).unwrap()
                         .end_render_pass().unwrap()
                         .build().unwrap())
            })
            .collect();
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

            self.draw_frame();

            if done {
                return;
            }
        }
    }

    fn draw_frame(&mut self) {
        // self.previous_frame_end.as_mut().unwrap().cleanup_finished();

        let _poses = self.vr.compositor.wait_get_poses().expect("wait_get_poses");

        if self.recreate_swap_chain {
            self.recreate_swap_chain();
            self.recreate_swap_chain = false;
        }

        let (image_index, acquire_future) = match vulkano::swapchain::acquire_next_image(self.swap_chain.clone(), None) {
            Ok(r) => r,
            Err(AcquireError::OutOfDate) => {
                self.recreate_swap_chain = true;
                return;
            }
            Err(e) => panic!("{:?}", e),
        };

        let command_buffer = self.command_buffers[image_index].clone();

        let future = acquire_future
            .then_execute(self.graphics_queue.clone(), command_buffer).unwrap()
            .then_swapchain_present(self.present_queue.clone(), self.swap_chain.clone(), image_index)
            .then_signal_fence_and_flush();

        let physical_device = PhysicalDevice::from_index(&self.instance, self.physical_device_index).unwrap();

        let swap_chain_image = &self.swap_chain_images[image_index];
        let [ width, height ] = swap_chain_image.dimensions().width_height();
        let format = swap_chain_image.format() as u32;
        let sample_count = swap_chain_image.samples();

        let eye = openvr::Eye::Left;
        let handle = openvr::compositor::texture::vulkan::Texture {
            image: swap_chain_image.inner().image.internal_object(),
            device: self.device.internal_object() as *mut _,
            physical_device: physical_device.internal_object() as *mut _,
            instance: self.instance.internal_object() as *mut _,
            queue: *self.graphics_queue.internal_object_guard().deref() as *mut _,
            queue_family_index: self.graphics_queue.id_within_family(),
            width, height, format, sample_count,

        };
        let texture = openvr::compositor::texture::Texture {
            handle:      openvr::compositor::texture::Handle::Vulkan(handle),
            color_space: openvr::compositor::texture::ColorSpace::Auto,
        };
        unsafe { self.vr.compositor.submit(eye, &texture, None, None).expect("submit") };

        match future {
            Ok(future) => {
                self.previous_frame_end = Some(Box::new(future) as Box<_>);
            },
            Err(vulkano::sync::FlushError::OutOfDate) => {
                self.recreate_swap_chain = true;
                self.previous_frame_end = Some(Box::new(vulkano::sync::now(self.device.clone())) as Box<_>);
            },
            Err(e) => {
                eprintln!("{:?}", e);
                self.previous_frame_end = Some(Box::new(vulkano::sync::now(self.device.clone())) as Box<_>);
            },
        }
    }
}

fn main() {
    let mut app = HelloTriangleApplication::new();
    app.main_loop();
}
