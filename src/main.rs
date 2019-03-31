use cgmath::Deg;
use cgmath::Matrix4;
use cgmath::Point3;
use cgmath::Rad;
use cgmath::Vector3;
use itertools::Itertools;
use png::HasParameters;
use std::collections::HashSet;
use std::fs::File;
use std::iter::FromIterator;
use std::sync::Arc;
use std::time::Instant;
use vulkano::buffer::BufferAccess;
use vulkano::buffer::BufferUsage;
use vulkano::buffer::CpuAccessibleBuffer;
use vulkano::buffer::TypedBufferAccess;
use vulkano::buffer::immutable::ImmutableBuffer;
use vulkano::command_buffer::AutoCommandBuffer;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::DynamicState;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::Device;
use vulkano::device::DeviceExtensions;
use vulkano::device::Features;
use vulkano::device::Queue;
use vulkano::format::Format;
use vulkano::framebuffer::Framebuffer;
use vulkano::framebuffer::FramebufferAbstract;
use vulkano::framebuffer::RenderPassAbstract;
use vulkano::framebuffer::Subpass;
use vulkano::image::ImageUsage;
use vulkano::image::swapchain::SwapchainImage;
use vulkano::impl_vertex;
use vulkano::instance::ApplicationInfo;
use vulkano::instance::Instance;
use vulkano::instance::PhysicalDevice;
use vulkano::instance::RawInstanceExtensions;
use vulkano::instance::Version;
use vulkano::instance::debug::DebugCallback;
use vulkano::instance::debug::MessageTypes;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::GraphicsPipelineAbstract;
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
use winit::VirtualKeyCode;
use winit::Window;
use winit::WindowBuilder;
use winit::WindowEvent;
use winit::dpi::LogicalSize;
use winit::os::unix::WindowBuilderExt;

const WIDTH:  u32 = 1280;
const HEIGHT: u32 = 720;

const VALIDATION_LAYERS: &[&str] = &[
    "VK_LAYER_LUNARG_standard_validation",
];

#[cfg(not(release))]
const ENABLE_VALIDATION_LAYERS: bool = true;
#[cfg(release)]
const ENABLE_VALIDATION_LAYERS: bool = false;

/// Required device extensions
fn device_extensions() -> DeviceExtensions {
    DeviceExtensions {
        khr_swapchain: true,
        ..vulkano::device::DeviceExtensions::none()
    }
}

#[derive(Copy, Clone, Debug)]
struct Vertex {
    pos:   [f32; 3],
    color: [f32; 3],
}

impl_vertex!(Vertex, pos, color);
impl Vertex {
    fn new(pos: [f32; 3], color: [f32; 3]) -> Self {
        Self {
            pos, color,
        }
    }
}

fn vertices() -> [Vertex; 4] {
    [
        Vertex::new([ -0.5, -0.5, 0.0 ], [ 0.24, 0.2, 0.0 ]),
        Vertex::new([  0.5, -0.5, 0.0 ], [ 0.24, 0.2, 0.0 ]),
        Vertex::new([  0.5,  0.5, 0.0 ], [ 0.24, 0.2, 0.0 ]),
        Vertex::new([ -0.5,  0.5, 1.0 ], [ 0.24, 0.2, 0.0 ]),
    ]
}

fn indices() -> [u32; 6] {
    [ 0, 1, 3, 3, 2, 0 ]
}

#[derive(Clone)]
struct UniformBufferObject {
    model: Matrix4<f32>,
    view:  Matrix4<f32>,
    proj:  Matrix4<f32>,
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

struct HelloTriangleApplication {
    instance: Arc<Instance>,
    _debug_callback: Option<DebugCallback>,

    events_loop: EventsLoop,
    surface: Arc<Surface<Window>>,

    physical_device_index: usize,
    device: Arc<Device>,

    graphics_queue: Arc<Queue>,
    present_queue: Arc<Queue>,

    swap_chain: Arc<Swapchain<Window>>,
    swap_chain_images: Vec<Arc<SwapchainImage<Window>>>,

    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    graphics_pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,

    swap_chain_framebuffers: Vec<Arc<dyn FramebufferAbstract + Send + Sync>>,

    vertex_buffer:   Arc<dyn BufferAccess + Send + Sync>,
    index_buffer:    Arc<dyn TypedBufferAccess<Content=[u32]> + Send + Sync>,
    uniform_buffers: Vec<Arc<CpuAccessibleBuffer<UniformBufferObject>>>,
    command_buffers: Vec<Arc<AutoCommandBuffer>>,

    previous_frame_end: Option<Box<GpuFuture>>,
    recreate_swap_chain: bool,

    state: UniformBufferObject,
    start_time: Instant,
}

impl HelloTriangleApplication {
    pub fn new() -> Self {
        let mut decoder = png::Decoder::new(File::open("heightmap.png").expect("heightmap file"));
        decoder.set(png::Transformations::IDENTITY);

        let (info, mut reader) = decoder.read_info().expect("heightmap format");
        let mut heightmap      = vec![0; info.buffer_size()];
        reader.next_frame(&mut heightmap).expect("heightmap data");

        let heightmap = heightmap.chunks(2)
            .map(|b| u16::from_be_bytes([ b[0], b[1] ]))
            // .chunks(info.width as usize)
            // .into_iter()
            // .map(Iterator::collect)
            .collect();

        let instance        = Self::create_instance();
        let _debug_callback = Self::setup_debug_callback(&instance);

        let (events_loop, surface) = Self::create_surface(&instance);

        let physical_device_index                   = Self::pick_physical_device(&instance, &surface);
        let (device, graphics_queue, present_queue) = Self::create_logical_device(&instance, &surface, physical_device_index);

        let (swap_chain, swap_chain_images) = Self::create_swap_chain(&instance, &surface, physical_device_index,
                                                                      &device, &graphics_queue, &present_queue, None);

        let render_pass       = Self::create_render_pass(&device, swap_chain.format());
        let graphics_pipeline = Self::create_graphics_pipeline(&device, swap_chain.dimensions(), &render_pass);

        let swap_chain_framebuffers = Self::create_framebuffers(&swap_chain_images, &render_pass);

        let state = Self::create_state(swap_chain.dimensions());
        let start_time = Instant::now();

        // let vertex_buffer   = Self::create_vertex_buffer(&graphics_queue);
        // let index_buffer    = Self::create_index_buffer(&graphics_queue);
        let (vertex_buffer, index_buffer) = Self::create_terrain(&graphics_queue, heightmap, info.width, info.height);
        let uniform_buffers = Self::create_uniform_buffers(&device, swap_chain_images.len(), &state);

        let previous_frame_end = Some(Self::create_sync_objects(&device));

        let mut app = Self {
            instance,
            _debug_callback,

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

            vertex_buffer,
            index_buffer,
            uniform_buffers,
            command_buffers: vec![],

            previous_frame_end,
            recreate_swap_chain: false,

            state,
            start_time,
        };

        app.create_command_buffers();

        app
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

        let extensions: RawInstanceExtensions = (&extensions).into();
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
        let available_extensions = DeviceExtensions::supported_by_device(*device);
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
            &device_extensions(),
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
            let mut actual_extent = [ WIDTH, HEIGHT ];
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
    ) -> Arc<dyn GraphicsPipelineAbstract + Send + Sync> {
        mod vertex_shader {
            vulkano_shaders::shader! {
               ty: "vertex",
               path: "src/shader.vert",
            }
        }

        mod fragment_shader {
            vulkano_shaders::shader! {
                ty: "fragment",
                path: "src/shader.frag",
            }
        }

        let vert_shader_module = vertex_shader::Shader::load(device.clone()).expect("vertex shader module");
        let frag_shader_module = fragment_shader::Shader::load(device.clone()).expect("fragment shader module");

        let dimensions = [ swap_chain_extent[0] as f32, swap_chain_extent[1] as f32 ];
        let viewport = Viewport {
            origin: [ 0.0, 0.0 ],
            dimensions,
            depth_range: 0.0 .. 1.0,
        };

        Arc::new(GraphicsPipeline::start()
            .vertex_input_single_buffer::<Vertex>()
            .vertex_shader(vert_shader_module.main_entry_point(), ())
            .triangle_strip()
            .primitive_restart(true)
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

    // fn create_vertex_buffer(graphics_queue: &Arc<Queue>) -> Arc<dyn BufferAccess + Send + Sync> {
    //     let (buffer, future) = ImmutableBuffer::from_iter(
    //             vertices().iter().cloned(), BufferUsage::vertex_buffer(),
    //             graphics_queue.clone())
    //         .unwrap();

    //     future.flush().unwrap();

    //     buffer
    // }

    // fn create_index_buffer(graphics_queue: &Arc<Queue>) -> Arc<dyn TypedBufferAccess<Content=[u16]> + Send + Sync> {
    //     let (buffer, future) = ImmutableBuffer::from_iter(
    //             indices().iter().cloned(), BufferUsage::index_buffer(),
    //             graphics_queue.clone())
    //         .unwrap();

    //     future.flush().unwrap();

    //     buffer
    // }

    fn create_terrain(graphics_queue: &Arc<Queue>, heightmap: Vec<u16>, width: u32, height: u32) -> (Arc<dyn BufferAccess + Send + Sync>, Arc<dyn TypedBufferAccess<Content=[u32]> + Send + Sync>) {
        let vertices = heightmap.iter().enumerate().map(|(idx, height)| {
            let x = ((idx as u32 % width) as f32 / width as f32 - 0.5) * 7.0;
            let y = ((idx as u32 / width) as f32 / width as f32 - 0.5) * 7.0;
            Vertex::new([ x, y, *height as f32 / 65535.0 ], [ 0.1, 0.2, 0.0 ])
        })
        .collect::<Vec<_>>();

        let indices = (1 .. height).flat_map(|y| {
                (0 .. width).flat_map(move |x| {
                    vec![
                        (y - 1) * height + x,
                        (y - 0) * height + x,
                    ]
                })
                .chain([ u32::max_value() ].into_iter().cloned())
            })
            .collect::<Vec<_>>();

        // let indices = [ 0, width, 1, width + 1 ];

        dbg!(vertices[0]);
        dbg!(vertices[width as usize]);
        dbg!(vertices[1]);
        dbg!(vertices[width as usize + 1]);

        let (vertex_buffer, vertex_future) = ImmutableBuffer::from_iter(
                vertices.iter().cloned(), BufferUsage::vertex_buffer(),
                graphics_queue.clone())
            .unwrap();

        let (index_buffer, index_future) = ImmutableBuffer::from_iter(
                indices.iter().cloned(), BufferUsage::index_buffer(),
                graphics_queue.clone())
            .unwrap();

        vertex_future.flush().unwrap();
        index_future.flush().unwrap();

        (vertex_buffer, index_buffer)
    }

    fn create_state(dimensions: [u32; 2]) -> UniformBufferObject {
        let dimensions = [ dimensions[0] as f32, dimensions[1] as f32 ];

        let model = Matrix4::from_angle_z(Rad(0.0));

        let view = Matrix4::look_at(
            Point3::new(2.0, 2.0, 1.0),
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        );

        let mut proj = cgmath::perspective(
            Rad::from(Deg(45.0)),
            dimensions[0] as f32 / dimensions[1] as f32,
            0.1,
            10.0,
        );

        proj.y.y *= -1.0;

        UniformBufferObject {
            model, view, proj,
        }
    }

    fn create_uniform_buffers(device: &Arc<Device>, num_buffers: usize, state: &UniformBufferObject) -> Vec<Arc<CpuAccessibleBuffer<UniformBufferObject>>> {
        let mut buffers = Vec::new();

        for _ in 0..num_buffers {
            let buffer = CpuAccessibleBuffer::from_data(
                device.clone(),
                BufferUsage::uniform_buffer_transfer_destination(),
                state.clone(),
            ).unwrap();

            buffers.push(buffer);
        }

        buffers
    }

    fn update_uniform_buffer(&self, idx: usize) {
        let mut buf = self.uniform_buffers[idx].write().unwrap();
        *buf = self.state.clone();
    }

    fn create_command_buffers(&mut self) {
        let queue_family = self.graphics_queue.family();

        self.command_buffers = self.swap_chain_framebuffers.iter().enumerate()
            .map(|(idx, framebuffer)| {
                let set = Arc::new(
                    PersistentDescriptorSet::start(self.graphics_pipeline.clone(), 0)
                        .add_buffer(self.uniform_buffers[idx].clone()).unwrap()
                        .build().unwrap()
                );

                Arc::new(AutoCommandBufferBuilder::primary_simultaneous_use(self.device.clone(), queue_family).unwrap()
                         .begin_render_pass(framebuffer.clone(), false, vec![ [ 0.0, 0.0, 0.0, 1.0 ].into() ]).unwrap()
                         .draw_indexed(
                             self.graphics_pipeline.clone(),
                             &DynamicState::none(),
                             vec![ self.vertex_buffer.clone() ],
                             self.index_buffer.clone(),
                             set.clone(),
                             (),
                         ).unwrap()
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
                    Event::WindowEvent { event: WindowEvent::KeyboardInput { input, .. }, .. } => {
                        match input.virtual_keycode {
                            Some(VirtualKeyCode::Escape) => {
                                done = true;
                            },
                            _ => ()
                        }
                    },
                    _ => ()
                }
            });

            self.update_state();

            self.draw_frame();

            if done {
                return;
            }
        }
    }

    fn update_state(&mut self) {
        let elapsed = self.start_time.elapsed();
        let elapsed = (elapsed.as_secs() * 1000) + u64::from(elapsed.subsec_millis());

        let model = Matrix4::from_angle_z(Rad::from(Deg(elapsed as f32 * 0.010)));

        self.state.model = model;
    }

    fn draw_frame(&mut self) {
        self.previous_frame_end.as_mut().unwrap().cleanup_finished();

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

        self.update_uniform_buffer(image_index);

        let command_buffer = self.command_buffers[image_index].clone();

        let future = acquire_future
            .then_execute(self.graphics_queue.clone(), command_buffer).unwrap()
            .then_swapchain_present(self.present_queue.clone(), self.swap_chain.clone(), image_index)
            .then_signal_fence_and_flush();

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
