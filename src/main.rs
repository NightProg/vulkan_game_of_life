use std::sync::Arc;
use std::time::Duration;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassBeginInfo,
    SubpassContents,
};
use vulkano::device::physical::PhysicalDevice;
use vulkano::device::{Device, Queue, QueueFlags};
use vulkano::image::view::{ImageView, ImageViewCreateInfo, ImageViewType};
use vulkano::image::{Image, ImageUsage};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::swapchain::{
    Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo, acquire_next_image,
};
use vulkano::sync::GpuFuture;
use vulkano::{DeviceSize, Version, VulkanLibrary, sync};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 800;
const CELL_SIZE: f32 = 50.0;

const GRID_COUNT: f32 = GRID_WIDTH * GRID_HEIGHT;
const GRID_WIDTH: f32 = WINDOW_WIDTH as f32 / CELL_SIZE;
const GRID_HEIGHT: f32 = WINDOW_HEIGHT as f32 / CELL_SIZE;

const GRID_LIVE_COLOR: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
const GRID_DEAD_COLOR: [f32; 4] = [1.0, 1.0, 1.0, 1.0];

type Rect = [GAFVertex; 6];

fn normalize_coord(coord: [f32; 2]) -> [f32; 2] {
    let x = (coord[0] / WINDOW_WIDTH as f32) * 2.0 - 1.0;
    let y = (coord[1] / WINDOW_HEIGHT as f32) * 2.0 - 1.0;
    [x, y]
}

fn rect(coord: [f32; 2], color: [f32; 4]) -> Rect {
    let [x, y] = normalize_coord(coord);
    let [nx, ny] = normalize_coord([coord[0] + CELL_SIZE, coord[1] + CELL_SIZE]);
    [
        GAFVertex {
            position: [x, y],
            color,
        },
        GAFVertex {
            position: [nx, y],
            color,
        },
        GAFVertex {
            position: [nx, ny],
            color,
        },
        GAFVertex {
            position: [x, y],
            color,
        },
        GAFVertex {
            position: [nx, ny],
            color,
        },
        GAFVertex {
            position: [x, ny],
            color,
        },
    ]
}

#[derive(BufferContents, Vertex, Copy)]
#[repr(C)]
#[derive(Clone)]
struct GAFVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],

    #[format(R32G32B32A32_SFLOAT)]
    color: [f32; 4],
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
                    #version 450

                    layout(location = 0) in vec2 position;
                    layout(location = 1) in vec4 color;

                    layout(location = 0) out vec4 outColor;


                    void main() {
                        gl_Position = vec4(position, 0.0, 1.0);
                        outColor = color;
                    }
                ",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
                    #version 450

                    layout(location = 0) out vec4 f_color;
                    layout(location = 0) in vec4 color;

                    void main() {
                        f_color = color;
                    }
                ",
    }
}

struct RenderCtx {
    ctx: VkContext,
    swapchain: Arc<Swapchain>,
    images: Vec<Arc<Image>>,
    image_views: Vec<Arc<ImageView>>,
    std_memory_allocator: Arc<StandardMemoryAllocator>,
    squares: Vec<Rect>,
    pipeline: Arc<GraphicsPipeline>,
    render_pass: Arc<RenderPass>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    framebuffer: Vec<Arc<Framebuffer>>,
    vertex_buffers: Vec<Subbuffer<[GAFVertex]>>,
    frame_index: usize,
    viewport: Viewport,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
}

impl RenderCtx {
    fn new(context: VkContext) -> Self {
        let surface_capabilities = context
            .physical_device
            .surface_capabilities(&*context.surface, Default::default())
            .expect("Failed to get surface capabilities");
        let extra_image_count = surface_capabilities.min_image_count + 1;
        let min_image_count = surface_capabilities
            .max_image_count
            .unwrap_or(extra_image_count)
            .min(extra_image_count);
        let image_format = context
            .physical_device
            .surface_formats(&*context.surface, Default::default())
            .expect("Failed to get surface formats")
            .get(0)
            .expect("No surface formats available")
            .0;

        let (swapchain, images) = {
            Swapchain::new(
                context.device.clone(),
                context.surface.clone(),
                SwapchainCreateInfo {
                    min_image_count,
                    image_format,
                    image_extent: [WINDOW_WIDTH, WINDOW_HEIGHT],
                    image_usage: ImageUsage::COLOR_ATTACHMENT,
                    composite_alpha: surface_capabilities
                        .supported_composite_alpha
                        .into_iter()
                        .next()
                        .expect("Failed to get composite alpha"),
                    ..Default::default()
                },
            )
            .expect("Failed to create swapchain")
        };

        let image_views = images
            .iter()
            .map(|image| {
                ImageView::new(
                    image.clone(),
                    ImageViewCreateInfo {
                        view_type: ImageViewType::Dim2d,
                        format: image.format(),
                        subresource_range: image.subresource_range(),
                        ..Default::default()
                    },
                )
                .expect("Failed to create image view")
            })
            .collect::<Vec<_>>();

        let standard_memory_allocator =
            Arc::new(StandardMemoryAllocator::new_default(context.device.clone()));

        let render_pass = vulkano::single_pass_renderpass!(
            context.device.clone(),
            attachments: {
                 color: {
                    format: swapchain.image_format(),
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {},
            }
        )
        .expect("Failed to create render pass");

        let subpass = Subpass::from(render_pass.clone(), 0).expect("Failed to create subpass");

        let vs = vs::load(context.device.clone()).expect("Failed to load vertex shader");
        let fs = fs::load(context.device.clone()).expect("Failed to load fragment shader");
        let vs_main = vs.entry_point("main").unwrap();
        let fs_main = fs.entry_point("main").unwrap();
        let vertex_input_state = GAFVertex::per_vertex().definition(&vs_main).unwrap();

        let pipeline_stage = vec![
            PipelineShaderStageCreateInfo::new(vs_main),
            PipelineShaderStageCreateInfo::new(fs_main),
        ];

        let layout = PipelineLayout::new(
            context.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&pipeline_stage)
                .into_pipeline_layout_create_info(context.device.clone())
                .expect("Failed to create pipeline layout"),
        )
        .expect("Failed to create pipeline layout");

        let pipeline = GraphicsPipeline::new(
            context.device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                layout: layout.clone(),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState::default()),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.num_color_attachments(),
                    ColorBlendAttachmentState::default(),
                )),
                dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                subpass: Some(subpass.into()),
                stages: pipeline_stage.into(),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .expect("Failed to create graphics_pipeline");

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            context.device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        ));

        let framebuffer = image_views
            .iter()
            .map(|image| {
                Framebuffer::new(
                    render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![image.clone()],
                        layers: 1,
                        ..Default::default()
                    },
                )
                .expect("Failed to create framebuffer")
            })
            .collect::<Vec<_>>();

        let viewport = Viewport {
            offset: [0.0; 2],
            depth_range: 0.0..=1.0,
            extent: [WINDOW_WIDTH as f32, WINDOW_HEIGHT as f32],
        };

        let vertex_buffers = (0..2)
            .map(|_| {
                Buffer::new_slice(
                    standard_memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::VERTEX_BUFFER,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE
                            | MemoryTypeFilter::PREFER_DEVICE,
                        ..Default::default()
                    },
                    (GRID_COUNT * 6.0) as DeviceSize,
                )
                .expect("Failed to create vertex buffer")
            })
            .collect::<Vec<_>>();

        let previous_frame_end = Some(sync::now(context.device.clone()).boxed());

        RenderCtx {
            swapchain,
            images,
            std_memory_allocator: standard_memory_allocator,
            pipeline,
            render_pass,
            ctx: context,
            command_buffer_allocator,
            image_views,
            framebuffer,
            viewport,
            vertex_buffers,
            frame_index: 0,
            previous_frame_end,
            squares: Vec::new(),
        }
    }

    fn add_grid(&mut self, grid: Grid) {
        for i in 0..grid.height() {
            for j in 0..grid.width() {
                if grid.grid[i][j] {
                    let coord = [j as f32 * CELL_SIZE, i as f32 * CELL_SIZE];
                    self.add_square(coord, GRID_LIVE_COLOR);
                } else {
                    let coord = [j as f32 * CELL_SIZE, i as f32 * CELL_SIZE];
                    self.add_square(coord, GRID_DEAD_COLOR);
                }
            }
        }
    }

    fn set_color_square(&mut self, coord: [f32; 2], color: [f32; 4]) {
        let coord = normalize_coord(coord);
        for square in &mut self.squares {
            if square[0].position == coord {
                square[0].color = color;
                square[1].color = color;
                square[2].color = color;
                square[3].color = color;
                square[4].color = color;
                square[5].color = color;
            }
        }
    }

    fn is_square_colored(&self, coord: [f32; 2]) -> bool {
        let coord = normalize_coord(coord);
        for square in &self.squares {
            if square[0].position == coord {
                return true;
            }
        }
        false
    }

    fn add_square(&mut self, coord: [f32; 2], color: [f32; 4]) {
        if self.is_square_colored(coord) {
            self.set_color_square(coord, color);
            return;
        }
        let square = rect(coord, color);
        self.squares.push(square);
    }

    fn update_vertex(&mut self) {
        let mut vertex_buffer = self.vertex_buffers[self.frame_index]
            .write()
            .expect("Can't write to vertex buffer");
        for (rect_index, square) in self.squares.iter().enumerate() {
            for vertex_index in 0..6 {
                vertex_buffer[rect_index * 6 + vertex_index] = square[vertex_index];
            }
        }
    }

    fn draw(&mut self) {
        let mut command_buffer = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.ctx.graphic_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("Failed to create command buffer builder");

        let (image_index, suboptimal, aquirefutur) =
            acquire_next_image(self.swapchain.clone(), Some(Duration::from_secs(1)))
                .expect("Failed to acquire next image");

        if suboptimal {
            panic!("Suboptimal");
        }

        let current_framebuffer = self.framebuffer[image_index as usize].clone();

        command_buffer
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([1.0, 1.0, 1.0, 1.0].into())],
                    ..RenderPassBeginInfo::framebuffer(current_framebuffer)
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )
            .expect("Failed to create command buffer")
            .set_viewport(0, vec![self.viewport.clone()].into())
            .expect("Failed to set viewport")
            .bind_pipeline_graphics(self.pipeline.clone())
            .expect("Failed to bind pipeline")
            .bind_vertex_buffers(0, self.vertex_buffers[self.frame_index].clone())
            .expect("Failed to bind vertex buffers");

        unsafe {
            command_buffer
                .draw((self.squares.len() * 6) as u32, 1, 0, 0)
                .expect("Failed to draw");
        }

        command_buffer.end_render_pass(Default::default()).unwrap();

        let command_buffer = command_buffer.build().unwrap();

        let future = self
            .previous_frame_end
            .take()
            .unwrap()
            .join(aquirefutur)
            .then_execute(self.ctx.graphic_queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(
                self.ctx.graphic_queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(self.swapchain.clone(), image_index),
            )
            .then_signal_fence_and_flush()
            .expect("Failed to flush swapchain");

        self.previous_frame_end = Some(Box::new(future));
        self.frame_index = (self.frame_index + 1) % self.vertex_buffers.len();
    }
}

#[derive(Clone)]
struct VkContext {
    instance: Arc<Instance>,
    physical_device: Arc<PhysicalDevice>,
    device: Arc<Device>,
    surface: Arc<Surface>,
    window: Arc<Window>,
    graphic_queue: Arc<Queue>,
}

impl VkContext {
    fn pick_physical_device(
        instance: &Arc<Instance>,
        surface: &Arc<Surface>,
    ) -> Option<(usize, Arc<PhysicalDevice>)> {
        let devices = instance
            .enumerate_physical_devices()
            .expect("Failed to enumerate physical devices");

        for device in devices {
            if !device.supported_extensions().khr_swapchain {
                continue;
            }
            let queue_family_properties = device.queue_family_properties();
            for (index, properties) in queue_family_properties.iter().enumerate() {
                if properties.queue_flags.contains(QueueFlags::GRAPHICS)
                    && device
                        .surface_support(index as u32, surface)
                        .expect("Failed to check surface support")
                {
                    return Some((index, device));
                }
            }
        }
        None
    }
    fn new(event_loop: &ActiveEventLoop) -> Self {
        let vulkan_library = VulkanLibrary::new().expect("Failed to load Vulkan library");
        let mut instance_flags = InstanceCreateFlags::empty();
        if cfg!(target_os = "macos") {
            instance_flags |= InstanceCreateFlags::ENUMERATE_PORTABILITY;
        }

        let instance_extension =
            Surface::required_extensions(event_loop).expect("Failed to get instance extension");
        let instance_create_info = InstanceCreateInfo {
            application_name: Some("Game of Life".into()),
            application_version: Version::V1_0,
            flags: instance_flags,
            enabled_extensions: instance_extension,
            ..Default::default()
        };

        let instance = Instance::new(vulkan_library, instance_create_info)
            .expect("Failed to create Vulkan instance");
        let window = Arc::new(
            event_loop
                .create_window(
                    WindowAttributes::default()
                        .with_title("Game of Life")
                        .with_inner_size(LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
                        .with_visible(true)
                        .with_resizable(false),
                )
                .expect("Failed to create window"),
        );

        let surface = Surface::from_window(instance.clone(), window.clone())
            .expect("Failed to create Vulkan surface");
        let (queue_graphic_family_index, physical_device) =
            VkContext::pick_physical_device(&instance, &surface)
                .expect("No suitable physical device found");

        let (device, mut queues) = Device::new(
            physical_device.clone(),
            vulkano::device::DeviceCreateInfo {
                queue_create_infos: vec![vulkano::device::QueueCreateInfo {
                    queue_family_index: queue_graphic_family_index as u32,
                    ..Default::default()
                }],
                enabled_extensions: vulkano::device::DeviceExtensions {
                    khr_swapchain: true,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .expect("Failed to create Vulkan device");

        let graphic_queue = queues
            .find(|q| q.queue_family_index() == queue_graphic_family_index as u32)
            .expect("Failed to find graphics queue");

        VkContext {
            instance,
            physical_device,
            device,
            window,
            surface,
            graphic_queue,
        }
    }
}

#[derive(Clone)]
struct Grid {
    grid: Vec<Vec<bool>>,
}

impl Grid {
    fn new() -> Self {
        let grid = vec![vec![false; GRID_WIDTH as usize]; GRID_HEIGHT as usize];
        Grid { grid }
    }

    fn randomize(&mut self) {
        for i in 0..self.grid.len() {
            for j in 0..self.grid[i].len() {
                self.grid[i][j] = rand::random();
            }
        }
    }

    fn height(&self) -> usize {
        self.grid.len()
    }

    fn width(&self) -> usize {
        self.grid[0].len()
    }

    fn update(&mut self) {
        let height = self.height();
        let width = self.width();

        let mut new_grid = self.grid.clone();

        for y in 0..height {
            for x in 0..width {
                let mut live_neighbors = 0;

                for dy in [-1, 0, 1] {
                    for dx in [-1, 0, 1] {
                        if dx == 0 && dy == 0 {
                            continue;
                        }

                        let ny = y as isize + dy;
                        let nx = x as isize + dx;

                        if ny >= 0 && ny < height as isize && nx >= 0 && nx < width as isize {
                            if self.grid[ny as usize][nx as usize] {
                                live_neighbors += 1;
                            }
                        }
                    }
                }

                let is_alive = self.grid[y][x];
                new_grid[y][x] = match (is_alive, live_neighbors) {
                    (true, 2) | (true, 3) => true,
                    (false, 3) => true,
                    _ => false,
                };
            }
        }

        self.grid = new_grid;
    }
}

struct GameOfLife {
    context: Option<VkContext>,
    render_ctx: Option<RenderCtx>,
    grid: Grid,
}

impl GameOfLife {
    fn new() -> Self {
        let mut grid = Grid::new();

        grid.randomize();

        GameOfLife {
            context: None,
            render_ctx: None,
            grid,
        }
    }
}

impl ApplicationHandler for GameOfLife {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let vk_context = VkContext::new(event_loop);
        let render_ctx = RenderCtx::new(vk_context.clone());
        self.context = Some(vk_context);
        self.render_ctx = Some(render_ctx);

        self.render_ctx
            .as_mut()
            .unwrap()
            .add_grid(self.grid.clone());
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                self.render_ctx.as_mut().unwrap().update_vertex();
                self.render_ctx.as_mut().unwrap().draw();
                self.render_ctx
                    .as_mut()
                    .unwrap()
                    .previous_frame_end
                    .as_mut()
                    .unwrap()
                    .cleanup_finished();

                self.context.as_ref().unwrap().window.request_redraw();
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                self.grid.update();
                self.render_ctx.as_mut().unwrap().squares.clear();
                self.render_ctx
                    .as_mut()
                    .unwrap()
                    .add_grid(self.grid.clone());
            }
            _ => (),
        }
    }
}

fn main() {
    let mut game = GameOfLife::new();

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop
        .run_app(&mut game)
        .expect("Failed to run application");
}
