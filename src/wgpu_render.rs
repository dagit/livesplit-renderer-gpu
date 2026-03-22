//! The main wgpu renderer: owns GPU state, consumes livesplit-core [`Scene`]
//! data, and issues draw calls.
//!
//! [`Scene`]: livesplit_core::rendering::Scene

use bytemuck::{Pod, Zeroable};
use livesplit_core::{
    layout::LayoutState,
    rendering::{Background, Entity, FillShader, Handle, LabelHandle, SceneManager, Transform},
    settings::{BackgroundImage, ImageCache},
};
use std::cell::RefCell;
use std::sync::Arc;
use wgpu::util::DeviceExt;

use crate::{
    common::{tessellate_stroke, vertex_bounds, BLUR_FACTOR, SHADOW_OFFSET},
    wgpu_allocator::WgpuAllocator,
    wgpu_shaders,
    wgpu_types::{WgpuFont, WgpuImage, WgpuLabel, WgpuPath},
};

/// Number of MSAA samples for antialiasing.
pub const MSAA_SAMPLES: u32 = 4;

/// Uniform data for the path shader, uploaded as a uniform buffer.
///
/// Layout must match the `PathUniforms` struct in the WGSL shader
/// (see [`wgpu_shaders::PATH_SHADER_SRC`]).
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
#[repr(C)]
struct PathUniformData {
    scale: [f32; 2],
    offset: [f32; 2],
    resolution: [f32; 2],
    bounds: [f32; 2],
    color_a: [f32; 4],
    color_b: [f32; 4],
    shader_type: i32,
    _pad0: i32,
    _pad1: i32,
    _pad2: i32,
}

// Compile-time checks that Rust struct sizes match WGSL expectations.
const _: () = assert!(std::mem::size_of::<PathUniformData>() == 80);

/// Uniform data for the image shader, uploaded as a uniform buffer.
///
/// Layout must match the `ImageUniforms` struct in the WGSL shader
/// (see [`wgpu_shaders::IMAGE_SHADER_SRC`]).
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
#[repr(C)]
struct ImageUniformData {
    scale: [f32; 2],
    offset: [f32; 2],
    resolution: [f32; 2],
    brightness: f32,
    opacity: f32,
    flip_uv_y: i32,
    _pad0: i32,
    _pad1: i32,
    _pad2: i32,
}

const _: () = assert!(std::mem::size_of::<ImageUniformData>() == 48);

/// Cached blurred background texture.
struct BlurCache {
    /// Identity of the source image (pointer address of its `Arc` data).
    source_ptr: usize,
    /// The blur setting this was computed for.
    blur_value: f32,
    /// The uploaded wgpu texture containing the blurred pixels.
    /// Must be kept alive so the bind group remains valid.
    #[allow(dead_code)]
    texture: wgpu::Texture,
    /// Bind group for sampling the blurred texture.
    bind_group: Arc<wgpu::BindGroup>,
}

/// A GPU buffer that grows to accommodate the largest data written to it.
///
/// Avoids per-draw buffer allocation by reusing a single buffer that is
/// resized (via reallocation) only when a larger payload is needed.
struct GrowableBuffer {
    buffer: wgpu::Buffer,
    capacity: u64,
    usage: wgpu::BufferUsages,
}

impl GrowableBuffer {
    fn new(device: &wgpu::Device, usage: wgpu::BufferUsages, initial_capacity: u64) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: initial_capacity,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            buffer,
            capacity: initial_capacity,
            usage,
        }
    }

    /// Write data into the buffer, growing it if necessary.
    fn write(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: &[u8]) {
        let len = data.len() as u64;
        if len == 0 {
            return;
        }
        if len > self.capacity {
            // Grow to at least 2× or the needed size, whichever is larger.
            let new_capacity = len.max(self.capacity * 2);
            self.buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: new_capacity,
                usage: self.usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.capacity = new_capacity;
        }
        queue.write_buffer(&self.buffer, 0, data);
    }
}

/// Shared vertex and index buffers, wrapped in [`RefCell`] to allow mutable
/// access from `&self` draw methods (needed because the scene borrow from
/// `SceneManager` prevents `&mut self`).
struct Buffers {
    vertex: GrowableBuffer,
    index: GrowableBuffer,
}

/// A GPU-accelerated renderer for livesplit-core layouts using wgpu.
///
/// Renders a livesplit-core [`LayoutState`] to a wgpu texture using two
/// render pipelines (one for filled/stroked paths, one for textured quads).
/// All paths are tessellated via lyon at creation time and drawn as indexed
/// triangle meshes.
///
/// # Two-layer caching
///
/// The livesplit-core scene is split into a bottom layer (background,
/// component backgrounds, static elements) and a top layer (dynamic text,
/// icons). The bottom layer is rendered to an off-screen MSAA texture,
/// resolved to a texture, and reused across frames when unchanged. This
/// avoids re-rendering the majority of the scene every frame.
///
/// # Example
///
/// ```no_run
/// # use livesplit_renderer_glow::WgpuRenderer;
/// # fn example(device: &wgpu::Device, queue: &wgpu::Queue,
/// #            state: &livesplit_core::layout::LayoutState,
/// #            image_cache: &livesplit_core::settings::ImageCache,
/// #            output_view: &wgpu::TextureView) {
/// let mut renderer = WgpuRenderer::new(device, wgpu::TextureFormat::Bgra8UnormSrgb);
///
/// // Each frame:
/// let new_size = renderer.render(device, queue, state, image_cache, [800, 600], &output_view);
/// # }
/// ```
pub struct WgpuRenderer {
    /// Resource allocator for paths, images, fonts, and labels.
    allocator: WgpuAllocator,

    /// Livesplit-core scene manager that diffs layout state into a scene
    /// graph of entities.
    scene_manager: SceneManager<Option<WgpuPath>, WgpuImage, WgpuFont, WgpuLabel>,

    /// Render pipeline for filled/stroked paths.
    path_pipeline: wgpu::RenderPipeline,
    /// Bind group layout for path uniforms.
    path_bind_group_layout: wgpu::BindGroupLayout,

    /// Render pipeline for textured quads (images, FBO blitting).
    image_pipeline: wgpu::RenderPipeline,
    /// Bind group layout for image uniforms.
    image_uniform_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group layout for image textures.
    image_texture_bind_group_layout: wgpu::BindGroupLayout,

    /// Default sampler for textures.
    sampler: wgpu::Sampler,

    /// The surface/output texture format.
    format: wgpu::TextureFormat,

    /// Reusable vertex and index buffers. In a [`RefCell`] because draw
    /// methods need mutable buffer access while the scene graph is borrowed
    /// immutably from [`scene_manager`](Self::scene_manager).
    buffers: RefCell<Buffers>,

    /// Off-screen resolve texture (non-MSAA) for the cached bottom layer.
    fbo_texture: Option<wgpu::Texture>,
    /// Texture view for the resolve target.
    fbo_texture_view: Option<wgpu::TextureView>,

    /// MSAA texture used as the rendering target for antialiased content.
    msaa_texture: Option<wgpu::Texture>,
    /// Texture view for the MSAA rendering target.
    msaa_texture_view: Option<wgpu::TextureView>,

    /// Current dimensions of the off-screen textures.
    fbo_size: [u32; 2],
    /// Whether the cached bottom layer needs re-rendering (e.g., after a
    /// resize).
    bottom_layer_dirty: bool,

    /// Cached blurred background image texture, reused across frames when
    /// the source image and blur setting are unchanged.
    blur_cache: Option<BlurCache>,
}

impl WgpuRenderer {
    /// Create a new renderer.
    ///
    /// Creates render pipelines and initializes the livesplit-core scene
    /// manager.
    ///
    #[must_use]
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let path_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("path_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let image_uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("image_uniform_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let image_texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("image_texture_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let path_pipeline =
            wgpu_shaders::create_path_pipeline(device, format, &path_bind_group_layout);
        let image_pipeline = wgpu_shaders::create_image_pipeline(
            device,
            format,
            &image_uniform_bind_group_layout,
            &image_texture_bind_group_layout,
        );

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("default_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Initial buffer sizes: 4 KiB for vertices/indices is enough for
        // several hundred-vertex paths before any reallocation is needed.
        let buffers = RefCell::new(Buffers {
            vertex: GrowableBuffer::new(device, wgpu::BufferUsages::VERTEX, 4096),
            index: GrowableBuffer::new(device, wgpu::BufferUsages::INDEX, 4096),
        });

        let mut allocator = WgpuAllocator::new();
        let scene_manager = SceneManager::new(&mut allocator);

        Self {
            allocator,
            scene_manager,
            path_pipeline,
            path_bind_group_layout,
            image_pipeline,
            image_uniform_bind_group_layout,
            image_texture_bind_group_layout,
            sampler,
            format,
            buffers,
            fbo_texture: None,
            fbo_texture_view: None,
            msaa_texture: None,
            msaa_texture_view: None,
            fbo_size: [0, 0],
            bottom_layer_dirty: true,
            blur_cache: None,
        }
    }

    /// Render the layout to the given output texture view.
    ///
    /// Returns an optional new resolution hint from livesplit-core's layout
    /// engine, indicating the layout's preferred size changed. The caller
    /// can use this to resize the window or viewport.
    /// # Panics
    ///
    /// Panics if off-screen textures have not been initialized (i.e., if
    /// the viewport dimensions are zero).
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        state: &LayoutState,
        image_cache: &ImageCache,
        [width, height]: [u32; 2],
        output_view: &wgpu::TextureView,
    ) -> Option<[f32; 2]> {
        if width == 0 || height == 0 {
            return None;
        }

        // Precision loss is acceptable: viewport dimensions are small
        // relative to f32 mantissa range.
        #[expect(clippy::cast_precision_loss)]
        let resolution = [width as f32, height as f32];

        // Ensure off-screen textures match the viewport size.
        if self.fbo_size != [width, height] {
            self.resize_fbo(device, width, height);
            self.bottom_layer_dirty = true;
        }

        let new_resolution =
            self.scene_manager
                .update_scene(&mut self.allocator, resolution, state, image_cache);

        // Pre-compute blur before starting render passes (needs &mut self).
        // Extract the blur parameters while scene is borrowed, then drop
        // the borrow before calling update_blur_cache.
        let blur_params = {
            let scene = self.scene_manager.scene();
            if scene.bottom_layer_changed() || self.bottom_layer_dirty {
                match scene.background() {
                    Some(Background::Image(bg_image, _)) if bg_image.blur > 0.0 => {
                        Some((Arc::clone(&bg_image.image.data), bg_image.blur))
                    }
                    _ => None,
                }
            } else {
                None
            }
        };
        if let Some((image_data, blur_value)) = blur_params {
            self.update_blur_cache(device, queue, &image_data, blur_value);
        }

        let scene = self.scene_manager.scene();
        let bottom_layer_changed = scene.bottom_layer_changed();

        let msaa_view = self
            .msaa_texture_view
            .as_ref()
            .expect("MSAA texture not initialized");
        let fbo_view = self
            .fbo_texture_view
            .as_ref()
            .expect("FBO texture not initialized");

        if bottom_layer_changed || self.bottom_layer_dirty {
            // Render bottom layer into MSAA texture, resolving to fbo_texture.
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("bottom_layer_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: msaa_view,
                        resolve_target: Some(fbo_view),
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                if let Some(bg) = scene.background() {
                    self.render_background(device, queue, &mut pass, bg, resolution);
                }

                for entity in scene.bottom_layer() {
                    self.render_entity(device, queue, &mut pass, entity, resolution);
                }
            }

            queue.submit(std::iter::once(encoder.finish()));
            self.bottom_layer_dirty = false;
        }

        // Composite: blit cached bottom layer + render top layer into MSAA,
        // resolving to the output.
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("composite_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: msaa_view,
                    resolve_target: Some(output_view),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Blit cached bottom layer.
            self.blit_fbo(device, queue, &mut pass, resolution);

            // Render top layer.
            for entity in scene.top_layer() {
                self.render_entity(device, queue, &mut pass, entity, resolution);
            }
        }

        queue.submit(std::iter::once(encoder.finish()));

        new_resolution
    }

    /// Pre-compute the blurred background texture if needed.
    ///
    /// Called before the render passes while we still have `&mut self`,
    /// so the result can be stored in `self.blur_cache`.
    fn update_blur_cache(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        image_data: &Arc<crate::wgpu_types::WgpuImageData>,
        blur_value: f32,
    ) {
        let source_ptr = Arc::as_ptr(image_data) as usize;

        // Check if the cache is already valid.
        if let Some(cache) = &self.blur_cache {
            if cache.source_ptr == source_ptr
                && (cache.blur_value - blur_value).abs() < f32::EPSILON
            {
                return;
            }
        }

        // Cache miss — blur on CPU and upload.
        let data = image_data;
        #[expect(clippy::cast_precision_loss)]
        let sigma = BLUR_FACTOR * blur_value * (data.width.max(data.height) as f32);

        let blurred = image::DynamicImage::ImageRgba8(
            image::RgbaImage::from_raw(data.width, data.height, data.pixels.clone())
                .expect("pixel data size mismatch"),
        )
        .blur(sigma);

        let blurred_rgba = blurred.to_rgba8();

        // Image source data is sRGB RGBA8, so Rgba8UnormSrgb is the correct
        // format regardless of the output surface format.
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("blur_texture"),
            size: wgpu::Extent3d {
                width: data.width,
                height: data.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &blurred_rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * data.width),
                rows_per_image: Some(data.height),
            },
            wgpu::Extent3d {
                width: data.width,
                height: data.height,
                depth_or_array_layers: 1,
            },
        );

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group = Arc::new(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blur_texture_bind_group"),
            layout: &self.image_texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        }));

        self.blur_cache = Some(BlurCache {
            source_ptr,
            blur_value,
            texture,
            bind_group,
        });
    }

    /// Render a single scene entity.
    fn render_entity(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pass: &mut wgpu::RenderPass<'_>,
        entity: &Entity<Option<WgpuPath>, WgpuImage, WgpuLabel>,
        resolution: [f32; 2],
    ) {
        match entity {
            Entity::FillPath(path, shader, transform) => {
                if let Some(path) = path.as_ref() {
                    self.draw_path(device, queue, pass, path, shader, transform, resolution);
                }
            }
            Entity::StrokePath(path, stroke_width, color, transform) => {
                if let Some(path) = path.as_ref() {
                    if let Some(stroked) = tessellate_stroke(path, *stroke_width) {
                        let shader = FillShader::SolidColor(*color);
                        self.draw_path(
                            device, queue, pass, &stroked, &shader, transform, resolution,
                        );
                    }
                }
            }
            Entity::Image(image, transform) => {
                self.draw_image(device, queue, pass, image, transform, resolution);
            }
            Entity::Label(label, shader, text_shadow, transform) => {
                self.draw_label(
                    device,
                    queue,
                    pass,
                    label,
                    shader,
                    text_shadow.as_ref(),
                    transform,
                    resolution,
                );
            }
        }
    }

    /// Draw a filled path with the given shader and transform.
    ///
    /// Reuses the renderer's vertex and index buffers, growing them as needed.
    #[allow(clippy::too_many_arguments)]
    fn draw_path(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pass: &mut wgpu::RenderPass<'_>,
        path: &WgpuPath,
        shader: &FillShader,
        transform: &Transform,
        resolution: [f32; 2],
    ) {
        if path.vertices.is_empty() || path.indices.is_empty() {
            return;
        }

        let uniform_data = Self::build_path_uniforms(shader, path, transform, resolution);

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("path_uniform_buffer"),
            contents: bytemuck::bytes_of(&uniform_data),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("path_bind_group"),
            layout: &self.path_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let mut bufs = self.buffers.borrow_mut();
        bufs.vertex
            .write(device, queue, bytemuck::cast_slice(&path.vertices));
        bufs.index
            .write(device, queue, bytemuck::cast_slice(&path.indices));

        pass.set_pipeline(&self.path_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_vertex_buffer(0, bufs.vertex.buffer.slice(..));
        pass.set_index_buffer(bufs.index.buffer.slice(..), wgpu::IndexFormat::Uint32);
        #[expect(clippy::cast_possible_truncation)]
        pass.draw_indexed(0..path.indices.len() as u32, 0, 0..1);
    }

    /// Build the uniform data for a path draw call.
    fn build_path_uniforms(
        shader: &FillShader,
        path: &WgpuPath,
        transform: &Transform,
        resolution: [f32; 2],
    ) -> PathUniformData {
        match shader {
            FillShader::SolidColor(color) => PathUniformData {
                scale: [transform.scale_x, transform.scale_y],
                offset: [transform.x, transform.y],
                resolution,
                bounds: [0.0, 0.0],
                color_a: *color,
                color_b: [0.0; 4],
                shader_type: 0,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            },
            FillShader::VerticalGradient(top, bottom) => {
                let [min, max] = vertex_bounds(&path.vertices, 1);
                PathUniformData {
                    scale: [transform.scale_x, transform.scale_y],
                    offset: [transform.x, transform.y],
                    resolution,
                    bounds: [min, max],
                    color_a: *top,
                    color_b: *bottom,
                    shader_type: 1,
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                }
            }
            FillShader::HorizontalGradient(left, right) => {
                let [min, max] = vertex_bounds(&path.vertices, 0);
                PathUniformData {
                    scale: [transform.scale_x, transform.scale_y],
                    offset: [transform.x, transform.y],
                    resolution,
                    bounds: [min, max],
                    color_a: *left,
                    color_b: *right,
                    shader_type: 2,
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                }
            }
        }
    }

    /// Draw a text label (each glyph is a filled path).
    ///
    /// If `text_shadow` is set, a shadow pass is rendered first at a small
    /// offset with the shadow color modulated by the label's alpha.
    ///
    /// # Panics
    ///
    /// Panics if the label's internal [`RwLock`](std::sync::RwLock) is poisoned.
    #[allow(clippy::too_many_arguments)]
    fn draw_label(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pass: &mut wgpu::RenderPass<'_>,
        label: &LabelHandle<WgpuLabel>,
        shader: &FillShader,
        text_shadow: Option<&[f32; 4]>,
        transform: &Transform,
        resolution: [f32; 2],
    ) {
        let label = label.read().expect("label RwLock poisoned");

        // Render shadow pass first.
        if let Some(shadow_color) = text_shadow {
            let alpha = match shader {
                FillShader::SolidColor([.., a]) => *a,
                FillShader::VerticalGradient([.., a1], [.., a2])
                | FillShader::HorizontalGradient([.., a1], [.., a2]) => 0.5 * (a1 + a2),
            };
            let shadow_rgba = [
                shadow_color[0],
                shadow_color[1],
                shadow_color[2],
                shadow_color[3] * alpha,
            ];
            let shadow_shader = FillShader::SolidColor(shadow_rgba);
            let shadow_transform = transform.pre_translate(SHADOW_OFFSET, SHADOW_OFFSET);

            for glyph in label.glyphs() {
                if let Some(path) = &glyph.path {
                    let t = shadow_transform
                        .pre_translate(glyph.x, glyph.y)
                        .pre_scale(glyph.scale, glyph.scale);
                    self.draw_path(device, queue, pass, path, &shadow_shader, &t, resolution);
                }
            }
        }

        // Render glyphs.
        for glyph in label.glyphs() {
            if let Some(path) = &glyph.path {
                let t = transform
                    .pre_translate(glyph.x, glyph.y)
                    .pre_scale(glyph.scale, glyph.scale);
                let glyph_shader = if let Some(color) = &glyph.color {
                    FillShader::SolidColor(*color)
                } else {
                    *shader
                };
                self.draw_path(device, queue, pass, path, &glyph_shader, &t, resolution);
            }
        }
    }

    /// Draw an image entity as a textured quad.
    fn draw_image(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pass: &mut wgpu::RenderPass<'_>,
        image: &Handle<WgpuImage>,
        transform: &Transform,
        resolution: [f32; 2],
    ) {
        let texture_bind_group = self.ensure_texture(device, queue, image);

        let uniform_data = ImageUniformData {
            scale: [transform.scale_x, transform.scale_y],
            offset: [transform.x, transform.y],
            resolution,
            brightness: 1.0,
            opacity: 1.0,
            flip_uv_y: 0,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        self.draw_textured_rect(device, queue, pass, &uniform_data, &texture_bind_group);
    }

    /// Ensure an image's pixel data is uploaded as a wgpu texture, returning
    /// a cloneable handle to the bind group for sampling it.
    ///
    /// On first call for a given image, this creates a new texture, uploads
    /// the RGBA pixel data, and caches the bind group. Subsequent calls
    /// return the cached bind group via cheap `Arc` clone.
    ///
    /// # Panics
    ///
    /// Panics if the image's [`RwLock`](std::sync::RwLock) fields are poisoned.
    fn ensure_texture(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        image: &Handle<WgpuImage>,
    ) -> Arc<wgpu::BindGroup> {
        let data = &image.data;

        // Fast path: return cached bind group.
        {
            let bg_lock = data.bind_group.read().expect("bind_group RwLock poisoned");
            if let Some(bg) = bg_lock.as_ref() {
                return Arc::clone(bg);
            }
        }

        // Image source data is sRGB RGBA8, so Rgba8UnormSrgb is the correct
        // format regardless of the output surface format.
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("image_texture"),
            size: wgpu::Extent3d {
                width: data.width,
                height: data.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &data.pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * data.width),
                rows_per_image: Some(data.height),
            },
            wgpu::Extent3d {
                width: data.width,
                height: data.height,
                depth_or_array_layers: 1,
            },
        );

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group = Arc::new(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("image_texture_bind_group"),
            layout: &self.image_texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        }));

        // Cache everything. The texture must stay alive for the bind group
        // to remain valid.
        *data.texture.write().expect("texture RwLock poisoned") = Some(texture);
        *data.bind_group.write().expect("bind_group RwLock poisoned") =
            Some(Arc::clone(&bind_group));

        bind_group
    }

    /// Render the scene background (solid color, gradient, or image fill).
    fn render_background(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pass: &mut wgpu::RenderPass<'_>,
        background: &Background<WgpuImage>,
        resolution: [f32; 2],
    ) {
        match background {
            Background::Shader(shader) => {
                // Full-screen quad using the scene rectangle.
                let transform = Transform {
                    scale_x: resolution[0],
                    scale_y: resolution[1],
                    x: 0.0,
                    y: 0.0,
                };
                let scene = self.scene_manager.scene();
                let rect = scene.rectangle();
                if let Some(path) = rect.as_ref() {
                    self.draw_path(device, queue, pass, path, shader, &transform, resolution);
                }
            }
            Background::Image(bg_image, transform) => {
                self.draw_background_image(device, queue, pass, bg_image, transform, resolution);
            }
        }
    }

    /// Draw a background image with brightness, opacity, and optional blur.
    fn draw_background_image(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pass: &mut wgpu::RenderPass<'_>,
        bg_image: &BackgroundImage<Handle<WgpuImage>>,
        transform: &Transform,
        resolution: [f32; 2],
    ) {
        // Determine which bind group to use: blurred (from pre-computed
        // cache) or original.
        let texture_bind_group: Arc<wgpu::BindGroup> = if bg_image.blur > 0.0 {
            self.blur_cache.as_ref().map_or_else(
                || self.ensure_texture(device, queue, &bg_image.image),
                |c| Arc::clone(&c.bind_group),
            )
        } else {
            self.ensure_texture(device, queue, &bg_image.image)
        };

        let uniform_data = ImageUniformData {
            scale: [transform.scale_x, transform.scale_y],
            offset: [transform.x, transform.y],
            resolution,
            brightness: bg_image.brightness,
            opacity: bg_image.opacity,
            flip_uv_y: 0,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        self.draw_textured_rect(device, queue, pass, &uniform_data, &texture_bind_group);
    }

    /// Draw the scene's unit rectangle as a textured quad with the given
    /// uniforms and texture bind group. Shared by image, background, and
    /// blit draws.
    fn draw_textured_rect(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pass: &mut wgpu::RenderPass<'_>,
        uniform_data: &ImageUniformData,
        texture_bind_group: &wgpu::BindGroup,
    ) {
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("image_uniform_buffer"),
            contents: bytemuck::bytes_of(uniform_data),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("image_uniform_bind_group"),
            layout: &self.image_uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let scene = self.scene_manager.scene();
        let rect = scene.rectangle();
        if let Some(path) = rect.as_ref() {
            let mut bufs = self.buffers.borrow_mut();
            bufs.vertex
                .write(device, queue, bytemuck::cast_slice(&path.vertices));
            bufs.index
                .write(device, queue, bytemuck::cast_slice(&path.indices));

            pass.set_pipeline(&self.image_pipeline);
            pass.set_bind_group(0, &uniform_bind_group, &[]);
            pass.set_bind_group(1, texture_bind_group, &[]);
            pass.set_vertex_buffer(0, bufs.vertex.buffer.slice(..));
            pass.set_index_buffer(bufs.index.buffer.slice(..), wgpu::IndexFormat::Uint32);
            #[expect(clippy::cast_possible_truncation)]
            pass.draw_indexed(0..path.indices.len() as u32, 0, 0..1);
        }
    }

    /// Blit the cached bottom-layer texture to the current render pass as a
    /// fullscreen textured quad.
    fn blit_fbo(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pass: &mut wgpu::RenderPass<'_>,
        resolution: [f32; 2],
    ) {
        let fbo_view = self
            .fbo_texture_view
            .as_ref()
            .expect("FBO texture view not initialized");

        let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit_texture_bind_group"),
            layout: &self.image_texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(fbo_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        let uniform_data = ImageUniformData {
            scale: resolution,
            offset: [0.0, 0.0],
            resolution,
            brightness: 1.0,
            opacity: 1.0,
            flip_uv_y: 1,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        self.draw_textured_rect(device, queue, pass, &uniform_data, &texture_bind_group);
    }

    /// Resize (or initially create) both the resolve texture and MSAA texture
    /// to match the given viewport dimensions.
    fn resize_fbo(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        // Create the resolve target (non-MSAA).
        let fbo_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("fbo_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let fbo_texture_view = fbo_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create the MSAA texture.
        let msaa_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("msaa_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: MSAA_SAMPLES,
            dimension: wgpu::TextureDimension::D2,
            format: self.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let msaa_texture_view = msaa_texture.create_view(&wgpu::TextureViewDescriptor::default());

        self.fbo_texture = Some(fbo_texture);
        self.fbo_texture_view = Some(fbo_texture_view);
        self.msaa_texture = Some(msaa_texture);
        self.msaa_texture_view = Some(msaa_texture_view);
        self.fbo_size = [width, height];
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn path_uniform_data_layout() {
        // Verify field offsets match the WGSL PathUniforms struct.
        assert_eq!(std::mem::offset_of!(PathUniformData, scale), 0);
        assert_eq!(std::mem::offset_of!(PathUniformData, offset), 8);
        assert_eq!(std::mem::offset_of!(PathUniformData, resolution), 16);
        assert_eq!(std::mem::offset_of!(PathUniformData, bounds), 24);
        assert_eq!(std::mem::offset_of!(PathUniformData, color_a), 32);
        assert_eq!(std::mem::offset_of!(PathUniformData, color_b), 48);
        assert_eq!(std::mem::offset_of!(PathUniformData, shader_type), 64);
    }

    #[test]
    fn image_uniform_data_layout() {
        // Verify field offsets match the WGSL ImageUniforms struct.
        assert_eq!(std::mem::offset_of!(ImageUniformData, scale), 0);
        assert_eq!(std::mem::offset_of!(ImageUniformData, offset), 8);
        assert_eq!(std::mem::offset_of!(ImageUniformData, resolution), 16);
        assert_eq!(std::mem::offset_of!(ImageUniformData, brightness), 24);
        assert_eq!(std::mem::offset_of!(ImageUniformData, opacity), 28);
        assert_eq!(std::mem::offset_of!(ImageUniformData, flip_uv_y), 32);
    }
}
