//! The main renderer: owns GL state, consumes livesplit-core [`Scene`] data,
//! and issues draw calls.
//!
//! [`Scene`]: livesplit_core::rendering::Scene

use glow::{HasContext, PixelUnpackData};
use livesplit_core::{
    layout::LayoutState,
    rendering::{Background, Entity, FillShader, Handle, LabelHandle, SceneManager, Transform},
    settings::{BackgroundImage, ImageCache},
};
use std::sync::Arc;

use crate::{
    allocator::GlAllocator,
    common::{tessellate_stroke, vertex_bounds, BLUR_FACTOR, SHADOW_OFFSET},
    shaders,
    types::{GlFont, GlImage, GlLabel, GlPath, Vertex},
};

/// Number of MSAA samples for antialiasing.
const MSAA_SAMPLES: i32 = 4;

/// GL internal format for RGBA8 textures, pre-cast to the `i32` that
/// `tex_image_2d` / `renderbuffer_storage_multisample` expect.
///
#[expect(clippy::cast_possible_wrap)]
const RGBA8_INTERNAL_FORMAT: i32 = glow::RGBA8 as i32;

/// Convert a `u32` to `i32` for GL API calls.
///
/// # Panics
///
/// Panics if `value > i32::MAX`. In practice, this is unreachable for
/// normal viewport dimensions and image sizes.
fn gl_size(value: u32) -> i32 {
    i32::try_from(value).expect("dimension exceeds i32::MAX")
}

/// Cached uniform locations for the path shader program.
struct PathUniforms {
    /// `u_scale` — entity width and height.
    scale: glow::UniformLocation,
    /// `u_offset` — entity translation.
    offset: glow::UniformLocation,
    /// `u_resolution` — viewport size in pixels.
    resolution: glow::UniformLocation,
    /// `u_shader_type` — 0 = solid, 1 = vertical gradient, 2 = horizontal.
    shader_type: glow::UniformLocation,
    /// `u_color_a` — solid color or gradient start.
    color_a: glow::UniformLocation,
    /// `u_color_b` — gradient end color.
    color_b: glow::UniformLocation,
    /// `u_bounds` — `[min, max]` for gradient interpolation axis.
    bounds: glow::UniformLocation,
}

/// Cached uniform locations for the image shader program.
struct ImageUniforms {
    /// `u_scale` — entity width and height.
    scale: glow::UniformLocation,
    /// `u_offset` — entity translation.
    offset: glow::UniformLocation,
    /// `u_resolution` — viewport size in pixels.
    resolution: glow::UniformLocation,
    /// `u_texture` — texture unit index (always 0).
    texture: glow::UniformLocation,
    /// `u_flip_uv_y` — whether to flip the V coordinate.
    flip_uv_y: glow::UniformLocation,
    /// `u_brightness` — brightness multiplier (1.0 = normal).
    brightness: glow::UniformLocation,
    /// `u_opacity` — opacity multiplier (1.0 = fully opaque).
    opacity: glow::UniformLocation,
}

/// Cached blurred background texture.
struct BlurCache {
    /// Identity of the source image (pointer address of its `Arc` data).
    source_ptr: usize,
    /// The blur setting this was computed for.
    blur_value: f32,
    /// The uploaded GL texture containing the blurred pixels.
    texture: glow::Texture,
}

/// A GPU-accelerated renderer for livesplit-core layouts.
///
/// Renders a livesplit-core [`LayoutState`] to the currently-bound OpenGL
/// framebuffer using two shader programs (one for filled/stroked paths, one
/// for textured quads). All paths are tessellated via lyon at creation time
/// and drawn as indexed triangle meshes.
///
/// # Two-layer caching
///
/// The livesplit-core scene is split into a bottom layer (background,
/// component backgrounds, static elements) and a top layer (dynamic text,
/// icons). The bottom layer is rendered to an off-screen MSAA framebuffer,
/// resolved to a texture, and reused across frames when unchanged. This
/// avoids re-rendering the majority of the scene every frame.
///
/// # Example
///
/// ```no_run
/// # use livesplit_renderer_glow::GlowRenderer;
/// # use std::sync::Arc;
/// # fn example(gl: Arc<glow::Context>, state: &livesplit_core::layout::LayoutState,
/// #            image_cache: &livesplit_core::settings::ImageCache) {
/// // During setup (with a current GL context):
/// let mut renderer = unsafe { GlowRenderer::new(gl) }.unwrap();
///
/// // Each frame:
/// let new_size = unsafe { renderer.render(state, image_cache, [800, 600]) };
/// # }
/// ```
pub struct GlowRenderer {
    /// The OpenGL context, shared via [`Arc`] so it can be stored alongside
    /// resources that reference it.
    gl: Arc<glow::Context>,

    /// Resource allocator for paths, images, fonts, and labels.
    allocator: GlAllocator,

    /// Livesplit-core scene manager that diffs layout state into a scene
    /// graph of entities.
    scene_manager: SceneManager<Option<GlPath>, GlImage, GlFont, GlLabel>,

    /// Compiled shader program for filled/stroked paths.
    path_program: glow::Program,
    /// Cached uniform locations for [`path_program`](Self::path_program).
    path_uniforms: PathUniforms,

    /// Compiled shader program for textured quads (images, FBO blitting).
    image_program: glow::Program,
    /// Cached uniform locations for [`image_program`](Self::image_program).
    image_uniforms: ImageUniforms,

    /// Vertex array object with a single `vec2` position attribute.
    vao: glow::VertexArray,
    /// Vertex buffer for streaming path vertex data each frame.
    vbo: glow::Buffer,
    /// Element (index) buffer for streaming path index data each frame.
    ebo: glow::Buffer,

    /// Non-MSAA framebuffer used as the resolve target for the cached bottom
    /// layer.
    fbo: glow::Framebuffer,
    /// Texture attached to [`fbo`](Self::fbo), sampled when compositing the
    /// cached bottom layer.
    fbo_texture: glow::Texture,

    /// MSAA framebuffer used as the rendering target for antialiased
    /// content.
    msaa_fbo: glow::Framebuffer,
    /// MSAA renderbuffer (color attachment) for [`msaa_fbo`](Self::msaa_fbo).
    msaa_rbo: glow::Renderbuffer,

    /// Current dimensions of the off-screen framebuffers.
    fbo_size: [u32; 2],
    /// Whether the cached bottom layer needs re-rendering (e.g., after a
    /// resize).
    bottom_layer_dirty: bool,

    /// Cached blurred background image texture, reused across frames when
    /// the source image and blur setting are unchanged.
    blur_cache: Option<BlurCache>,
}

impl GlowRenderer {
    /// Create a new renderer.
    ///
    /// Compiles shader programs, creates GL buffer objects and framebuffers,
    /// and initializes the livesplit-core scene manager.
    ///
    /// # Safety
    ///
    /// The `gl` context must be current and valid. The caller must ensure
    /// that [`destroy`](Self::destroy) is called before the context is
    /// dropped.
    ///
    /// # Errors
    ///
    /// Returns an error string if shader compilation, program linking, or
    /// GL resource creation fails.
    ///
    /// # Panics
    ///
    /// Panics if any shader uniform location cannot be found, which
    /// indicates a bug in the shader source code.
    #[expect(clippy::too_many_lines)] // GL initialization is inherently verbose
    pub unsafe fn new(gl: Arc<glow::Context>) -> Result<Self, String> {
        let path_program = unsafe {
            shaders::compile_program(&gl, shaders::PATH_VERTEX_SRC, shaders::PATH_FRAGMENT_SRC)?
        };
        let image_program = unsafe {
            shaders::compile_program(&gl, shaders::IMAGE_VERTEX_SRC, shaders::IMAGE_FRAGMENT_SRC)?
        };

        let path_uniforms = unsafe {
            PathUniforms {
                scale: gl
                    .get_uniform_location(path_program, "u_scale")
                    .expect("u_scale missing from path shader"),
                offset: gl
                    .get_uniform_location(path_program, "u_offset")
                    .expect("u_offset missing from path shader"),
                resolution: gl
                    .get_uniform_location(path_program, "u_resolution")
                    .expect("u_resolution missing from path shader"),
                shader_type: gl
                    .get_uniform_location(path_program, "u_shader_type")
                    .expect("u_shader_type missing from path shader"),
                color_a: gl
                    .get_uniform_location(path_program, "u_color_a")
                    .expect("u_color_a missing from path shader"),
                color_b: gl
                    .get_uniform_location(path_program, "u_color_b")
                    .expect("u_color_b missing from path shader"),
                bounds: gl
                    .get_uniform_location(path_program, "u_bounds")
                    .expect("u_bounds missing from path shader"),
            }
        };

        let image_uniforms = unsafe {
            ImageUniforms {
                scale: gl
                    .get_uniform_location(image_program, "u_scale")
                    .expect("u_scale missing from image shader"),
                offset: gl
                    .get_uniform_location(image_program, "u_offset")
                    .expect("u_offset missing from image shader"),
                resolution: gl
                    .get_uniform_location(image_program, "u_resolution")
                    .expect("u_resolution missing from image shader"),
                texture: gl
                    .get_uniform_location(image_program, "u_texture")
                    .expect("u_texture missing from image shader"),
                flip_uv_y: gl
                    .get_uniform_location(image_program, "u_flip_uv_y")
                    .expect("u_flip_uv_y missing from image shader"),
                brightness: gl
                    .get_uniform_location(image_program, "u_brightness")
                    .expect("u_brightness missing from image shader"),
                opacity: gl
                    .get_uniform_location(image_program, "u_opacity")
                    .expect("u_opacity missing from image shader"),
            }
        };

        let (vao, vbo, ebo) = unsafe {
            let vao = gl.create_vertex_array()?;
            let vbo = gl.create_buffer()?;
            let ebo = gl.create_buffer()?;

            // Set up VAO with a single vec2 position attribute.
            gl.bind_vertex_array(Some(vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(ebo));
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(
                0,
                2,
                glow::FLOAT,
                false,
                // Vertex is 8 bytes — well within i32 range.
                #[expect(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
                {
                    std::mem::size_of::<Vertex>() as i32
                },
                0,
            );
            gl.bind_vertex_array(None);

            (vao, vbo, ebo)
        };

        // Create framebuffers (sized lazily on first render).
        // fbo vs rbo are standard GL terminology (framebuffer object vs renderbuffer object).
        let (fbo, fbo_texture, msaa_framebuffer, msaa_renderbuffer) = unsafe {
            let fbo = gl.create_framebuffer()?;
            let fbo_texture = gl.create_texture()?;
            let msaa_framebuffer = gl.create_framebuffer()?;
            let msaa_renderbuffer = gl.create_renderbuffer()?;
            (fbo, fbo_texture, msaa_framebuffer, msaa_renderbuffer)
        };

        let mut allocator = GlAllocator::new();
        let scene_manager = SceneManager::new(&mut allocator);

        Ok(Self {
            gl,
            allocator,
            scene_manager,
            path_program,
            path_uniforms,
            image_program,
            image_uniforms,
            vao,
            vbo,
            ebo,
            fbo,
            fbo_texture,
            msaa_fbo: msaa_framebuffer,
            msaa_rbo: msaa_renderbuffer,
            fbo_size: [0, 0],
            bottom_layer_dirty: true,
            blur_cache: None,
        })
    }

    /// Render the layout into the currently-bound framebuffer (typically the
    /// default framebuffer / screen).
    ///
    /// Returns an optional new resolution hint from livesplit-core's layout
    /// engine, indicating the layout's preferred size changed. The caller
    /// can use this to resize the window or viewport.
    ///
    /// # Safety
    ///
    /// Requires a current GL context matching the one passed to
    /// [`new`](Self::new).
    pub unsafe fn render(
        &mut self,
        state: &LayoutState,
        image_cache: &ImageCache,
        [width, height]: [u32; 2],
    ) -> Option<[f32; 2]> {
        if width == 0 || height == 0 {
            return None;
        }

        // Precision loss is acceptable: viewport dimensions are small
        // relative to f32 mantissa range.
        #[expect(clippy::cast_precision_loss)]
        let resolution = [width as f32, height as f32];

        // Ensure FBOs match the viewport size.
        if self.fbo_size != [width, height] {
            unsafe { self.resize_fbo(width, height) };
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
            unsafe { self.update_blur_cache(&image_data, blur_value) };
        }

        let scene = self.scene_manager.scene();
        let bottom_layer_changed = scene.bottom_layer_changed();

        let gl = &self.gl;

        unsafe {
            // Set up blending for premultiplied alpha.
            gl.enable(glow::BLEND);
            gl.blend_func(glow::ONE, glow::ONE_MINUS_SRC_ALPHA);
        }

        let w = gl_size(width);
        let h = gl_size(height);

        if bottom_layer_changed || self.bottom_layer_dirty {
            // Render bottom layer into MSAA FBO.
            unsafe {
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.msaa_fbo));
                gl.viewport(0, 0, w, h);
                gl.clear_color(0.0, 0.0, 0.0, 0.0);
                gl.clear(glow::COLOR_BUFFER_BIT);
            }

            if let Some(bg) = scene.background() {
                unsafe { self.render_background(bg, resolution) };
            }

            for entity in scene.bottom_layer() {
                unsafe { self.render_entity(entity, resolution) };
            }

            // Resolve MSAA to cached texture.
            unsafe {
                gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(self.msaa_fbo));
                gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(self.fbo));
                gl.blit_framebuffer(
                    0,
                    0,
                    w,
                    h,
                    0,
                    0,
                    w,
                    h,
                    glow::COLOR_BUFFER_BIT,
                    glow::NEAREST,
                );
            }

            self.bottom_layer_dirty = false;
        }

        // Composite: blit cached bottom layer + render top layer into MSAA FBO.
        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.msaa_fbo));
            gl.viewport(0, 0, w, h);
            gl.clear_color(0.0, 0.0, 0.0, 0.0);
            gl.clear(glow::COLOR_BUFFER_BIT);
        }

        // Draw cached bottom layer texture into MSAA FBO.
        unsafe { self.blit_fbo(resolution) };

        // Render top layer into MSAA FBO.
        for entity in scene.top_layer() {
            unsafe { self.render_entity(entity, resolution) };
        }

        // Resolve MSAA to default framebuffer (screen).
        unsafe {
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(self.msaa_fbo));
            gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, None);
            gl.blit_framebuffer(
                0,
                0,
                w,
                h,
                0,
                0,
                w,
                h,
                glow::COLOR_BUFFER_BIT,
                glow::NEAREST,
            );

            gl.disable(glow::BLEND);
        }

        new_resolution
    }

    /// Render a single scene entity.
    unsafe fn render_entity(
        &self,
        entity: &Entity<Option<GlPath>, GlImage, GlLabel>,
        resolution: [f32; 2],
    ) {
        match entity {
            Entity::FillPath(path, shader, transform) => {
                if let Some(path) = path.as_ref() {
                    unsafe { self.draw_path(path, shader, transform, resolution) };
                }
            }
            Entity::StrokePath(path, stroke_width, color, transform) => {
                if let Some(path) = path.as_ref() {
                    if let Some(stroked) = tessellate_stroke(path, *stroke_width) {
                        let shader = FillShader::SolidColor(*color);
                        unsafe { self.draw_path(&stroked, &shader, transform, resolution) };
                    }
                }
            }
            Entity::Image(image, transform) => {
                unsafe { self.draw_image(image, transform, resolution) };
            }
            Entity::Label(label, shader, text_shadow, transform) => {
                unsafe {
                    self.draw_label(label, shader, text_shadow.as_ref(), transform, resolution);
                };
            }
        }
    }

    /// Draw a filled path with the given shader and transform.
    unsafe fn draw_path(
        &self,
        path: &GlPath,
        shader: &FillShader,
        transform: &Transform,
        resolution: [f32; 2],
    ) {
        let gl = &self.gl;

        unsafe {
            gl.use_program(Some(self.path_program));
            gl.uniform_2_f32(
                Some(&self.path_uniforms.resolution),
                resolution[0],
                resolution[1],
            );
            gl.uniform_2_f32(
                Some(&self.path_uniforms.scale),
                transform.scale_x,
                transform.scale_y,
            );
            gl.uniform_2_f32(Some(&self.path_uniforms.offset), transform.x, transform.y);

            self.set_shader_uniforms(shader, path);
            self.upload_and_draw(path);
        }
    }

    /// Configure the fragment shader uniforms for a fill shader.
    ///
    /// For gradient shaders, we compute the bounding box of the path vertices
    /// in local space to determine the interpolation range.
    unsafe fn set_shader_uniforms(&self, shader: &FillShader, path: &GlPath) {
        let gl = &self.gl;
        let u = &self.path_uniforms;

        unsafe {
            match shader {
                FillShader::SolidColor(color) => {
                    gl.uniform_1_i32(Some(&u.shader_type), 0);
                    gl.uniform_4_f32(Some(&u.color_a), color[0], color[1], color[2], color[3]);
                }
                FillShader::VerticalGradient(top, bottom) => {
                    let [min, max] = vertex_bounds(&path.vertices, 1);
                    gl.uniform_1_i32(Some(&u.shader_type), 1);
                    gl.uniform_4_f32(Some(&u.color_a), top[0], top[1], top[2], top[3]);
                    gl.uniform_4_f32(Some(&u.color_b), bottom[0], bottom[1], bottom[2], bottom[3]);
                    gl.uniform_2_f32(Some(&u.bounds), min, max);
                }
                FillShader::HorizontalGradient(left, right) => {
                    let [min, max] = vertex_bounds(&path.vertices, 0);
                    gl.uniform_1_i32(Some(&u.shader_type), 2);
                    gl.uniform_4_f32(Some(&u.color_a), left[0], left[1], left[2], left[3]);
                    gl.uniform_4_f32(Some(&u.color_b), right[0], right[1], right[2], right[3]);
                    gl.uniform_2_f32(Some(&u.bounds), min, max);
                }
            }
        }
    }

    /// Upload vertex/index data and issue the draw call.
    ///
    /// # Panics
    ///
    /// Panics if the index count exceeds `i32::MAX`.
    unsafe fn upload_and_draw(&self, path: &GlPath) {
        let gl = &self.gl;

        unsafe {
            gl.bind_vertex_array(Some(self.vao));

            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.vbo));
            gl.buffer_data_u8_slice(
                glow::ARRAY_BUFFER,
                bytemuck::cast_slice(&path.vertices),
                glow::STREAM_DRAW,
            );

            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(self.ebo));
            gl.buffer_data_u8_slice(
                glow::ELEMENT_ARRAY_BUFFER,
                bytemuck::cast_slice(&path.indices),
                glow::STREAM_DRAW,
            );

            let index_count =
                i32::try_from(path.indices.len()).expect("index count exceeds i32::MAX");
            gl.draw_elements(glow::TRIANGLES, index_count, glow::UNSIGNED_INT, 0);

            gl.bind_vertex_array(None);
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
    unsafe fn draw_label(
        &self,
        label: &LabelHandle<GlLabel>,
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
                    unsafe { self.draw_path(path, &shadow_shader, &t, resolution) };
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
                unsafe { self.draw_path(path, &glyph_shader, &t, resolution) };
            }
        }
    }

    /// Draw an image entity as a textured quad.
    unsafe fn draw_image(
        &self,
        image: &Handle<GlImage>,
        transform: &Transform,
        resolution: [f32; 2],
    ) {
        let gl = &self.gl;
        let texture = unsafe { self.ensure_texture(image) };

        unsafe {
            gl.use_program(Some(self.image_program));
            gl.uniform_2_f32(
                Some(&self.image_uniforms.resolution),
                resolution[0],
                resolution[1],
            );
            gl.uniform_2_f32(
                Some(&self.image_uniforms.scale),
                transform.scale_x,
                transform.scale_y,
            );
            gl.uniform_2_f32(Some(&self.image_uniforms.offset), transform.x, transform.y);
            gl.uniform_1_i32(Some(&self.image_uniforms.flip_uv_y), 0);
            gl.uniform_1_f32(Some(&self.image_uniforms.brightness), 1.0);
            gl.uniform_1_f32(Some(&self.image_uniforms.opacity), 1.0);

            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            gl.uniform_1_i32(Some(&self.image_uniforms.texture), 0);
        }

        // Draw the scene's unit rectangle with this texture.
        let scene = self.scene_manager.scene();
        let rect = scene.rectangle();
        if let Some(path) = rect.as_ref() {
            unsafe { self.upload_and_draw(path) };
        }

        unsafe { gl.bind_texture(glow::TEXTURE_2D, None) };
    }

    /// Ensure an image's pixel data is uploaded as a GL texture, returning
    /// the texture handle.
    ///
    /// On first call for a given image, this creates a new texture, uploads
    /// the RGBA pixel data, and caches the handle. Subsequent calls return
    /// the cached handle.
    ///
    /// # Panics
    ///
    /// Panics if the image's texture [`RwLock`](std::sync::RwLock) is
    /// poisoned, or if the GL context has been lost.
    unsafe fn ensure_texture(&self, image: &Handle<GlImage>) -> glow::Texture {
        let data = &image.data;
        let mut tex_lock = data.texture.write().expect("texture RwLock poisoned");

        if let Some(tex) = *tex_lock {
            return tex;
        }

        let gl = &self.gl;
        let texture = unsafe { gl.create_texture() }.expect("GL context lost: create_texture");
        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                RGBA8_INTERNAL_FORMAT,
                gl_size(data.width),
                gl_size(data.height),
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                PixelUnpackData::Slice(Some(&data.pixels)),
            );
            Self::set_default_tex_params(gl);
            gl.bind_texture(glow::TEXTURE_2D, None);
        }

        *tex_lock = Some(texture);
        texture
    }

    /// Set default texture filtering and wrapping parameters.
    unsafe fn set_default_tex_params(gl: &glow::Context) {
        // GL constant values are small enough that the cast is always safe.
        #[expect(clippy::cast_possible_wrap)]
        unsafe {
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::LINEAR as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::LINEAR as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_S,
                glow::CLAMP_TO_EDGE as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_T,
                glow::CLAMP_TO_EDGE as i32,
            );
        }
    }

    /// Render the scene background (solid color, gradient, or image fill).
    unsafe fn render_background(&self, background: &Background<GlImage>, resolution: [f32; 2]) {
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
                    unsafe { self.draw_path(path, shader, &transform, resolution) };
                }
            }
            Background::Image(bg_image, transform) => {
                unsafe { self.draw_background_image(bg_image, transform, resolution) };
            }
        }
    }

    /// Draw a background image with brightness, opacity, and optional blur.
    ///
    /// # Panics
    ///
    /// Panics if the GL context has been lost.
    unsafe fn draw_background_image(
        &self,
        bg_image: &BackgroundImage<Handle<GlImage>>,
        transform: &Transform,
        resolution: [f32; 2],
    ) {
        let gl = &self.gl;

        // Determine which texture to use: blurred (from pre-computed cache) or original.
        let texture = if bg_image.blur > 0.0 {
            self.blur_cache.as_ref().map_or_else(
                || unsafe { self.ensure_texture(&bg_image.image) },
                |c| c.texture,
            )
        } else {
            unsafe { self.ensure_texture(&bg_image.image) }
        };

        unsafe {
            gl.use_program(Some(self.image_program));
            gl.uniform_2_f32(
                Some(&self.image_uniforms.resolution),
                resolution[0],
                resolution[1],
            );
            gl.uniform_2_f32(
                Some(&self.image_uniforms.scale),
                transform.scale_x,
                transform.scale_y,
            );
            gl.uniform_2_f32(Some(&self.image_uniforms.offset), transform.x, transform.y);
            gl.uniform_1_i32(Some(&self.image_uniforms.flip_uv_y), 0);
            gl.uniform_1_f32(Some(&self.image_uniforms.brightness), bg_image.brightness);
            gl.uniform_1_f32(Some(&self.image_uniforms.opacity), bg_image.opacity);

            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            gl.uniform_1_i32(Some(&self.image_uniforms.texture), 0);
        }

        // Draw the scene's unit rectangle with this texture.
        let scene = self.scene_manager.scene();
        let rect = scene.rectangle();
        if let Some(path) = rect.as_ref() {
            unsafe { self.upload_and_draw(path) };
        }

        unsafe { gl.bind_texture(glow::TEXTURE_2D, None) };
    }

    /// Pre-compute the blurred background texture if needed.
    ///
    /// Called before the render passes while we still have `&mut self`,
    /// so the result can be stored in `self.blur_cache`.
    ///
    /// # Safety
    ///
    /// The GL context must be current and valid.
    ///
    /// # Panics
    ///
    /// Panics if the GL context has been lost.
    unsafe fn update_blur_cache(
        &mut self,
        image_data: &Arc<crate::types::GlImageData>,
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
        let gl = &self.gl;

        // Delete old cached texture before creating a new one.
        if let Some(old_cache) = &self.blur_cache {
            unsafe { gl.delete_texture(old_cache.texture) };
        }

        let texture = unsafe { gl.create_texture() }.expect("GL context lost: create_texture");
        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                RGBA8_INTERNAL_FORMAT,
                gl_size(data.width),
                gl_size(data.height),
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                PixelUnpackData::Slice(Some(&blurred_rgba)),
            );
            Self::set_default_tex_params(gl);
            gl.bind_texture(glow::TEXTURE_2D, None);
        }

        self.blur_cache = Some(BlurCache {
            source_ptr,
            blur_value,
            texture,
        });
    }

    /// Blit the cached bottom-layer FBO texture to the current framebuffer as
    /// a fullscreen textured quad.
    unsafe fn blit_fbo(&self, resolution: [f32; 2]) {
        let gl = &self.gl;

        unsafe {
            gl.use_program(Some(self.image_program));
            gl.uniform_2_f32(
                Some(&self.image_uniforms.resolution),
                resolution[0],
                resolution[1],
            );
            gl.uniform_2_f32(
                Some(&self.image_uniforms.scale),
                resolution[0],
                resolution[1],
            );
            gl.uniform_2_f32(Some(&self.image_uniforms.offset), 0.0, 0.0);
            gl.uniform_1_i32(Some(&self.image_uniforms.flip_uv_y), 1);
            gl.uniform_1_f32(Some(&self.image_uniforms.brightness), 1.0);
            gl.uniform_1_f32(Some(&self.image_uniforms.opacity), 1.0);

            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(self.fbo_texture));
            gl.uniform_1_i32(Some(&self.image_uniforms.texture), 0);
        }

        let scene = self.scene_manager.scene();
        let rect = scene.rectangle();
        if let Some(path) = rect.as_ref() {
            unsafe { self.upload_and_draw(path) };
        }

        unsafe { gl.bind_texture(glow::TEXTURE_2D, None) };
    }

    /// Resize (or initially create) both the resolve FBO and MSAA FBO to
    /// match the given viewport dimensions.
    unsafe fn resize_fbo(&mut self, width: u32, height: u32) {
        let gl = &self.gl;
        let w = gl_size(width);
        let h = gl_size(height);

        unsafe {
            // Set up the resolve target (non-MSAA texture).
            gl.bind_texture(glow::TEXTURE_2D, Some(self.fbo_texture));
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                RGBA8_INTERNAL_FORMAT,
                w,
                h,
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                PixelUnpackData::Slice(None),
            );
            // GL constant values are small enough that the cast is always safe.
            #[expect(clippy::cast_possible_wrap)]
            {
                gl.tex_parameter_i32(
                    glow::TEXTURE_2D,
                    glow::TEXTURE_MIN_FILTER,
                    glow::LINEAR as i32,
                );
                gl.tex_parameter_i32(
                    glow::TEXTURE_2D,
                    glow::TEXTURE_MAG_FILTER,
                    glow::LINEAR as i32,
                );
            }

            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.fbo));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(self.fbo_texture),
                0,
            );
            debug_assert_eq!(
                gl.check_framebuffer_status(glow::FRAMEBUFFER),
                glow::FRAMEBUFFER_COMPLETE,
                "resolve FBO incomplete",
            );

            // Set up the MSAA renderbuffer.
            gl.bind_renderbuffer(glow::RENDERBUFFER, Some(self.msaa_rbo));
            gl.renderbuffer_storage_multisample(
                glow::RENDERBUFFER,
                MSAA_SAMPLES,
                glow::RGBA8,
                w,
                h,
            );

            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.msaa_fbo));
            gl.framebuffer_renderbuffer(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::RENDERBUFFER,
                Some(self.msaa_rbo),
            );
            debug_assert_eq!(
                gl.check_framebuffer_status(glow::FRAMEBUFFER),
                glow::FRAMEBUFFER_COMPLETE,
                "MSAA FBO incomplete",
            );

            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.bind_renderbuffer(glow::RENDERBUFFER, None);
            gl.bind_texture(glow::TEXTURE_2D, None);
        }

        self.fbo_size = [width, height];
    }

    /// Clean up all GL resources owned by this renderer.
    ///
    /// # Safety
    ///
    /// Must be called with the same GL context that was used to create the
    /// renderer, and must be called exactly once.
    pub unsafe fn destroy(&self) {
        let gl = &self.gl;
        unsafe {
            gl.delete_program(self.path_program);
            gl.delete_program(self.image_program);
            gl.delete_vertex_array(self.vao);
            gl.delete_buffer(self.vbo);
            gl.delete_buffer(self.ebo);
            gl.delete_framebuffer(self.fbo);
            gl.delete_texture(self.fbo_texture);
            gl.delete_framebuffer(self.msaa_fbo);
            gl.delete_renderbuffer(self.msaa_rbo);
        }
        if let Some(cache) = &self.blur_cache {
            unsafe { gl.delete_texture(cache.texture) };
        }
    }
}
