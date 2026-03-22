//! WGSL shader sources and pipeline creation helpers.
//!
//! All shaders use WGSL, targeting wgpu's shader model.

/// Vertex shader for filled/stroked paths.
///
/// Transforms vertices by the entity's scale+translate transform, and passes
/// the *local-space* position to the fragment shader for gradient
/// interpolation.
///
/// # Uniforms (via bind group 0)
///
/// | Name           | Type   | Description                              |
/// |----------------|--------|------------------------------------------|
/// | `scale`        | `vec2f`| Entity scale (width, height)             |
/// | `offset`       | `vec2f`| Entity translation (x, y)                |
/// | `resolution`   | `vec2f`| Viewport size in pixels                  |
/// | `shader_type`  | `i32`  | 0 = solid, 1 = vertical gradient, 2 = horizontal |
/// | `color_a`      | `vec4f`| Solid color or gradient start            |
/// | `color_b`      | `vec4f`| Gradient end color                       |
/// | `bounds`       | `vec2f`| [min, max] for gradient axis             |
pub const PATH_SHADER_SRC: &str = r"
struct PathUniforms {
    scale: vec2f,
    offset: vec2f,
    resolution: vec2f,
    bounds: vec2f,
    color_a: vec4f,
    color_b: vec4f,
    shader_type: i32,
    _pad0: i32,
    _pad1: i32,
    _pad2: i32,
}

@group(0) @binding(0)
var<uniform> u: PathUniforms;

struct VertexInput {
    @location(0) position: vec2f,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) local: vec2f,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.local = in.position;

    let world = u.offset + u.scale * in.position;

    // Convert from [0, resolution] to [-1, 1] (flip Y for clip space)
    var ndc = (world / u.resolution) * 2.0 - 1.0;
    ndc.y = -ndc.y;

    out.clip_position = vec4f(ndc, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    var frag_color: vec4f;

    if u.shader_type == 0 {
        // Solid color
        frag_color = u.color_a;
    } else {
        // Gradient
        var coord: f32;
        if u.shader_type == 1 {
            coord = in.local.y;  // vertical
        } else {
            coord = in.local.x;  // horizontal
        }
        let range = u.bounds.y - u.bounds.x;
        var t: f32;
        if range > 0.0 {
            t = clamp((coord - u.bounds.x) / range, 0.0, 1.0);
        } else {
            t = 0.0;
        }
        frag_color = mix(u.color_a, u.color_b, t);
    }

    // Premultiply alpha for correct blending
    frag_color = vec4f(frag_color.rgb * frag_color.a, frag_color.a);
    return frag_color;
}
";

/// Combined vertex and fragment shader for textured quads (images).
///
/// The image entity uses the scene's unit rectangle `[0,1]x[0,1]` transformed
/// by the entity transform. UV coordinates are derived from the local vertex
/// position, with an optional Y-flip for framebuffer blitting.
///
/// # Uniforms (via bind group 0)
///
/// | Name           | Type    | Description                           |
/// |----------------|---------|---------------------------------------|
/// | `scale`        | `vec2f` | Entity scale (width, height)          |
/// | `offset`       | `vec2f` | Entity translation (x, y)             |
/// | `resolution`   | `vec2f` | Viewport size in pixels               |
/// | `flip_uv_y`    | `i32`   | Flip V coordinate (for FBO blitting)  |
/// | `brightness`   | `f32`   | Brightness multiplier (1.0 = normal)  |
/// | `opacity`      | `f32`   | Opacity multiplier (1.0 = opaque)     |
pub const IMAGE_SHADER_SRC: &str = r"
struct ImageUniforms {
    scale: vec2f,
    offset: vec2f,
    resolution: vec2f,
    brightness: f32,
    opacity: f32,
    flip_uv_y: i32,
    _pad0: i32,
    _pad1: i32,
    _pad2: i32,
}

@group(0) @binding(0)
var<uniform> iu: ImageUniforms;

@group(1) @binding(0)
var t_texture: texture_2d<f32>;
@group(1) @binding(1)
var s_sampler: sampler;

struct VertexInput {
    @location(0) position: vec2f,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.uv = in.position;
    if iu.flip_uv_y != 0 {
        out.uv.y = 1.0 - out.uv.y;
    }

    let world = iu.offset + iu.scale * in.position;
    var ndc = (world / iu.resolution) * 2.0 - 1.0;
    ndc.y = -ndc.y;

    out.clip_position = vec4f(ndc, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    var frag_color = textureSample(t_texture, s_sampler, in.uv);
    frag_color = vec4f(frag_color.rgb * iu.brightness, frag_color.a);
    frag_color = vec4f(frag_color.rgb, frag_color.a * iu.opacity);
    // Premultiply alpha
    frag_color = vec4f(frag_color.rgb * frag_color.a, frag_color.a);
    return frag_color;
}
";

/// Create a wgpu render pipeline for path rendering.
///
/// # Panics
///
/// Panics if shader compilation fails (indicates a bug in the shader source).
pub fn create_path_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    path_bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("path_shader"),
        source: wgpu::ShaderSource::Wgsl(PATH_SHADER_SRC.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("path_pipeline_layout"),
        bind_group_layouts: &[path_bind_group_layout],
        push_constant_ranges: &[],
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("path_pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<super::wgpu_types::Vertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 0,
                    shader_location: 0,
                }],
            }],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: super::wgpu_render::MSAA_SAMPLES,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        multiview: None,
        cache: None,
    })
}

/// Create a wgpu render pipeline for image rendering.
///
/// # Panics
///
/// Panics if shader compilation fails (indicates a bug in the shader source).
pub fn create_image_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    image_uniform_bind_group_layout: &wgpu::BindGroupLayout,
    image_texture_bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("image_shader"),
        source: wgpu::ShaderSource::Wgsl(IMAGE_SHADER_SRC.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("image_pipeline_layout"),
        bind_group_layouts: &[
            image_uniform_bind_group_layout,
            image_texture_bind_group_layout,
        ],
        push_constant_ranges: &[],
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("image_pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<super::wgpu_types::Vertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 0,
                    shader_location: 0,
                }],
            }],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: super::wgpu_render::MSAA_SAMPLES,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        multiview: None,
        cache: None,
    })
}
