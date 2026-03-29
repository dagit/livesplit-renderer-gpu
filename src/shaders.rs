//! GLSL shader sources and compilation helpers.
//!
//! All shaders target GLSL 1.40 (OpenGL 3.1), which is widely supported on
//! desktop platforms.

use glow::HasContext;

/// Vertex shader for filled/stroked paths.
///
/// Transforms vertices by the entity's scale+translate transform, and passes
/// the *local-space* position to the fragment shader for gradient
/// interpolation.
///
/// # Uniforms
///
/// | Name           | Type   | Description                              |
/// |----------------|--------|------------------------------------------|
/// | `u_scale`      | `vec2` | Entity scale (width, height)             |
/// | `u_offset`     | `vec2` | Entity translation (x, y)                |
/// | `u_resolution` | `vec2` | Viewport size in pixels                  |
pub const PATH_VERTEX_SRC: &str = r"#version 140

in vec2 a_position;

// Entity transform: output = offset + scale * input
uniform vec2 u_scale;
uniform vec2 u_offset;

// Viewport resolution for NDC conversion
uniform vec2 u_resolution;

// Local-space position for gradient interpolation
out vec2 v_local;

void main() {
    v_local = a_position;

    vec2 world = u_offset + u_scale * a_position;

    // Convert from [0, resolution] to [-1, 1] (flip Y for GL)
    vec2 ndc = (world / u_resolution) * 2.0 - 1.0;
    ndc.y = -ndc.y;

    gl_Position = vec4(ndc, 0.0, 1.0);
}
";

/// Fragment shader for filled paths.
///
/// Supports three shader modes via `u_shader_type`:
///
/// | Value | Mode                | Interpolation axis |
/// |-------|---------------------|--------------------|
/// | `0`   | Solid color         | —                  |
/// | `1`   | Vertical gradient   | local Y            |
/// | `2`   | Horizontal gradient | local X            |
///
/// All output colors are premultiplied by alpha before writing.
pub const PATH_FRAGMENT_SRC: &str = r"#version 140

in vec2 v_local;

uniform int u_shader_type;
uniform vec4 u_color_a;   // solid color, or gradient start
uniform vec4 u_color_b;   // gradient end (unused for solid)
uniform vec2 u_bounds;    // [min, max] for gradient axis

out vec4 frag_color;

void main() {
    if (u_shader_type == 0) {
        // Solid color
        frag_color = u_color_a;
    } else {
        // Gradient
        float coord;
        if (u_shader_type == 1) {
            coord = v_local.y;  // vertical
        } else {
            coord = v_local.x;  // horizontal
        }
        float range = u_bounds.y - u_bounds.x;
        float t = (range > 0.0) ? clamp((coord - u_bounds.x) / range, 0.0, 1.0) : 0.0;
        frag_color = mix(u_color_a, u_color_b, t);
    }

    // Premultiply alpha for correct blending
    frag_color.rgb *= frag_color.a;
}
";

/// Vertex shader for textured quads (images).
///
/// The image entity uses the scene's unit rectangle `[0,1]x[0,1]` transformed
/// by the entity transform. UV coordinates are derived from the local vertex
/// position, with an optional Y-flip for framebuffer blitting.
///
/// # Uniforms
///
/// | Name           | Type    | Description                           |
/// |----------------|---------|---------------------------------------|
/// | `u_scale`      | `vec2`  | Entity scale (width, height)          |
/// | `u_offset`     | `vec2`  | Entity translation (x, y)             |
/// | `u_resolution` | `vec2`  | Viewport size in pixels               |
/// | `u_flip_uv_y`  | `bool`  | Flip V coordinate (for FBO blitting)  |
pub const IMAGE_VERTEX_SRC: &str = r"#version 140

in vec2 a_position;

uniform vec2 u_scale;
uniform vec2 u_offset;
uniform vec2 u_resolution;
uniform bool u_flip_uv_y;

out vec2 v_uv;

void main() {
    v_uv = a_position;
    if (u_flip_uv_y) {
        v_uv.y = 1.0 - v_uv.y;
    }

    vec2 world = u_offset + u_scale * a_position;
    vec2 ndc = (world / u_resolution) * 2.0 - 1.0;
    ndc.y = -ndc.y;

    gl_Position = vec4(ndc, 0.0, 1.0);
}
";

/// Fragment shader for textured quads.
///
/// Samples the bound texture, applies brightness and opacity adjustments,
/// and premultiplies the result by alpha.
///
/// # Uniforms
///
/// | Name                     | Type        | Description                                        |
/// |--------------------------|-------------|----------------------------------------------------|
/// | `u_texture`              | `sampler2D` | Bound texture unit                                 |
/// | `u_brightness`           | `float`     | Brightness multiplier (1.0 = normal)               |
/// | `u_opacity`              | `float`     | Opacity multiplier (1.0 = opaque)                  |
/// | `u_already_premultiplied`| `int`       | If non-zero, skip alpha premultiplication (FBO blit)|
pub const IMAGE_FRAGMENT_SRC: &str = r"#version 140

in vec2 v_uv;

uniform sampler2D u_texture;
uniform float u_brightness;
uniform float u_opacity;
uniform int u_already_premultiplied;

out vec4 frag_color;

void main() {
    frag_color = texture(u_texture, v_uv);
    frag_color.rgb *= u_brightness;
    frag_color.a *= u_opacity;
    // Premultiply alpha (skip if content is already premultiplied, e.g. FBO blit)
    if (u_already_premultiplied == 0) {
        frag_color.rgb *= frag_color.a;
    }
}
";

/// Compile a shader program from vertex and fragment source strings.
///
/// The compiled shader objects are detached and deleted after successful
/// linking, so only the program handle needs to be cleaned up by the caller.
///
/// # Safety
///
/// Requires a valid, current OpenGL context.
///
/// # Errors
///
/// Returns a descriptive error string if shader compilation or program
/// linking fails.
pub unsafe fn compile_program(
    gl: &glow::Context,
    vertex_src: &str,
    fragment_src: &str,
) -> Result<glow::Program, String> {
    let program = unsafe { gl.create_program() }?;

    let vs = unsafe { compile_shader(gl, glow::VERTEX_SHADER, vertex_src) }?;
    let fs = unsafe { compile_shader(gl, glow::FRAGMENT_SHADER, fragment_src) }?;

    unsafe {
        gl.attach_shader(program, vs);
        gl.attach_shader(program, fs);
        gl.link_program(program);

        if !gl.get_program_link_status(program) {
            let log = gl.get_program_info_log(program);
            gl.delete_program(program);
            gl.delete_shader(vs);
            gl.delete_shader(fs);
            return Err(format!("Program link error: {log}"));
        }

        // Shaders can be detached and deleted after successful linking.
        gl.detach_shader(program, vs);
        gl.detach_shader(program, fs);
        gl.delete_shader(vs);
        gl.delete_shader(fs);
    }

    Ok(program)
}

/// Compile a single shader stage (vertex or fragment) from source.
///
/// # Safety
///
/// Requires a valid, current OpenGL context.
unsafe fn compile_shader(
    gl: &glow::Context,
    shader_type: u32,
    source: &str,
) -> Result<glow::Shader, String> {
    unsafe {
        let shader = gl.create_shader(shader_type)?;
        gl.shader_source(shader, source);
        gl.compile_shader(shader);

        if !gl.get_shader_compile_status(shader) {
            let log = gl.get_shader_info_log(shader);
            gl.delete_shader(shader);
            return Err(format!("Shader compile error: {log}"));
        }

        Ok(shader)
    }
}
