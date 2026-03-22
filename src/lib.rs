//! A GPU-accelerated renderer for [livesplit-core] using OpenGL via [glow].
//!
//! This crate provides [`GlowRenderer`], which renders livesplit-core layout
//! scenes to an OpenGL framebuffer. Paths are tessellated via [lyon] at
//! creation time and rendered as indexed triangle meshes. The scene's two-layer
//! design is honored: the bottom layer is cached in a framebuffer object and
//! only re-rendered when it changes.
//!
//! # Features
//!
//! - **4× MSAA** antialiasing on all rendered content.
//! - **Two-layer caching**: the bottom layer is rendered to an off-screen
//!   texture and reused across frames when unchanged.
//! - **Gradient fills**: solid, vertical, and horizontal gradients are
//!   supported natively in the fragment shader.
//! - **Text rendering** via livesplit-core's built-in text engine, with
//!   optional text shadows.
//! - **Lazy texture upload**: images are decoded on the CPU and uploaded to
//!   the GPU only when first drawn.
//!
//! # Safety
//!
//! Creating and using a [`GlowRenderer`] requires a valid, current OpenGL
//! context. All rendering methods are `unsafe` because they issue raw GL
//! calls.
//!
//! [livesplit-core]: https://github.com/LiveSplit/livesplit-core
//! [glow]: https://docs.rs/glow
//! [lyon]: https://docs.rs/lyon

mod common;

mod allocator;
mod render;
mod shaders;
mod types;

mod wgpu_allocator;
mod wgpu_render;
mod wgpu_shaders;
mod wgpu_types;

pub use render::GlowRenderer;
pub use wgpu_render::WgpuRenderer;
