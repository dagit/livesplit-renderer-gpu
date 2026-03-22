//! Resource types for the wgpu renderer.
//!
//! Path, vertex, font, and label types are shared with the glow renderer
//! via [`crate::common`]. This module adds the wgpu-specific [`WgpuImage`]
//! type with its lazy texture and bind group handles.

use std::sync::{Arc, RwLock};

use livesplit_core::rendering::{self, SharedOwnership};

// Re-export shared types under Wgpu-prefixed aliases for readability
// in the wgpu-specific code.
pub use crate::common::Font as WgpuFont;
pub use crate::common::Label as WgpuLabel;
#[allow(unused_imports)]
pub use crate::common::LockedLabel as WgpuLockedLabel;
pub use crate::common::Path as WgpuPath;
pub use crate::common::Vertex;

/// A decoded image ready for wgpu texture upload.
///
/// The raw pixel data is shared via [`Arc`] so that cloning an image is
/// cheap. The wgpu texture handle is lazily created on first draw.
#[derive(Clone)]
pub struct WgpuImage {
    /// Shared image data (pixels, dimensions, and cached texture handle).
    pub data: Arc<WgpuImageData>,
}

/// Backing store for a [`WgpuImage`].
///
/// Contains the decoded RGBA pixel data and an optional wgpu texture and
/// bind group that are populated on first use.
pub struct WgpuImageData {
    /// Raw pixel data in RGBA8 format, row-major, top-to-bottom.
    pub pixels: Vec<u8>,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Precomputed width / height, used by the scene layout engine.
    pub aspect_ratio: f32,
    /// Wgpu texture, lazily uploaded on first draw. `None` until then.
    /// Kept alive so the bind group remains valid.
    pub texture: RwLock<Option<wgpu::Texture>>,
    /// Wgpu bind group for sampling this texture, lazily created on first
    /// draw. Wrapped in [`Arc`] so it can be cheaply cloned out of the lock.
    pub bind_group: RwLock<Option<Arc<wgpu::BindGroup>>>,
}

impl rendering::Image for WgpuImage {
    fn aspect_ratio(&self) -> f32 {
        self.data.aspect_ratio
    }
}

impl SharedOwnership for WgpuImage {
    fn share(&self) -> Self {
        self.clone()
    }
}
