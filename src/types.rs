//! Resource types for the GL renderer.
//!
//! Path, vertex, font, and label types are shared with the wgpu renderer
//! via [`crate::common`]. This module adds the GL-specific [`GlImage`] type
//! with its lazy texture handle.

use std::sync::{Arc, RwLock};

use livesplit_core::rendering::{self, SharedOwnership};

// Re-export shared types under GL-prefixed aliases for backwards
// compatibility and readability in the GL-specific code.
pub use crate::common::Font as GlFont;
pub use crate::common::Label as GlLabel;
#[allow(unused_imports)]
pub use crate::common::LockedLabel as GlLockedLabel;
pub use crate::common::Path as GlPath;
pub use crate::common::Vertex;

/// A decoded image ready for GL texture upload.
///
/// The raw pixel data is shared via [`Arc`] so that cloning an image is
/// cheap. The GL texture handle is lazily created on first draw.
#[derive(Clone)]
pub struct GlImage {
    /// Shared image data (pixels, dimensions, and cached texture handle).
    pub data: Arc<GlImageData>,
}

/// Backing store for a [`GlImage`].
///
/// Contains the decoded RGBA pixel data and an optional GL texture handle
/// that is populated on first use.
pub struct GlImageData {
    /// Raw pixel data in RGBA8 format, row-major, top-to-bottom.
    pub pixels: Vec<u8>,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Precomputed width / height, used by the scene layout engine.
    pub aspect_ratio: f32,
    /// GL texture name, lazily uploaded on first draw. `None` until then.
    pub texture: RwLock<Option<glow::Texture>>,
}

impl rendering::Image for GlImage {
    fn aspect_ratio(&self) -> f32 {
        self.data.aspect_ratio
    }
}

impl SharedOwnership for GlImage {
    fn share(&self) -> Self {
        self.clone()
    }
}
