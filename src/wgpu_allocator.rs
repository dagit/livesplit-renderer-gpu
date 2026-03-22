//! [`ResourceAllocator`] implementation that tessellates paths via lyon and
//! delegates text shaping to livesplit-core's default text engine.

use livesplit_core::{
    rendering::{default_text_engine::TextEngine, FontKind, ResourceAllocator},
    settings,
};
use std::sync::Arc;

use crate::common::CommonPathBuilder;
use crate::wgpu_types::{WgpuFont, WgpuImage, WgpuImageData, WgpuLabel, WgpuPath};

/// The resource allocator that wires together path tessellation (via lyon)
/// and text shaping (via livesplit-core's default text engine).
pub struct WgpuAllocator {
    /// Text engine instance used for font loading, glyph shaping, and label
    /// management.
    pub(crate) text_engine: TextEngine<Option<WgpuPath>>,
}

impl WgpuAllocator {
    /// Create a new allocator with a fresh text engine.
    pub fn new() -> Self {
        Self {
            text_engine: TextEngine::new(),
        }
    }
}

impl Default for WgpuAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl ResourceAllocator for WgpuAllocator {
    type PathBuilder = CommonPathBuilder;
    type Path = Option<WgpuPath>;
    type Image = WgpuImage;
    type Font = WgpuFont;
    type Label = WgpuLabel;

    fn path_builder(&mut self) -> Self::PathBuilder {
        CommonPathBuilder::new()
    }

    fn create_image(&mut self, data: &[u8]) -> Option<Self::Image> {
        let img = image::load_from_memory(data).ok()?.to_rgba8();
        let (width, height) = img.dimensions();
        Some(WgpuImage {
            data: Arc::new(WgpuImageData {
                pixels: img.into_raw(),
                width,
                height,
                // Precision loss is acceptable: viewport dimensions are small
                // relative to f32 mantissa range.
                #[expect(clippy::cast_precision_loss)]
                aspect_ratio: width as f32 / height as f32,
                texture: std::sync::RwLock::new(None),
                bind_group: std::sync::RwLock::new(None),
            }),
        })
    }

    fn create_font(&mut self, font: Option<&settings::Font>, kind: FontKind) -> Self::Font {
        self.text_engine.create_font(font, kind)
    }

    fn create_label(
        &mut self,
        text: &str,
        font: &mut Self::Font,
        max_width: Option<f32>,
    ) -> Self::Label {
        self.text_engine
            .create_label(CommonPathBuilder::new, text, font, max_width)
    }

    fn update_label(
        &mut self,
        label: &mut Self::Label,
        text: &str,
        font: &mut Self::Font,
        max_width: Option<f32>,
    ) {
        self.text_engine
            .update_label(CommonPathBuilder::new, label, text, font, max_width);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use image::ImageEncoder;

    use super::*;
    use livesplit_core::rendering::ResourceAllocator;

    #[test]
    fn create_image_with_invalid_data_returns_none() {
        let mut alloc = WgpuAllocator::new();
        assert!(alloc.create_image(b"not an image").is_none());
    }

    #[test]
    fn create_image_with_valid_png() {
        // Encode a 2x1 PNG at runtime so the bytes are always correct.
        let mut buf = std::io::Cursor::new(Vec::new());
        {
            let encoder = image::codecs::png::PngEncoder::new(&mut buf);
            // 2x1 RGBA image: red pixel, blue pixel
            let pixels: &[u8] = &[255, 0, 0, 255, 0, 0, 255, 255];
            encoder
                .write_image(pixels, 2, 1, image::ExtendedColorType::Rgba8)
                .unwrap();
        }

        let mut alloc = WgpuAllocator::new();
        let image = alloc.create_image(buf.get_ref());
        assert!(image.is_some(), "valid PNG should produce an image");

        let img = image.unwrap();
        assert_eq!(img.data.width, 2);
        assert_eq!(img.data.height, 1);
        assert!((img.data.aspect_ratio - 2.0).abs() < f32::EPSILON);
        // RGBA: 4 bytes per pixel x 2 pixels
        assert_eq!(img.data.pixels.len(), 8);
    }

    #[test]
    fn allocator_default_matches_new() {
        let _alloc: WgpuAllocator = WgpuAllocator::default();
    }
}
