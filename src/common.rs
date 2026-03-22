//! Types and utilities shared between the glow and wgpu renderers.

/// Shadow offset in component coordinate space.
pub(crate) const SHADOW_OFFSET: f32 = 0.05;

/// Blur sigma scale factor applied to the larger image dimension.
/// Matches livesplit-core's `BLUR_FACTOR`.
pub(crate) const BLUR_FACTOR: f32 = 0.05;

use std::sync::{Arc, RwLock};

use bytemuck::{Pod, Zeroable};
use livesplit_core::rendering::SharedOwnership;
use lyon::path::Path as LyonPath;

/// A vertex in a tessellated path, ready for the GPU.
///
/// Uses `#[repr(C)]` and derives [`Pod`] so the vertex array can be
/// reinterpreted as a byte slice for GPU upload.
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct Vertex {
    /// X and Y position in local (pre-transform) coordinate space.
    pub position: [f32; 2],
}

/// A tessellated path stored as indexed triangle data.
///
/// Created at `PathBuilder::finish()` time via lyon tessellation.
///
/// Vertices and indices are wrapped in [`Arc`] so that
/// [`SharedOwnership::share`] is a cheap reference-count bump.
/// The original lyon path is retained for stroke tessellation.
#[allow(clippy::struct_field_names)]
pub struct Path {
    /// Triangle vertices in local coordinate space.
    pub vertices: Arc<Vec<Vertex>>,
    /// Triangle indices into [`vertices`](Self::vertices).
    pub indices: Arc<Vec<u32>>,
    /// The original lyon path, retained for stroke tessellation.
    pub lyon_path: Arc<LyonPath>,
    /// Cached stroke tessellation, keyed by stroke width.
    stroke_cache: RwLock<Option<StrokeCache>>,
}

/// Shared vertex and index buffers for a tessellated path.
type StrokeGeometry = (Arc<Vec<Vertex>>, Arc<Vec<u32>>);

/// Cached stroke tessellation data for a specific line width.
struct StrokeCache {
    /// The stroke width this cache was tessellated for.
    width: f32,
    /// Stroke triangle vertices.
    vertices: Arc<Vec<Vertex>>,
    /// Stroke triangle indices.
    indices: Arc<Vec<u32>>,
}

impl Path {
    /// Create a new `Path` from tessellated geometry and the original path.
    pub fn new(vertices: Vec<Vertex>, indices: Vec<u32>, lyon_path: Arc<LyonPath>) -> Self {
        Self {
            vertices: Arc::new(vertices),
            indices: Arc::new(indices),
            lyon_path,
            stroke_cache: RwLock::new(None),
        }
    }

    /// Create a `Path` from pre-shared vertex and index buffers.
    pub fn from_arcs(
        vertices: Arc<Vec<Vertex>>,
        indices: Arc<Vec<u32>>,
        lyon_path: Arc<LyonPath>,
    ) -> Self {
        Self {
            vertices,
            indices,
            lyon_path,
            stroke_cache: RwLock::new(None),
        }
    }

    /// Get the cached stroke tessellation for a given width, or `None` if
    /// the cache is empty or was tessellated for a different width.
    pub fn cached_stroke(&self, width: f32) -> Option<StrokeGeometry> {
        let cache = self
            .stroke_cache
            .read()
            .expect("stroke cache RwLock poisoned");
        cache.as_ref().and_then(|c| {
            if (c.width - width).abs() < f32::EPSILON {
                Some((Arc::clone(&c.vertices), Arc::clone(&c.indices)))
            } else {
                None
            }
        })
    }

    /// Store a stroke tessellation in the cache for a given width.
    pub fn set_stroke_cache(&self, width: f32, vertices: Arc<Vec<Vertex>>, indices: Arc<Vec<u32>>) {
        let mut cache = self
            .stroke_cache
            .write()
            .expect("stroke cache RwLock poisoned");
        *cache = Some(StrokeCache {
            width,
            vertices,
            indices,
        });
    }
}

impl Clone for Path {
    fn clone(&self) -> Self {
        Self {
            vertices: Arc::clone(&self.vertices),
            indices: Arc::clone(&self.indices),
            lyon_path: Arc::clone(&self.lyon_path),
            // Start with an empty cache — it will be populated on first stroke draw.
            stroke_cache: RwLock::new(None),
        }
    }
}

impl std::fmt::Debug for Path {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Path")
            .field("vertices", &self.vertices.len())
            .field("indices", &self.indices.len())
            .finish_non_exhaustive()
    }
}

impl SharedOwnership for Path {
    fn share(&self) -> Self {
        self.clone()
    }
}

/// Font handle.
///
/// All font loading and shaping is delegated to livesplit-core's default
/// text engine, so this is a re-export of its font type.
pub type Font = livesplit_core::rendering::default_text_engine::Font;

/// Label handle.
///
/// The default text engine produces labels containing glyph outlines
/// tessellated as [`Path`] values.
pub type Label = livesplit_core::rendering::default_text_engine::Label<Option<Path>>;

/// The locked (read-guard) form of a [`Label`].
#[allow(dead_code)]
pub type LockedLabel = livesplit_core::rendering::default_text_engine::LockedLabel<Option<Path>>;

/// Compute the min/max of a single axis across all path vertices.
///
/// Used to determine the interpolation range for gradient shaders.
/// `axis` is the index into `Vertex::position` (0 = X, 1 = Y).
///
/// Returns `[0.0, 0.0]` for an empty vertex slice.
pub fn vertex_bounds(vertices: &[Vertex], axis: usize) -> [f32; 2] {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for v in vertices {
        let val = v.position[axis];
        min = min.min(val);
        max = max.max(val);
    }
    if max < min {
        [0.0, 0.0]
    } else {
        [min, max]
    }
}

/// Lyon-backed path builder that produces a [`Path`] on `finish()`.
///
/// Implements livesplit-core's [`PathBuilder`](livesplit_core::rendering::PathBuilder) trait,
/// converting path commands (move/line/quad/curve/close) into a lyon path and
/// then tessellating it into an indexed triangle mesh.
pub struct CommonPathBuilder {
    /// The underlying lyon path builder.
    builder: lyon::path::path::Builder,
    /// Whether a sub-path has been opened (via `begin`) but not yet closed.
    /// Lyon panics if `build()` is called while a sub-path is still open.
    sub_path_open: bool,
}

impl CommonPathBuilder {
    /// Create a new path builder.
    pub fn new() -> Self {
        Self {
            builder: LyonPath::builder(),
            sub_path_open: false,
        }
    }
}

impl livesplit_core::rendering::PathBuilder for CommonPathBuilder {
    type Path = Option<Path>;

    fn move_to(&mut self, x: f32, y: f32) {
        if self.sub_path_open {
            self.builder.end(false);
        }
        self.builder.begin(lyon::math::point(x, y));
        self.sub_path_open = true;
    }

    fn line_to(&mut self, x: f32, y: f32) {
        self.builder.line_to(lyon::math::point(x, y));
    }

    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        self.builder
            .quadratic_bezier_to(lyon::math::point(x1, y1), lyon::math::point(x, y));
    }

    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        self.builder.cubic_bezier_to(
            lyon::math::point(x1, y1),
            lyon::math::point(x2, y2),
            lyon::math::point(x, y),
        );
    }

    fn close(&mut self) {
        self.builder.close();
        self.sub_path_open = false;
    }

    fn finish(mut self) -> Self::Path {
        if self.sub_path_open {
            self.builder.end(false);
        }
        let path = self.builder.build();
        tessellate_path(&path)
    }
}

/// Tessellate a lyon path into an indexed triangle mesh (fill).
///
/// Uses a fill tessellator with the non-zero fill rule and a tolerance of
/// 0.01 (suitable for the small coordinate spaces livesplit-core uses).
///
/// Returns `None` if tessellation fails or produces no vertices.
pub fn tessellate_path(path: &LyonPath) -> Option<Path> {
    use lyon::tessellation::{
        BuffersBuilder, FillOptions, FillRule, FillTessellator, FillVertex, VertexBuffers,
    };

    let mut geometry: VertexBuffers<Vertex, u32> = VertexBuffers::new();
    let mut tessellator = FillTessellator::new();

    let result = tessellator.tessellate_path(
        path,
        &FillOptions::tolerance(0.01).with_fill_rule(FillRule::NonZero),
        &mut BuffersBuilder::new(&mut geometry, |vertex: FillVertex| Vertex {
            position: vertex.position().to_array(),
        }),
    );

    match result {
        Ok(()) if !geometry.vertices.is_empty() => Some(Path::new(
            geometry.vertices,
            geometry.indices,
            Arc::new(path.clone()),
        )),
        _ => None,
    }
}

/// Tessellate a path outline (stroke) into an indexed triangle mesh.
///
/// Uses lyon's stroke tessellator with the given `stroke_width` and a
/// tolerance of 0.01. Results are cached inside the [`Path`]'s stroke
/// cache so that repeated draws at the same width do not re-tessellate.
///
/// Returns `None` if tessellation fails or produces no geometry.
pub fn tessellate_stroke(path: &Path, stroke_width: f32) -> Option<Path> {
    use lyon::tessellation::{
        BuffersBuilder, StrokeOptions, StrokeTessellator, StrokeVertex, VertexBuffers,
    };

    // Check the cache first.
    if let Some((verts, idxs)) = path.cached_stroke(stroke_width) {
        return Some(Path::from_arcs(verts, idxs, Arc::clone(&path.lyon_path)));
    }

    // Cache miss — tessellate the stroke.
    let mut geometry: VertexBuffers<Vertex, u32> = VertexBuffers::new();
    let mut tessellator = StrokeTessellator::new();

    let result = tessellator.tessellate_path(
        &*path.lyon_path,
        &StrokeOptions::tolerance(0.01).with_line_width(stroke_width),
        &mut BuffersBuilder::new(&mut geometry, |vertex: StrokeVertex| Vertex {
            position: vertex.position().to_array(),
        }),
    );

    match result {
        Ok(()) if !geometry.vertices.is_empty() => {
            let verts = Arc::new(geometry.vertices);
            let idxs = Arc::new(geometry.indices);

            // Populate the cache for next time.
            path.set_stroke_cache(stroke_width, Arc::clone(&verts), Arc::clone(&idxs));

            Some(Path::from_arcs(verts, idxs, Arc::clone(&path.lyon_path)))
        }
        _ => None,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
pub(crate) mod tests {
    use super::*;
    use livesplit_core::rendering::SharedOwnership;

    /// Helper to compare `[f32; 2]` with tolerance.
    pub(crate) fn assert_bounds_eq(actual: [f32; 2], expected: [f32; 2]) {
        assert!(
            (actual[0] - expected[0]).abs() < f32::EPSILON
                && (actual[1] - expected[1]).abs() < f32::EPSILON,
            "expected {expected:?}, got {actual:?}",
        );
    }

    #[test]
    fn vertex_bounds_y_known_values() {
        let vertices = [
            Vertex {
                position: [0.0, 1.0],
            },
            Vertex {
                position: [1.0, 3.0],
            },
            Vertex {
                position: [2.0, 2.0],
            },
        ];
        assert_bounds_eq(vertex_bounds(&vertices, 1), [1.0, 3.0]);
    }

    #[test]
    fn vertex_bounds_x_known_values() {
        let vertices = [
            Vertex {
                position: [5.0, 0.0],
            },
            Vertex {
                position: [2.0, 0.0],
            },
            Vertex {
                position: [8.0, 0.0],
            },
        ];
        assert_bounds_eq(vertex_bounds(&vertices, 0), [2.0, 8.0]);
    }

    #[test]
    fn vertex_bounds_empty_returns_zeros() {
        let empty: &[Vertex] = &[];
        assert_bounds_eq(vertex_bounds(empty, 0), [0.0, 0.0]);
        assert_bounds_eq(vertex_bounds(empty, 1), [0.0, 0.0]);
    }

    #[test]
    fn vertex_bounds_single_vertex() {
        let vertices = [Vertex {
            position: [3.0, 7.0],
        }];
        let [min, max] = vertex_bounds(&vertices, 0);
        assert!((min - max).abs() < f32::EPSILON);
        assert!((min - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn vertex_bounds_negative_coordinates() {
        let vertices = [
            Vertex {
                position: [-5.0, -10.0],
            },
            Vertex {
                position: [5.0, 10.0],
            },
        ];
        assert_bounds_eq(vertex_bounds(&vertices, 0), [-5.0, 5.0]);
        assert_bounds_eq(vertex_bounds(&vertices, 1), [-10.0, 10.0]);
    }

    #[test]
    fn vertex_bounds_identical_coordinates() {
        let vertices = [
            Vertex {
                position: [5.0, 5.0],
            },
            Vertex {
                position: [5.0, 5.0],
            },
        ];
        assert_bounds_eq(vertex_bounds(&vertices, 0), [5.0, 5.0]);
        assert_bounds_eq(vertex_bounds(&vertices, 1), [5.0, 5.0]);
    }

    #[test]
    fn path_share_clones_arcs() {
        use lyon::math::point;
        let mut builder = LyonPath::builder();
        builder.begin(point(0.0, 0.0));
        builder.line_to(point(1.0, 0.0));
        builder.line_to(point(0.5, 1.0));
        builder.close();
        let lyon_path = builder.build();

        let path = Path::new(
            vec![
                Vertex {
                    position: [0.0, 0.0],
                },
                Vertex {
                    position: [1.0, 0.0],
                },
                Vertex {
                    position: [0.5, 1.0],
                },
            ],
            vec![0, 1, 2],
            Arc::new(lyon_path),
        );

        let shared = SharedOwnership::share(&path);
        assert!(Arc::ptr_eq(&path.vertices, &shared.vertices));
        assert!(Arc::ptr_eq(&path.indices, &shared.indices));
    }

    // --- tessellate_path tests ---

    #[test]
    fn tessellate_path_unit_rectangle() {
        use lyon::math::point;
        let mut builder = LyonPath::builder();
        builder.begin(point(0.0, 0.0));
        builder.line_to(point(1.0, 0.0));
        builder.line_to(point(1.0, 1.0));
        builder.line_to(point(0.0, 1.0));
        builder.close();
        let lyon_path = builder.build();

        let result = tessellate_path(&lyon_path);
        assert!(result.is_some(), "rectangle should tessellate successfully");

        let path = result.unwrap();
        assert!(!path.vertices.is_empty(), "rectangle should have vertices");
        assert!(!path.indices.is_empty(), "rectangle should have indices");
        assert_eq!(
            path.indices.len() % 3,
            0,
            "index count must be a multiple of 3 (triangles)"
        );
    }

    #[test]
    fn tessellate_path_empty_returns_none() {
        let builder = LyonPath::builder();
        let lyon_path = builder.build();
        assert!(
            tessellate_path(&lyon_path).is_none(),
            "empty path should return None"
        );
    }

    #[test]
    fn tessellate_path_degenerate_single_point_returns_none() {
        use lyon::math::point;
        let mut builder = LyonPath::builder();
        builder.begin(point(1.0, 1.0));
        builder.close();
        let lyon_path = builder.build();
        assert!(
            tessellate_path(&lyon_path).is_none(),
            "single-point path should return None"
        );
    }

    // --- CommonPathBuilder tests ---

    #[test]
    fn path_builder_builds_triangle() {
        use livesplit_core::rendering::PathBuilder;
        let mut pb = CommonPathBuilder::new();
        pb.move_to(0.0, 0.0);
        pb.line_to(1.0, 0.0);
        pb.line_to(0.5, 1.0);
        pb.close();
        let result = pb.finish();

        assert!(result.is_some(), "triangle should tessellate successfully");
        let path = result.unwrap();
        assert_eq!(path.indices.len(), 3, "triangle should have 3 indices");
    }

    #[test]
    fn path_builder_quadratic_curve() {
        use livesplit_core::rendering::PathBuilder;
        let mut pb = CommonPathBuilder::new();
        pb.move_to(0.0, 0.0);
        pb.quad_to(0.5, 1.0, 1.0, 0.0);
        pb.close();
        assert!(
            pb.finish().is_some(),
            "quadratic curve path should tessellate"
        );
    }

    #[test]
    fn path_builder_cubic_curve() {
        use livesplit_core::rendering::PathBuilder;
        let mut pb = CommonPathBuilder::new();
        pb.move_to(0.0, 0.0);
        pb.curve_to(0.25, 1.0, 0.75, 1.0, 1.0, 0.0);
        pb.close();
        assert!(pb.finish().is_some(), "cubic curve path should tessellate");
    }

    #[test]
    fn path_builder_finish_without_close_does_not_panic() {
        use livesplit_core::rendering::PathBuilder;
        let mut pb = CommonPathBuilder::new();
        pb.move_to(0.0, 0.0);
        pb.line_to(1.0, 0.0);
        pb.line_to(1.0, 1.0);
        // No close() — finish() should end the sub-path automatically.
        let _result = pb.finish();
    }

    #[test]
    fn path_builder_multiple_open_subpaths_does_not_panic() {
        use livesplit_core::rendering::PathBuilder;
        let mut pb = CommonPathBuilder::new();
        pb.move_to(0.0, 0.0);
        pb.line_to(1.0, 0.0);
        pb.move_to(2.0, 0.0);
        pb.line_to(3.0, 0.0);
        let _result = pb.finish();
    }

    // --- tessellate_stroke tests ---

    #[test]
    fn tessellate_stroke_rectangle() {
        use livesplit_core::rendering::PathBuilder;
        let mut pb = CommonPathBuilder::new();
        pb.move_to(0.0, 0.0);
        pb.line_to(1.0, 0.0);
        pb.line_to(1.0, 1.0);
        pb.line_to(0.0, 1.0);
        pb.close();
        let path = pb.finish().unwrap();

        let stroked = tessellate_stroke(&path, 0.1);
        assert!(
            stroked.is_some(),
            "rectangle stroke should produce geometry"
        );
        let stroked = stroked.unwrap();
        assert!(!stroked.vertices.is_empty());
        assert!(!stroked.indices.is_empty());
    }

    #[test]
    fn tessellate_stroke_open_line() {
        use lyon::math::point;
        let mut builder = LyonPath::builder();
        builder.begin(point(0.0, 0.0));
        builder.line_to(point(1.0, 0.0));
        builder.end(false);
        let lyon_path = builder.build();

        // Fill of an open line may be None, so create a Path manually.
        let fill = tessellate_path(&lyon_path);
        let path = fill.unwrap_or_else(|| Path::new(vec![], vec![], Arc::new(lyon_path)));

        let stroked = tessellate_stroke(&path, 0.1);
        assert!(
            stroked.is_some(),
            "open line stroke should produce geometry"
        );
    }

    #[test]
    fn tessellate_stroke_cache_hit() {
        use livesplit_core::rendering::PathBuilder;
        let mut pb = CommonPathBuilder::new();
        pb.move_to(0.0, 0.0);
        pb.line_to(1.0, 0.0);
        pb.line_to(0.5, 1.0);
        pb.close();
        let path = pb.finish().unwrap();

        // First call populates the cache.
        let first = tessellate_stroke(&path, 0.1);
        assert!(first.is_some());

        // Cache should now be populated.
        let cached = path.cached_stroke(0.1);
        assert!(
            cached.is_some(),
            "cache should be populated after first stroke"
        );

        // Second call should return geometry from cache with same Arc pointers.
        let second = tessellate_stroke(&path, 0.1).unwrap();
        let (cached_verts, cached_idxs) = cached.unwrap();
        assert!(Arc::ptr_eq(&second.vertices, &cached_verts));
        assert!(Arc::ptr_eq(&second.indices, &cached_idxs));
    }

    #[test]
    fn tessellate_stroke_cache_miss_on_different_width() {
        use livesplit_core::rendering::PathBuilder;
        let mut pb = CommonPathBuilder::new();
        pb.move_to(0.0, 0.0);
        pb.line_to(1.0, 0.0);
        pb.line_to(0.5, 1.0);
        pb.close();
        let path = pb.finish().unwrap();

        let _ = tessellate_stroke(&path, 0.1);

        // Different width should miss the cache.
        let cached = path.cached_stroke(0.2);
        assert!(cached.is_none(), "cache should miss for different width");
    }

    #[test]
    fn tessellate_stroke_different_width_evicts_cache() {
        use livesplit_core::rendering::PathBuilder;
        let mut pb = CommonPathBuilder::new();
        pb.move_to(0.0, 0.0);
        pb.line_to(1.0, 0.0);
        pb.line_to(0.5, 1.0);
        pb.close();
        let path = pb.finish().unwrap();

        // Populate cache with width 0.1
        let first = tessellate_stroke(&path, 0.1).unwrap();

        // Tessellate with different width — should evict old cache entry.
        let second = tessellate_stroke(&path, 0.2).unwrap();

        // The new cache should be for width 0.2, not 0.1.
        assert!(path.cached_stroke(0.2).is_some());
        assert!(path.cached_stroke(0.1).is_none());

        // Arcs should differ since they're from different tessellations.
        assert!(!Arc::ptr_eq(&first.vertices, &second.vertices));
    }

    #[test]
    fn stroke_cache_epsilon_boundary() {
        use livesplit_core::rendering::PathBuilder;
        let mut pb = CommonPathBuilder::new();
        pb.move_to(0.0, 0.0);
        pb.line_to(1.0, 0.0);
        pb.line_to(0.5, 1.0);
        pb.close();
        let path = pb.finish().unwrap();

        let _ = tessellate_stroke(&path, 1.0);

        // Width differing by exactly EPSILON should miss.
        assert!(path.cached_stroke(1.0 + f32::EPSILON).is_none());
        // Width differing by less than EPSILON should hit.
        assert!(path.cached_stroke(1.0 + f32::EPSILON / 2.0).is_some());
    }
}
