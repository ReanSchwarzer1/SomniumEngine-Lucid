//! The editor's launch splash (Phase ALYX-J; Lucid's mark since 6 Nov 2026).
//!
//! Lucid's mark builds itself: the lens's four bands slide in one after another, then the ray extends out of the
//! lens; LUCID tracks in beneath it, and what the editor is doing; then, once loading is done, the cover fades and the
//! editor shows through. The mark and the word are filled shapes read from the full-colour SVGs of the logo set
//! (`assets/brand/lucid-mark.svg`, `lucid-wordmark.svg`; the set and its build script are in
//! `F:/Somnium Games/rimeborn-fp/design/lucid_logo/final`). The accent shapes, the ray and the crescent C, are told by
//! their colour in those files, so the splash can never disagree with the logo about which shapes carry it.
//!
//! Timing curves are After Effects-style temporal eases, translated from EffectCraft's `keyframe` crate
//! (`ease_progress`; MIT or Apache-2.0, `Downloads/Craft_Source/effectcraft-main/crates/keyframe/src/lib.rs`).
use std::time::Instant;

use glam::Vec2;

use crate::{
    draw::DrawingContext,
    message::UiMessage,
    node::{Control, LayoutCtx, UiNode},
    shaped::ShapedInstance,
    types::Rect,
    widget::{Widget, WidgetBuilder},
};

/// The full-colour logo files the splash draws.
const MARK_SVG: &str = include_str!("../../assets/brand/lucid-mark.svg");
const WORD_SVG: &str = include_str!("../../assets/brand/lucid-wordmark.svg");
/// The mark's view box.
const MARK_BOX: f32 = 64.0;
/// On screen.
const MARK_PX: f32 = 104.0;
/// LUCID's cap height on screen; the wordmark file draws it 100 units tall.
const CAP_PX: f32 = 26.0;
const WORD_CAP: f32 = 100.0;
/// The intro's length, and how long the cover takes to leave.
const INTRO_S: f32 = 1.1;
const EXIT_S: f32 = 0.6;

fn bez(a: f64, b: f64, c: f64, d: f64, u: f64) -> f64 {
    let v = 1.0 - u;
    v * v * v * a + 3.0 * v * v * u * b + 3.0 * v * u * u * c + u * u * u * d
}

fn solve_u(x0: f64, x1: f64, x2: f64, x3: f64, x: f64) -> f64 {
    let (mut lo, mut hi) = (0.0, 1.0);
    for _ in 0..48 {
        let mid = (lo + hi) * 0.5;
        if bez(x0, x1, x2, x3, mid) < x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) * 0.5
}

/// EffectCraft's `ease_progress`: progress through a temporal Bezier segment with normalised out/in speeds and
/// influences. Speeds of 0 with influences of a third are After Effects' Easy Ease.
#[must_use]
pub fn ease_progress(out_speed: f64, out_influence: f64, in_speed: f64, in_influence: f64, x: f64) -> f64 {
    let (i0, i1) = (out_influence.clamp(0.0, 1.0), in_influence.clamp(0.0, 1.0));
    let u = solve_u(0.0, i0, 1.0 - i1, 1.0, x.clamp(0.0, 1.0));
    bez(0.0, out_speed * i0, 1.0 - in_speed * i1, 1.0, u)
}

/// A long, soft landing (out fast, in very slowly), for things arriving.
fn arrive(x: f32) -> f32 {
    ease_progress(1.6, 0.25, 0.0, 0.85, f64::from(x)) as f32
}

/// Easy Ease, for things leaving.
fn easy(x: f32) -> f32 {
    ease_progress(0.0, 0.33, 0.0, 0.33, f64::from(x)) as f32
}

/// One filled shape of the logo, triangulated in its file's units.
#[derive(Clone, Debug)]
struct Shape {
    triangles: Vec<Vec2>,
    min: Vec2,
    max: Vec2,
    /// Drawn in Lucid's cyan in the file: the ray, the crescent C.
    accent: bool,
}

/// The filled paths of a full-colour logo file, in document order: each path's closed contours (every transform
/// applied) and whether its fill is the accent.
fn paths(source: &str) -> Vec<(Vec<Vec<Vec2>>, bool)> {
    let Ok(tree) = resvg::usvg::Tree::from_str(source, &resvg::usvg::Options::default()) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    gather(tree.root(), &mut out);
    out
}

fn gather(group: &resvg::usvg::Group, out: &mut Vec<(Vec<Vec<Vec2>>, bool)>) {
    let cyan = crate::brand::LUCID_CYAN;
    for node in group.children() {
        match node {
            resvg::usvg::Node::Group(inner) => gather(inner, out),
            resvg::usvg::Node::Path(path) => {
                let Some(fill) = path.fill() else { continue };
                let accent = matches!(fill.paint(), resvg::usvg::Paint::Color(c)
                    if [c.red, c.green, c.blue] == [cyan[0], cyan[1], cyan[2]]);
                let contours = contours(path);
                if !contours.is_empty() {
                    out.push((contours, accent));
                }
            }
            _ => {}
        }
    }
}

/// A path's closed contours, flattened. Curves split by their length, so the many short arc steps a logo file holds
/// stay one segment each and a long bowl gets enough.
fn contours(path: &resvg::usvg::Path) -> Vec<Vec<Vec2>> {
    use resvg::tiny_skia::PathSegment;
    let t = path.abs_transform();
    let map = |p: resvg::tiny_skia::Point| Vec2::new(t.sx * p.x + t.kx * p.y + t.tx, t.ky * p.x + t.sy * p.y + t.ty);
    let steps = |length: f32| (length / 1.5).ceil().clamp(1.0, 16.0) as usize;
    let mut out = Vec::new();
    let mut ring: Vec<Vec2> = Vec::new();
    for segment in path.data().segments() {
        match segment {
            PathSegment::MoveTo(p) => {
                out.push(std::mem::take(&mut ring));
                ring.push(map(p));
            }
            PathSegment::LineTo(p) => ring.push(map(p)),
            PathSegment::QuadTo(c, p) => {
                let (a, c, p) = (*ring.last().unwrap_or(&map(c)), map(c), map(p));
                let n = steps(a.distance(c) + c.distance(p));
                for i in 1..=n {
                    let u = i as f32 / n as f32;
                    ring.push(a * (1.0 - u) * (1.0 - u) + c * 2.0 * u * (1.0 - u) + p * u * u);
                }
            }
            PathSegment::CubicTo(c1, c2, p) => {
                let (a, c1, c2, p) = (*ring.last().unwrap_or(&map(c1)), map(c1), map(c2), map(p));
                let n = steps(a.distance(c1) + c1.distance(c2) + c2.distance(p));
                for i in 1..=n {
                    let u = i as f32 / n as f32;
                    let v = 1.0 - u;
                    ring.push(a * v * v * v + c1 * 3.0 * v * v * u + c2 * 3.0 * v * u * u + p * u * u * u);
                }
            }
            PathSegment::Close => out.push(std::mem::take(&mut ring)),
        }
    }
    out.push(ring);
    for ring in &mut out {
        ring.dedup_by(|b, a| a.distance(*b) < 1e-4);
        if ring.len() > 1 && ring[0].distance(ring[ring.len() - 1]) < 1e-4 {
            ring.pop();
        }
    }
    out.retain(|ring| ring.len() > 2);
    out
}

fn shapes(source: &str) -> Vec<Shape> {
    paths(source)
        .into_iter()
        .map(|(contours, accent)| {
            let all = contours.iter().flatten();
            let min = all.clone().fold(Vec2::splat(f32::MAX), |m, p| m.min(*p));
            let max = all.fold(Vec2::splat(f32::MIN), |m, p| m.max(*p));
            Shape { triangles: fill_even_odd(&contours), min, max, accent }
        })
        .collect()
}

fn cross(a: Vec2, b: Vec2) -> f32 {
    a.x * b.y - a.y * b.x
}

fn signed_area(ring: &[Vec2]) -> f32 {
    (0..ring.len()).map(|i| cross(ring[i], ring[(i + 1) % ring.len()])).sum::<f32>() * 0.5
}

/// Even-odd point in polygon.
fn inside(p: Vec2, ring: &[Vec2]) -> bool {
    let mut odd = false;
    for i in 0..ring.len() {
        let (a, b) = (ring[i], ring[(i + 1) % ring.len()]);
        if (a.y > p.y) != (b.y > p.y) && p.x < a.x + (p.y - a.y) / (b.y - a.y) * (b.x - a.x) {
            odd = !odd;
        }
    }
    odd
}

/// Whether two segments cross at a point inside both (touching ends do not count).
fn crosses(a0: Vec2, a1: Vec2, b0: Vec2, b1: Vec2) -> bool {
    let d1 = cross(a1 - a0, b0 - a0);
    let d2 = cross(a1 - a0, b1 - a0);
    let d3 = cross(b1 - b0, a0 - b0);
    let d4 = cross(b1 - b0, a1 - b0);
    d1 * d2 < 0.0 && d3 * d4 < 0.0
}

/// Triangles filling `contours` even-odd. `path::fill_contour` clips one contour at a time, so the D's counter would
/// come out solid: here each hole is bridged into the outline around it (to the nearest outline vertex the bridge can
/// see), and the joined ring is ear clipped by a clipper that tolerates the bridge's repeated points.
fn fill_even_odd(contours: &[Vec<Vec2>]) -> Vec<Vec2> {
    let depth: Vec<usize> = contours
        .iter()
        .enumerate()
        .map(|(i, ring)| contours.iter().enumerate().filter(|(j, other)| *j != i && inside(ring[0], other)).count())
        .collect();
    let mut out = Vec::new();
    for (i, outline) in contours.iter().enumerate().filter(|(i, _)| depth[*i] % 2 == 0) {
        let mut ring = outline.clone();
        if signed_area(&ring) < 0.0 {
            ring.reverse();
        }
        let mut holes: Vec<Vec<Vec2>> = contours
            .iter()
            .enumerate()
            .filter(|(j, hole)| depth[*j] == depth[i] + 1 && inside(hole[0], outline))
            .map(|(_, hole)| {
                let mut hole = hole.clone();
                if signed_area(&hole) > 0.0 {
                    hole.reverse();
                }
                hole
            })
            .collect();
        let right = |ring: &Vec<Vec2>| ring.iter().fold(f32::MIN, |m, p| m.max(p.x));
        holes.sort_by(|a, b| right(b).total_cmp(&right(a)));
        for (k, hole) in holes.iter().enumerate() {
            ring = bridge(&ring, hole, &holes[k + 1..]);
        }
        out.extend(ear_clip(&ring));
    }
    out
}

/// Join `hole` into `ring` along the shortest bridge that crosses neither, nor the holes still to come.
fn bridge(ring: &[Vec2], hole: &[Vec2], later: &[Vec<Vec2>]) -> Vec<Vec2> {
    let m = (0..hole.len()).max_by(|a, b| hole[*a].x.total_cmp(&hole[*b].x)).unwrap_or(0);
    let from = hole[m];
    let edges = |r: &[Vec2]| (0..r.len()).map(|i| (r[i], r[(i + 1) % r.len()])).collect::<Vec<_>>();
    let mut blockers = edges(ring);
    blockers.extend(edges(hole));
    for other in later {
        blockers.extend(edges(other));
    }
    let clear = |to: Vec2| blockers.iter().all(|(a, b)| !crosses(from, to, *a, *b));
    let Some(p) = (0..ring.len())
        .filter(|&p| clear(ring[p]))
        .min_by(|a, b| from.distance_squared(ring[*a]).total_cmp(&from.distance_squared(ring[*b])))
    else {
        return ring.to_vec();
    };
    let mut joined = ring[..=p].to_vec();
    joined.extend_from_slice(&hole[m..]);
    joined.extend_from_slice(&hole[..=m]);
    joined.extend_from_slice(&ring[p..]);
    joined
}

/// Ear clipping for a counter-clockwise ring that may touch itself along bridges: a point that repeats a corner of
/// the candidate ear never blocks it, and a vertex that adds no area is dropped.
fn ear_clip(ring: &[Vec2]) -> Vec<Vec2> {
    let mut ring = ring.to_vec();
    let mut out = Vec::new();
    let mut guard = ring.len() * ring.len() + 8;
    while ring.len() > 3 && guard > 0 {
        guard -= 1;
        let n = ring.len();
        let mut clipped = false;
        for i in 0..n {
            let (a, b, c) = (ring[(i + n - 1) % n], ring[i], ring[(i + 1) % n]);
            let turn = cross(b - a, c - b);
            if turn.abs() < 1e-7 {
                ring.remove(i);
                clipped = true;
                break;
            }
            if turn < 0.0 {
                continue;
            }
            let blocked = ring.iter().any(|&p| {
                p != a
                    && p != b
                    && p != c
                    && cross(b - a, p - a) > 0.0
                    && cross(c - b, p - b) > 0.0
                    && cross(a - c, p - c) > 0.0
            });
            if blocked {
                continue;
            }
            out.extend([a, b, c]);
            ring.remove(i);
            clipped = true;
            break;
        }
        if !clipped {
            break;
        }
    }
    if ring.len() == 3 && signed_area(&ring) > 0.0 {
        out.extend(ring);
    }
    out
}

#[derive(Clone, Debug)]
pub enum SplashMessage {
    /// What the editor is doing now, in a few words.
    Status(String),
    /// Loading is done: leave once the intro has played.
    Finish,
    /// Draw everything at rest: no slide, no extend, only the fade.
    ReducedMotion(bool),
}

pub struct Splash {
    born: Instant,
    finish: Option<Instant>,
    status: String,
    mark: Vec<Shape>,
    word: Vec<Shape>,
    font_id: u8,
    reduced: bool,
}

impl Splash {
    fn intro(&self, now: Instant) -> f32 {
        if self.reduced {
            1.0
        } else {
            (now - self.born).as_secs_f32() / INTRO_S
        }
    }

    /// 0 while showing, rising to 1 as the cover leaves.
    #[must_use]
    pub fn exit(&self, now: Instant) -> f32 {
        exit_progress(self.born, self.finish, now)
    }

    /// Whether it has finished leaving, so its owner can hide it.
    #[must_use]
    pub fn gone(&self, now: Instant) -> bool {
        self.exit(now) >= 1.0
    }
}

/// How far a splash born at `born` and told to finish at `finish` has left by `now`: 0 while showing, 1 once gone.
/// The exit starts when asked, but never before the intro has played. Its owner uses the same clock to drop it.
#[must_use]
pub fn exit_progress(born: Instant, finish: Option<Instant>, now: Instant) -> f32 {
    finish
        .map(|finish| finish.max(born + std::time::Duration::from_secs_f32(INTRO_S)))
        .map_or(0.0, |start| (now.saturating_duration_since(start).as_secs_f32() / EXIT_S).clamp(0.0, 1.0))
}

/// How far into place band `index` of the lens has come at `intro` (0 to 1), and the ray's length then.
fn band_arrival(intro: f32, index: usize) -> f32 {
    arrive(((intro - 0.08 * index as f32) / 0.5).clamp(0.0, 1.0))
}

fn ray_length(intro: f32) -> f32 {
    arrive(((intro - 0.45) / 0.45).clamp(0.0, 1.0))
}

impl Control for Splash {
    fn measure_override(&self, _widget: &Widget, _ctx: &mut LayoutCtx, available: Vec2) -> Vec2 {
        available
    }

    fn draw(&self, widget: &Widget, ctx: &mut DrawingContext) {
        let now = Instant::now();
        let t = crate::theme::active();
        let b = widget.screen_bounds();
        let exit = easy(self.exit(now));
        let fade = 1.0 - exit;
        let alpha = |colour: [u8; 4], a: f32| [colour[0], colour[1], colour[2], (f32::from(colour[3]) * a) as u8];
        ctx.push_rect_filled(b, alpha(t.semantic.surface.canvas.bytes(), fade));

        let centre = Vec2::new(b.x + b.w * 0.5, b.y + b.h * 0.5 - 36.0);
        let scale = MARK_PX / MARK_BOX * (1.0 + 0.12 * exit);
        let intro = self.intro(now);
        let neutral = t.semantic.text.emphasis.bytes();
        let accent = crate::brand::LUCID_CYAN;
        let place = |p: Vec2| centre + (p - Vec2::splat(MARK_BOX * 0.5)) * scale;

        // The lens: its bands slide in one after another, from alternate sides. Then the ray extends out of it.
        let mut band = 0;
        for shape in &self.mark {
            if shape.accent {
                let length = ray_length(intro);
                if length <= 0.001 {
                    continue;
                }
                let root = shape.min.x;
                let points: Vec<Vec2> = shape
                    .triangles
                    .iter()
                    .map(|p| place(Vec2::new(root + (p.x - root) * length, p.y)))
                    .collect();
                ctx.push_shaped(ShapedInstance::identity(alpha(accent, fade)), &points);
            } else {
                let arrived = band_arrival(intro, band);
                let side = if band % 2 == 0 { -1.0 } else { 1.0 };
                band += 1;
                if arrived <= 0.001 {
                    continue;
                }
                let shift = Vec2::new(side * 18.0 * (1.0 - arrived), 0.0);
                let points: Vec<Vec2> = shape.triangles.iter().map(|p| place(*p + shift)).collect();
                ctx.push_shaped(ShapedInstance::identity(alpha(neutral, arrived * fade)), &points);
            }
        }

        // LUCID tracks in from wide to its resting spacing; its C is the accent.
        let word = ((intro - 0.35) / 0.65).clamp(0.0, 1.0);
        if word > 0.0 && !self.word.is_empty() {
            let settle = arrive(word);
            let lo = self.word.iter().fold(Vec2::splat(f32::MAX), |m, s| m.min(s.min));
            let hi = self.word.iter().fold(Vec2::splat(f32::MIN), |m, s| m.max(s.max));
            let s = CAP_PX / WORD_CAP;
            let left = centre.x - (hi.x - lo.x) * s * 0.5;
            let top = centre.y + MARK_PX * 0.5 + 20.0;
            let middle = (self.word.len() as f32 - 1.0) * 0.5;
            for (index, letter) in self.word.iter().enumerate() {
                let spread = (index as f32 - middle) * 9.0 * (1.0 - settle);
                let points: Vec<Vec2> = letter
                    .triangles
                    .iter()
                    .map(|p| Vec2::new(left + (p.x - lo.x) * s + spread, top + (p.y - lo.y) * s))
                    .collect();
                let colour = if letter.accent { accent } else { neutral };
                ctx.push_shaped(ShapedInstance::identity(alpha(colour, settle * fade)), &points);
            }
        }

        // What is happening, and a hairline that says the editor is alive while it happens.
        let line_y = centre.y + MARK_PX * 0.5 + 84.0;
        let status = if self.finish.is_some() { "Ready" } else { self.status.as_str() };
        let status_w = ctx.font_atlas.measure_text(status, 12.0, self.font_id).x;
        ctx.push_text(
            status,
            Vec2::new(centre.x - status_w * 0.5, line_y),
            self.font_id,
            12.0,
            alpha(t.semantic.text.muted.bytes(), fade),
        );
        let progress = t.semantic.accent.default.bytes();
        let track = Rect::new(centre.x - 110.0, line_y + 24.0, 220.0, 2.0);
        ctx.push_rect_filled(track, alpha(t.semantic.border.subtle.bytes(), fade));
        if self.finish.is_some() {
            ctx.push_rect_filled(track, alpha(progress, fade));
        } else {
            let cycle = ((now - self.born).as_secs_f32() / 1.4).fract();
            let head = easy(cycle);
            let x0 = track.x + track.w * (head - 0.28).max(0.0);
            let x1 = track.x + track.w * head.min(1.0);
            ctx.push_rect_filled(Rect::new(x0, track.y, (x1 - x0).max(0.0), track.h), alpha(progress, fade));
        }
    }

    fn handle_routed_message(&mut self, _widget: &mut Widget, msg: &mut UiMessage, _emit: &mut Vec<UiMessage>) {
        match msg.data::<SplashMessage>() {
            Some(SplashMessage::Status(text)) => self.status.clone_from(text),
            Some(SplashMessage::Finish) => {
                self.finish.get_or_insert_with(Instant::now);
            }
            Some(SplashMessage::ReducedMotion(on)) => self.reduced = *on,
            None => return,
        }
        msg.handled = true;
    }
}

pub struct SplashBuilder {
    widget: WidgetBuilder,
    status: String,
    font_id: u8,
}

impl SplashBuilder {
    #[must_use]
    pub fn new(widget: WidgetBuilder) -> Self {
        Self { widget, status: "Starting".into(), font_id: 0 }
    }

    #[must_use]
    pub fn with_status(mut self, status: impl Into<String>) -> Self {
        self.status = status.into();
        self
    }

    #[must_use]
    pub fn with_font_id(mut self, font_id: u8) -> Self {
        self.font_id = font_id;
        self
    }

    #[must_use]
    pub fn build(self) -> UiNode {
        UiNode::new(
            self.widget.build(),
            Box::new(Splash {
                born: Instant::now(),
                finish: None,
                status: self.status,
                mark: shapes(MARK_SVG),
                word: shapes(WORD_SVG),
                font_id: self.font_id,
                reduced: false,
            }),
        )
    }
}

/// Lucid's horizontal lock-up, the mark and LUCID, drawn from the same files as the splash (the About page shows it).
/// It keeps the logo's layout: LUCID's cap line and baseline on the band edges (y 16 and 48 of the mark's 64), and the
/// word starting 15 units after the ray's point, 199 units in all.
pub struct Lockup {
    mark: Vec<Shape>,
    word: Vec<Shape>,
    height: f32,
}

const LOCKUP_W: f32 = 199.0;
const LOCKUP_WORD: Vec2 = Vec2::new(76.0, 16.0);
const LOCKUP_CAP: f32 = 32.0;

impl Lockup {
    /// Where a point of either file lands, in the mark's 64 units: the word's L (its first shape) puts its top-left,
    /// the cap line, at `LOCKUP_WORD`.
    fn word_point(&self, p: Vec2) -> Vec2 {
        let origin = self.word.first().map_or(Vec2::ZERO, |l| l.min);
        LOCKUP_WORD + (p - origin) * (LOCKUP_CAP / WORD_CAP)
    }
}

impl Control for Lockup {
    fn measure_override(&self, _widget: &Widget, _ctx: &mut LayoutCtx, _available: Vec2) -> Vec2 {
        Vec2::new(LOCKUP_W / MARK_BOX * self.height, self.height)
    }

    fn draw(&self, widget: &Widget, ctx: &mut DrawingContext) {
        let t = crate::theme::active();
        let b = widget.screen_bounds();
        let (corner, s) = (Vec2::new(b.x, b.y), self.height / MARK_BOX);
        let colour = |shape: &Shape| if shape.accent { crate::brand::LUCID_CYAN } else { t.semantic.text.emphasis.bytes() };
        for shape in &self.mark {
            let points: Vec<Vec2> = shape.triangles.iter().map(|p| corner + *p * s).collect();
            ctx.push_shaped(ShapedInstance::identity(colour(shape)), &points);
        }
        for shape in &self.word {
            let points: Vec<Vec2> = shape.triangles.iter().map(|p| corner + self.word_point(*p) * s).collect();
            ctx.push_shaped(ShapedInstance::identity(colour(shape)), &points);
        }
    }
}

pub struct LockupBuilder {
    widget: WidgetBuilder,
    height: f32,
}

impl LockupBuilder {
    #[must_use]
    pub fn new(widget: WidgetBuilder) -> Self {
        Self { widget, height: 48.0 }
    }

    /// The mark's height on screen; the word follows it.
    #[must_use]
    pub fn with_height(mut self, height: f32) -> Self {
        self.height = height;
        self
    }

    #[must_use]
    pub fn build(self) -> UiNode {
        UiNode::new(
            self.widget.build(),
            Box::new(Lockup { mark: shapes(MARK_SVG), word: shapes(WORD_SVG), height: self.height }),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_lockup_puts_lucid_between_the_band_edges_after_the_ray() {
        let lockup = Lockup { mark: shapes(MARK_SVG), word: shapes(WORD_SVG), height: 64.0 };
        let l = &lockup.word[0];
        let (top, bottom) = (lockup.word_point(l.min).y, lockup.word_point(l.max).y);
        assert!((top - 16.0).abs() < 1e-3 && (bottom - 48.0).abs() < 1e-3, "cap line and baseline on the bands");
        let ray_tip = lockup.mark.iter().filter(|s| s.accent).map(|s| s.max.x).fold(f32::MIN, f32::max);
        assert!((lockup.word_point(l.min).x - ray_tip - 15.0).abs() < 0.1, "15 units after the ray's point");
        let right = lockup.word.iter().map(|s| lockup.word_point(s.max).x).fold(f32::MIN, f32::max);
        assert!(right < LOCKUP_W && right > LOCKUP_W - 7.0, "the word ends inside the 199-unit box");
    }

    fn area(triangles: &[Vec2]) -> f32 {
        triangles.chunks_exact(3).map(|t| cross(t[1] - t[0], t[2] - t[0]).abs() * 0.5).sum()
    }

    /// What even-odd filling covers: outlines add, holes take away.
    fn even_odd_area(contours: &[Vec<Vec2>]) -> f32 {
        contours
            .iter()
            .enumerate()
            .map(|(i, ring)| {
                let depth = contours.iter().enumerate().filter(|(j, o)| *j != i && inside(ring[0], o)).count();
                signed_area(ring).abs() * if depth % 2 == 0 { 1.0 } else { -1.0 }
            })
            .sum()
    }

    #[test]
    fn the_ease_starts_and_lands_where_it_should() {
        for f in [arrive, easy] {
            assert!(f(0.0).abs() < 1e-3);
            assert!((f(1.0) - 1.0).abs() < 1e-3);
            assert!(f(0.25) < f(0.5) && f(0.5) < f(0.75), "monotone");
        }
        assert!(arrive(0.3) > 0.5, "arriving is fast early and slow late");
    }

    #[test]
    fn the_mark_is_four_filled_bands_and_the_ray() {
        let mark = shapes(MARK_SVG);
        assert_eq!(mark.len(), 5, "four bands and the ray");
        let rays: Vec<&Shape> = mark.iter().filter(|s| s.accent).collect();
        assert_eq!(rays.len(), 1, "one accent: the ray");
        assert!(rays[0].max.x > 60.0 && rays[0].min.x < 6.0, "the ray runs from the lens to its point at x 61");
        for shape in &mark {
            assert!(!shape.triangles.is_empty(), "every band fills");
            assert!(shape.min.cmpgt(Vec2::splat(-1.0)).all() && shape.max.cmplt(Vec2::splat(MARK_BOX + 1.0)).all());
        }
    }

    #[test]
    fn the_word_is_lucid_and_its_crescent_c_is_the_accent() {
        let word = shapes(WORD_SVG);
        assert_eq!(word.iter().map(|s| s.accent).collect::<Vec<_>>(), [false, false, true, false, false]);
        let lo = word.iter().fold(f32::MAX, |m, s| m.min(s.min.y));
        let hi = word.iter().fold(f32::MIN, |m, s| m.max(s.max.y));
        assert!((hi - lo - WORD_CAP).abs() < 4.0, "cap height 100 with the moon's and the U's overshoots");
    }

    #[test]
    fn every_shape_fills_exactly_its_even_odd_area() {
        // The D has a counter: one contour at a time would fill it solid, and the triangles would hold too much.
        for source in [MARK_SVG, WORD_SVG] {
            for (contours, _) in paths(source) {
                let expected = even_odd_area(&contours);
                let got = area(&fill_even_odd(&contours));
                assert!((got - expected).abs() <= expected * 0.002 + 1e-3, "filled {got}, expected {expected}");
            }
        }
        let with_counter = paths(WORD_SVG).into_iter().filter(|(c, _)| c.len() == 2).count();
        assert_eq!(with_counter, 1, "the D is the one letter with a counter");
    }

    #[test]
    fn a_square_with_a_square_hole_fills_its_frame() {
        let outer = vec![Vec2::ZERO, Vec2::new(10.0, 0.0), Vec2::new(10.0, 10.0), Vec2::new(0.0, 10.0)];
        let hole = vec![Vec2::new(3.0, 3.0), Vec2::new(7.0, 3.0), Vec2::new(7.0, 7.0), Vec2::new(3.0, 7.0)];
        let triangles = fill_even_odd(&[outer, hole]);
        assert!((area(&triangles) - 84.0).abs() < 1e-3);
        for t in triangles.chunks_exact(3) {
            let centre = (t[0] + t[1] + t[2]) / 3.0;
            assert!(!(centre.x > 3.0 && centre.x < 7.0 && centre.y > 3.0 && centre.y < 7.0), "nothing in the hole");
        }
    }

    #[test]
    fn the_bands_arrive_before_the_ray_and_all_rest_by_the_end() {
        // The ease lands on 0 only to within its bisection; the draw skips anything under 0.001.
        assert!(band_arrival(0.2, 0) > 0.1 && ray_length(0.2) < 1e-3, "the bands first");
        assert!(band_arrival(0.2, 0) > band_arrival(0.2, 3), "one after another");
        for index in 0..4 {
            assert!((band_arrival(1.0, index) - 1.0).abs() < 1e-3);
        }
        assert!((ray_length(1.0) - 1.0).abs() < 1e-3, "the ray reaches its point when the intro ends");
    }

    #[test]
    fn the_exit_waits_for_the_intro_and_then_finishes() {
        let now = Instant::now();
        let mut splash = Splash {
            born: now,
            finish: Some(now),
            status: String::new(),
            mark: Vec::new(),
            word: Vec::new(),
            font_id: 0,
            reduced: false,
        };
        assert_eq!(splash.exit(now), 0.0, "finishing early still lets the intro play");
        let later = now + std::time::Duration::from_secs_f32(INTRO_S + EXIT_S + 0.01);
        assert!(splash.gone(later));
        splash.finish = None;
        assert!(!splash.gone(later), "never leaves on its own");
    }
}
