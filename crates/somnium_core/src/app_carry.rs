//! Move on the Ground (G): the selection follows the mouse over whatever is under it (terrain, a floor, a roof, a
//! rock) until a click or Enter puts it down, and Esc puts it back. Design: `design/editor/MOVE_ON_THE_GROUND_2026-11-27.md`
//! in the Sunscar project.
//!
//! A drag whose target is the surface under the cursor instead of an axis handle: one cursor ray a frame for the lead
//! object and one ray down for each carried object. Putting down is exactly Drop to Ground (Align to Surface with
//! Shift), folded with the move into one undo step.
//!
//! Copy and Move (Shift+D) is the same carry with a copy in the hand: the copy is made when it is picked up, enters
//! the undo history as one step when it is put down, and leaves the world again on Esc.

use super::*;

/// A wheel notch turns the carried things this much when the snap angle is off.
const CARRY_TURN_DEG: f32 = 15.0;
/// How far along the cursor ray a surface is looked for.
const CARRY_REACH: f32 = 5_000.0;
/// A ray down that finds nothing from just above where an object would stand is tried again from this much higher (an
/// object carried up a slope steeper than its own height).
const CARRY_RETRY_LIFT: f32 = 200.0;
/// Touchpad scrolling arrives in pixels: this many make one notch.
const CARRY_PIXELS_PER_NOTCH: f64 = 50.0;
/// The status line while carrying.
const CARRY_HINT: &str = "Click or Enter puts it down, Esc puts it back. Shift tilts to the slope, the wheel turns it, Ctrl snaps to the grid.";
/// The status line while no surface is under the mouse.
const CARRY_LOST: &str = "No ground under the mouse: it waits at the last place.";
/// The status line while a copy is in hand (Copy and Move).
const CARRY_COPY_HINT: &str = "A copy: click puts it down, Esc removes it, Shift+D makes another. Shift tilts, the wheel turns, Ctrl snaps.";

/// A copy in hand (Copy and Move, Shift+D): made when it was picked up, and in the undo history only once it is put
/// down, as one step. Esc takes it out of the world again.
pub(super) struct CarriedCopy {
    /// The paste that made the copies, already executed.
    command: crate::clipboard::PasteEntitiesCmd,
    /// What was selected before the copy, and its primary: Esc selects them again.
    originals: Vec<Entity>,
    primary: Option<Entity>,
}

/// One top-level carried object, as it was picked up.
pub(super) struct CarryObject {
    entity: Entity,
    /// Its local transform at the start.
    start: Transform,
    parent_inverse: glam::Mat4,
    parent_rotation: glam::Quat,
    /// The middle of its box seen from above, at the height of its lowest point: what is put on the ground.
    anchor: glam::Vec3,
    /// Its origin from `anchor`, in the world.
    arm: glam::Vec3,
    /// Its box's height.
    height: f32,
}

/// Every surface a carry (or a model dragged in from the Content Drawer) can be put on, gathered once: the loaded
/// meshes, the terrains and the static box colliders.
pub(super) struct Surfaces {
    /// Left out of every ray: what is carried, parts too.
    members: HashSet<Entity>,
    index: MeshIndex,
    terrains: Vec<u32>,
    /// Static box colliders, less those inside a carried object (its own, left behind where it stood).
    boxes: Vec<(glam::Mat4, glam::Vec3, glam::Vec3)>,
}

/// The carry in hand.
pub(super) struct CarryState {
    /// The objects, the lead (the one that is or holds the primary selection) first.
    objects: Vec<CarryObject>,
    /// The primary's twins in other levels of detail: they take the primary's transform as it is.
    riders: Vec<GizmoFollower>,
    /// The lead is the primary itself, so the riders ride it.
    lead_is_primary: bool,
    /// Every selected entity with its transform at the start, for Esc and for the undo step.
    starts: Vec<(Entity, Transform)>,
    /// The selection at the start, sorted: losing it ends the carry.
    selection: Vec<Entity>,
    /// What it can be carried over.
    surfaces: Surfaces,
    /// The cursor's point on the ground at the start (the lead's base when there was none).
    start_point: glam::Vec3,
    /// The last good point under the cursor.
    point: Option<glam::Vec3>,
    /// Wheel notches turned so far, and the touchpad's part-notch.
    turns: i32,
    pixels: f64,
    /// No surface under the cursor now.
    lost: bool,
    /// What the last placement was made from (cursor, Shift, Ctrl, turns): unchanged, nothing is done.
    seen: Option<((f32, f32), bool, bool, i32)>,
    /// What is in hand is a copy made for this carry (Copy and Move), not the selection itself.
    copy: Option<CarriedCopy>,
    /// Shift does not tilt now: it is held for Shift+D, which puts this copy down and makes the next.
    upright: bool,
}

/// What the toast says when a copy is put down.
pub(super) fn copied_message(count: usize, name: &str) -> String {
    if count == 1 && !name.is_empty() {
        format!("Copied {name}.")
    } else if count == 1 {
        "Copied 1 object.".to_owned()
    } else {
        format!("Copied {count} objects.")
    }
}

/// Where the lead's anchor goes seen from above: moved by as much as the cursor's point on the ground moved, so nothing
/// jumps when the carry starts; on the grid when `grid` is above zero.
pub(super) fn lead_plan(anchor: glam::Vec3, start_point: glam::Vec3, point: glam::Vec3, grid: f32) -> glam::Vec2 {
    glam::Vec2::new(
        SnapSettings::quantise(anchor.x + (point.x - start_point.x), grid),
        SnapSettings::quantise(anchor.z + (point.z - start_point.z), grid),
    )
}

/// Where another carried object's anchor goes seen from above: it keeps its place to the lead's, turned with the lead
/// about the lead's anchor.
pub(super) fn follower_plan(anchor: glam::Vec3, lead_anchor: glam::Vec3, lead_plan: glam::Vec2, turn: glam::Quat) -> glam::Vec2 {
    let arm = turn * glam::Vec3::new(anchor.x - lead_anchor.x, 0.0, anchor.z - lead_anchor.z);
    lead_plan + glam::Vec2::new(arm.x, arm.z)
}

/// How far one wheel notch turns a carried thing, in degrees: the bar's snap angle, or 15 when it is off.
pub(super) fn wheel_turn_degrees(snap_rotate_deg: f32) -> f32 {
    if snap_rotate_deg > 0.0 && snap_rotate_deg.is_finite() { snap_rotate_deg } else { CARRY_TURN_DEG }
}

/// Whole notches in a wheel event, carrying a touchpad's remainder in `pixels`.
pub(super) fn wheel_notches(delta: winit::event::MouseScrollDelta, pixels: &mut f64) -> i32 {
    match delta {
        winit::event::MouseScrollDelta::LineDelta(_, y) => {
            let notches = y.round() as i32;
            if notches == 0 && y != 0.0 { y.signum() as i32 } else { notches }
        }
        winit::event::MouseScrollDelta::PixelDelta(position) => {
            *pixels += position.y;
            let notches = (*pixels / CARRY_PIXELS_PER_NOTCH).trunc();
            *pixels -= notches * CARRY_PIXELS_PER_NOTCH;
            notches as i32
        }
    }
}

/// The distance along a unit ray to where it enters a world box: zero from inside it.
fn box_entry(origin: glam::Vec3, direction: glam::Vec3, min: glam::Vec3, max: glam::Vec3) -> Option<f32> {
    let inside = origin.cmpge(min).all() && origin.cmple(max).all();
    if inside {
        return Some(0.0);
    }
    ray_aabb_distance(origin, direction, min, max)
}

impl<G: GameApp> Engine<G> {
    /// A carry is in hand.
    pub(crate) fn carrying(&self) -> bool {
        self.editing.carry.is_some()
    }

    /// G: pick up the selection, or with one in hand put it back.
    pub(crate) fn toggle_carry(&mut self) {
        if self.carrying() {
            self.cancel_carry();
        } else {
            self.begin_carry();
        }
    }

    /// The paste that duplicates the selection, not executed yet: every part of each object (an imported model's root
    /// alone is an empty shell), under its original's parent, named "<name>_copy" and moved by `offset`.
    pub(crate) fn duplicate_paste(&mut self, offset: glam::Vec3, step: &'static str) -> crate::clipboard::PasteEntitiesCmd {
        let mut clipboard = crate::clipboard::EntityClipboard::duplicate(&self.world, self.selection.as_slice());
        let (renderer, render_ctx) = (&mut self.renderer, &self.render_ctx);
        clipboard.for_each_snapshot_mut(|snapshot, root| {
            if root {
                if let Some(transform) = snapshot.transform.as_mut() {
                    transform.translation += offset;
                }
                let name = snapshot.name.as_ref().map_or_else(|| "Entity".to_owned(), |n| n.as_str().to_owned());
                snapshot.name = Some(Name::new(&format!("{name}_copy")));
            }
            // Terrains are not duplicated — two entities sharing one terrain_id would draw the same
            // terrain twice. Nor is the scene's environment, or what was painted onto the original.
            snapshot.environment = false;
            snapshot.terrain = None;
            snapshot.world_partition = None;
            snapshot.ui_canvas = None;
            snapshot.voxel_terrain = None;
            snapshot.foliage = None;
            if let Some(component) = snapshot.water.as_mut()
                && let (Some(renderer), Some(render_ctx)) = (renderer.as_mut(), render_ctx.as_ref())
            {
                component.water_id = renderer.allocate_water_body_id();
                if let Err(error) = renderer.ensure_water_body(render_ctx, component.descriptor()) {
                    tracing::warn!("Failed to duplicate water data: {error}");
                }
            }
        });
        crate::clipboard::PasteEntitiesCmd::new(clipboard, None).named(step)
    }

    /// Copy and Move (Shift+D): a copy of the selection, in the hand at once. It follows the mouse over the ground
    /// like any carry; putting it down is one undo step, and Esc removes it. With a thing in hand already, that is
    /// put down first, so Shift+D, move, Shift+D, move leaves a row of copies.
    pub(crate) fn copy_and_carry(&mut self) {
        if self.carrying() {
            // Shift is held for the key itself: it does not tilt what is put down by it.
            if let Some(carry) = self.editing.carry.as_mut() {
                carry.upright = true;
                carry.seen = None;
            }
            self.put_down_carry();
        }
        if let Some(message) = self.carry_refusal() {
            self.toast(message);
            return;
        }
        if self.selection.is_empty() {
            self.toast("Select an object first");
            return;
        }
        use crate::editor_commands::EditorCommand as _;
        let originals: Vec<Entity> = self.selection.as_slice().to_vec();
        let primary = self.selection.primary;
        let mut command = self.duplicate_paste(glam::Vec3::ZERO, "Copy and Move");
        command.execute(&mut self.world, &mut self.selection.primary);
        let roots = command.roots().to_vec();
        if roots.is_empty() {
            self.selection.reconcile();
            self.toast("Nothing here can be copied");
            return;
        }
        self.selection.set_many(roots);
        // The copy of what was the primary is the primary, so its twins in other levels of detail ride it as before.
        if let Some(lead) = primary.and_then(|entity| command.root_of(entity)) {
            self.selection.primary = Some(lead);
        }
        self.after_selection_change();
        if !self.begin_carry() {
            // It cannot be picked up (the toast said why): the copy goes again and the selection comes back.
            command.undo(&mut self.world, &mut self.selection.primary);
            self.selection.set_many(originals);
            if primary.is_some_and(|entity| self.selection.contains(entity)) {
                self.selection.primary = primary;
            }
            self.after_selection_change();
            return;
        }
        let count = command.roots().len();
        let mut lost = false;
        if let Some(carry) = self.editing.carry.as_mut() {
            // Shift is still down from the key: no tilt until it has been let go once.
            carry.upright = self.shortcut_modifiers.shift;
            carry.copy = Some(CarriedCopy { command, originals, primary });
            lost = carry.lost;
        }
        if !lost && let Some(ui) = self.ui_manager.as_mut() {
            ui.set_status_text(CARRY_COPY_HINT);
        }
        info!(objects = count, "carry: copied and picked up");
    }

    /// Why a carry cannot start now, in the words the gizmo uses where it has some.
    fn carry_refusal(&self) -> Option<&'static str> {
        if self.play_session_active {
            Some("Stop Play to move objects")
        } else if self.settings.editor().select_only {
            Some("Select Only is on — click it in the toolbar to move, rotate and scale")
        } else if self.terrain_edit_active || self.foliage_paint_active || self.vertex_paint_active {
            Some("Put the brush away to move objects")
        } else if self.gizmo_drag.is_some() {
            Some("Let go of the gizmo first")
        } else {
            None
        }
    }

    /// Pick up the selection. True when the carry began; when it cannot, a toast says why.
    pub(crate) fn begin_carry(&mut self) -> bool {
        let refuse = |this: &mut Self, message: &str| {
            this.toast(message);
            false
        };
        if let Some(message) = self.carry_refusal() {
            return refuse(self, message);
        }
        let Some(primary) = self.selection.primary.filter(|e| self.world.is_alive(*e)) else {
            return refuse(self, "Select an object first");
        };
        if self.renderer.is_none() {
            return refuse(self, "Nothing is drawn yet to move it over");
        }
        crate::propagate_transforms(&mut self.world);
        let selection: Vec<Entity> = self.selection.as_slice().to_vec();
        let Some(followers) = capture_followers(&self.world, &selection, primary.index()) else {
            return refuse(self, "Its parent is squashed flat, so it cannot be moved");
        };
        let riders: Vec<GizmoFollower> = followers.into_iter().filter(|f| f.rides_primary).collect();
        let ridden: HashSet<u32> = riders.iter().map(|f| f.entity_index).collect();
        let still = |world: &World, e: Entity| {
            crate::is_hidden(world, e) || world.get::<EditorFlags>(e).is_some_and(|flags| flags.locked)
        };
        let mut chosen: Vec<Entity> = self
            .selected_objects()
            .into_iter()
            .filter(|e| !ridden.contains(&e.index()) && !still(&self.world, *e) && self.world.get::<Transform>(*e).is_some())
            .collect();
        if chosen.is_empty() {
            return refuse(self, "Nothing to move: what is selected is locked or hidden");
        }
        // The lead: the object that is the primary, or holds it.
        let holds = |world: &World, object: Entity| {
            let mut current = primary;
            for _ in 0..64 {
                if current == object {
                    return true;
                }
                match world.get::<Parent>(current).map(|p| p.entity) {
                    Some(parent) if world.is_alive(parent) => current = parent,
                    _ => return false,
                }
            }
            false
        };
        if let Some(at) = chosen.iter().position(|e| holds(&self.world, *e)) {
            chosen.swap(0, at);
        }
        let lead_is_primary = chosen[0] == primary;
        let children = children_map(&self.world);
        let mut members: HashSet<Entity> = HashSet::new();
        let mut objects: Vec<CarryObject> = Vec::new();
        let mut bounds: Vec<(glam::Vec3, glam::Vec3)> = Vec::new();
        for &entity in &chosen {
            let Some(start) = self.world.get::<Transform>(entity).copied() else {
                continue;
            };
            let parent_world = parent_world_matrix(&self.world, entity);
            let Some(parent_inverse) = invert_affine(parent_world) else {
                return refuse(self, "Its parent is squashed flat, so it cannot be moved");
            };
            let parts = subtree(&children, entity);
            let origin = model_of(&self.world, entity).map_or(start.translation, |m| m.w_axis.truncate());
            let renderer = self.renderer.as_ref().expect("checked above");
            let (min, max) = world_bounds(&self.world, renderer, &parts).unwrap_or((origin, origin));
            let anchor = glam::Vec3::new((min.x + max.x) * 0.5, min.y, (min.z + max.z) * 0.5);
            members.extend(parts);
            bounds.push((min, max));
            objects.push(CarryObject {
                entity,
                start,
                parent_inverse,
                parent_rotation: parent_world.to_scale_rotation_translation().1,
                anchor,
                arm: origin - anchor,
                height: max.y - min.y,
            });
        }
        for rider in &riders {
            if let Some(entity) = self.world.find_entity_by_index(rider.entity_index) {
                members.extend(subtree(&children, entity));
            }
        }
        let starts: Vec<(Entity, Transform)> = selection
            .iter()
            .filter_map(|e| Some((*e, *self.world.get::<Transform>(*e)?)))
            .collect();
        let Some(surfaces) = self.gather_surfaces(members, &bounds) else {
            return refuse(self, "Nothing is drawn yet to move it over");
        };
        let mut sorted = selection.clone();
        sorted.sort_unstable_by_key(|e| e.index());
        let lead_anchor = objects[0].anchor;
        let mut carry = CarryState {
            objects,
            riders,
            lead_is_primary,
            starts,
            selection: sorted,
            surfaces,
            start_point: lead_anchor,
            point: None,
            turns: 0,
            pixels: 0.0,
            lost: true,
            seen: None,
            copy: None,
            upright: false,
        };
        // Nothing jumps: the lead keeps its distance from the cursor's point on the ground. With no ground under the
        // cursor it waits, and goes to the cursor when the cursor finds some.
        if let Some(point) = self.carry_cursor_point(&carry.surfaces) {
            carry.start_point = point;
            carry.point = Some(point);
            carry.lost = false;
        }
        let lost = carry.lost;
        self.editing.carry = Some(carry);
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.clear_gizmo();
        }
        if let Some(ui) = self.ui_manager.as_mut() {
            ui.set_status_text(if lost { CARRY_LOST } else { CARRY_HINT });
        }
        info!(objects = chosen.len(), "carry: picked up");
        true
    }

    /// Each frame: the carried things follow the cursor. Ends the carry (as Esc) when the selection, the scene or the
    /// mode it began in is gone; waits while the right button flies the camera.
    pub(crate) fn update_carry(&mut self) {
        let Some(mut carry) = self.editing.carry.take() else {
            return;
        };
        // Shift held over from Shift+D has been let go: from now on it tilts again.
        if carry.upright && !self.shortcut_modifiers.shift {
            carry.upright = false;
        }
        let mut now: Vec<Entity> = self.selection.as_slice().to_vec();
        now.sort_unstable_by_key(|e| e.index());
        let gone = carry.objects.iter().any(|object| !self.world.is_alive(object.entity));
        if self.play_session_active || gone || now != carry.selection || self.carry_refusal().is_some() {
            self.editing.carry = Some(carry);
            self.cancel_carry();
            return;
        }
        if self.ui_manager.as_ref().is_some_and(somnium_ui::UiManager::viewport_camera_active) {
            // The camera is being flown: the thing waits, and is placed afresh when the button is let go.
            carry.seen = None;
            self.editing.carry = Some(carry);
            return;
        }
        self.place_carry(&mut carry, false);
        self.editing.carry = Some(carry);
    }

    /// Put the carried things where the cursor says, unless nothing changed since the last time (or `force`).
    fn place_carry(&mut self, carry: &mut CarryState, force: bool) {
        // Over a panel the pointer is the panel's: the thing waits where it is.
        if !self.carry_pointer_in_view() {
            return;
        }
        let tilt = self.shortcut_modifiers.shift && !carry.upright;
        let ctrl = self.shortcut_modifiers.command()
            || self.ui_manager.as_ref().is_some_and(somnium_ui::UiManager::command_modifier_held);
        let key = (self.cursor_pos, tilt, ctrl, carry.turns);
        if !force && carry.seen == Some(key) {
            return;
        }
        carry.seen = Some(key);
        match self.carry_cursor_point(&carry.surfaces) {
            Some(point) => {
                carry.point = Some(point);
                if carry.lost {
                    carry.lost = false;
                    if let Some(ui) = self.ui_manager.as_mut() {
                        ui.set_status_text(if carry.copy.is_some() { CARRY_COPY_HINT } else { CARRY_HINT });
                    }
                }
            }
            None if !carry.lost => {
                carry.lost = true;
                if let Some(ui) = self.ui_manager.as_mut() {
                    ui.set_status_text(CARRY_LOST);
                }
            }
            None => {}
        }
        let Some(point) = carry.point else {
            return;
        };
        let carry: &CarryState = carry;
        let (grid, angle, scale) = {
            let editor = self.settings.editor();
            (editor.snap_translate_m, editor.snap_rotate_deg, editor.snap_scale)
        };
        // Ctrl applies or inverts the grid exactly as it does for the gizmo.
        let snap = SnapSettings { translate_m: grid, rotate_deg: angle, scale }.inverted(ctrl);
        let turn = glam::Quat::from_rotation_y((carry.turns as f32 * wheel_turn_degrees(angle)).to_radians());
        let lead_anchor = carry.objects[0].anchor;
        let lead = lead_plan(lead_anchor, carry.start_point, point, snap.translate_m);
        let mut placed: Vec<(Entity, Transform)> = Vec::with_capacity(carry.objects.len());
        for (i, object) in carry.objects.iter().enumerate() {
            let plan = if i == 0 { lead } else { follower_plan(object.anchor, lead_anchor, lead, turn) };
            // Its own ground, at its own place: a ray down from a little over where it would stand, the height it
            // stood at over the lead kept.
            let guess = point.y + (object.anchor.y - lead_anchor.y);
            let top = guess + object.height.max(0.5) + 0.5;
            let down = |this: &mut Self, from: f32| {
                this.carry_ray(&carry.surfaces, glam::Vec3::new(plan.x, from, plan.y), glam::Vec3::NEG_Y, editor_ground::SEARCH_DEPTH + (from - guess), true)
                    .map(|hit| (from - hit.distance, hit.normal))
            };
            let (ground, normal) = down(self, top)
                .or_else(|| down(self, top + CARRY_RETRY_LIFT))
                .unwrap_or((guess, glam::Vec3::Y));
            let tilt_turn = if tilt { glam::Quat::from_rotation_arc(glam::Vec3::Y, normal) * turn } else { turn };
            let origin = glam::Vec3::new(plan.x, ground, plan.y) + tilt_turn * object.arm;
            let mut transform = object.start;
            transform.translation = object.parent_inverse.transform_point3(origin);
            transform.rotation =
                (crate::editor_gizmo::spin_in_parent(object.parent_rotation, tilt_turn) * object.start.rotation).normalize();
            placed.push((object.entity, transform));
        }
        let lead_transform = placed[0].1;
        for (entity, transform) in placed {
            if let Some(t) = self.world.get_mut::<Transform>(entity) {
                *t = transform;
            }
        }
        if carry.lead_is_primary {
            self.ride_carry(carry, lead_transform);
        }
        crate::propagate_transforms(&mut self.world);
    }

    /// The primary's twins take its transform as it is.
    fn ride_carry(&mut self, carry: &CarryState, lead: Transform) {
        for rider in &carry.riders {
            if let Some(entity) = self.world.find_entity_by_index(rider.entity_index)
                && let Some(t) = self.world.get_mut::<Transform>(entity)
            {
                *t = lead;
            }
        }
    }

    /// Every surface in the loaded world, `members` left out, and the static boxes inside one of `bounds` (a carried
    /// object's own colliders). `None` without a renderer.
    fn gather_surfaces(&mut self, members: HashSet<Entity>, bounds: &[(glam::Vec3, glam::Vec3)]) -> Option<Surfaces> {
        crate::propagate_transforms(&mut self.world);
        let renderer = self.renderer.as_ref()?;
        let index = MeshIndex::build(&self.world, renderer);
        let terrains = self.prepare_terrains();
        let boxes: Vec<(glam::Mat4, glam::Vec3, glam::Vec3)> = self
            .physics
            .as_ref()
            .map(|physics| physics.static_boxes())
            .unwrap_or_default()
            .into_iter()
            .map(|(half, position, rotation)| {
                let model = glam::Mat4::from_scale_rotation_translation(half * 2.0, rotation, position);
                let (lo, hi) = transformed_box(model, glam::Vec3::splat(-0.5), glam::Vec3::splat(0.5));
                (model, lo, hi)
            })
            .filter(|(_, lo, hi)| {
                !bounds.iter().any(|(min, max)| {
                    lo.cmpge(*min - glam::Vec3::splat(0.1)).all() && hi.cmple(*max + glam::Vec3::splat(0.1)).all()
                })
            })
            .collect();
        Some(Surfaces { members, index, terrains, boxes })
    }

    /// The point on the nearest surface under the cursor, the carried things left out.
    fn carry_cursor_point(&mut self, surfaces: &Surfaces) -> Option<glam::Vec3> {
        let (origin, direction) = self.cursor_ray()?;
        let hit = self.carry_ray(surfaces, origin, direction, CARRY_REACH, false)?;
        Some(origin + direction * hit.distance)
    }

    /// The nearest surface along a unit ray within `max`: terrain, any mesh in the loaded world, a static box; never
    /// what is carried. `column`: the ray is short (straight down), so the mesh grid is asked; otherwise every mesh's
    /// box is tested.
    fn carry_ray(
        &mut self,
        surfaces: &Surfaces,
        origin: glam::Vec3,
        direction: glam::Vec3,
        max: f32,
        column: bool,
    ) -> Option<Hit> {
        let renderer = self.renderer.as_ref()?;
        let mut best = terrain_hit(renderer, &surfaces.terrains, origin, direction, max);
        let end = origin + direction * max;
        let (lo, hi) = (origin.min(end), origin.max(end));
        let index = &surfaces.index;
        let near: Vec<usize> = if column { index.query(lo, hi) } else { (0..index.items.len()).collect() };
        let mut entries: Vec<(f32, usize)> = near
            .into_iter()
            .filter(|&i| !surfaces.members.contains(&index.items[i].entity))
            .filter_map(|i| {
                let item = &index.items[i];
                box_entry(origin, direction, item.min, item.max).filter(|t| *t <= max).map(|t| (t, i))
            })
            .collect();
        entries.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (entry, i) in entries {
            let limit = best.map_or(max, |hit| hit.distance);
            if entry > limit {
                break;
            }
            let item = &index.items[i];
            let Some(bvh) = self.bvh_for(item.mesh) else {
                continue;
            };
            let surface = MeshSurface { bvh: &bvh, model: item.model, inverse: item.model.inverse(), min: item.min, max: item.max };
            if let Some(hit) = surface.raycast(origin, direction, limit)
                && best.is_none_or(|b| hit.distance < b.distance)
            {
                best = Some(hit);
            }
        }
        if !surfaces.boxes.is_empty() {
            let cube = self.editing.cube.get_or_insert_with(|| Arc::new(unit_cube())).clone();
            for &(model, box_min, box_max) in &surfaces.boxes {
                let limit = best.map_or(max, |hit| hit.distance);
                let surface = MeshSurface { bvh: &cube, model, inverse: model.inverse(), min: box_min, max: box_max };
                if let Some(hit) = surface.raycast(origin, direction, limit)
                    && best.is_none_or(|b| hit.distance < b.distance)
                {
                    best = Some(hit);
                }
            }
        }
        best
    }

    /// Click or Enter: put it down where it is. The last step is exactly Drop to Ground (Align to Surface with Shift
    /// held), and the whole move is one undo step, "Move on Ground".
    pub(crate) fn put_down_carry(&mut self) {
        let Some(mut carry) = self.editing.carry.take() else {
            return;
        };
        // Where the cursor is now: a click can come in the same batch as the move that brought it there.
        if !self.ui_manager.as_ref().is_some_and(somnium_ui::UiManager::viewport_camera_active) {
            self.place_carry(&mut carry, true);
        }
        let align = self.shortcut_modifiers.shift && !carry.upright;
        let objects: Vec<Entity> = carry.objects.iter().map(|object| object.entity).collect();
        if let Some((changes, _, _)) = self.ground_moves(&objects, align) {
            for (entity, _, end) in changes {
                if let Some(t) = self.world.get_mut::<Transform>(entity) {
                    *t = end;
                }
            }
        }
        if carry.lead_is_primary
            && let Some(lead) = self.world.get::<Transform>(carry.objects[0].entity).copied()
        {
            self.ride_carry(&carry, lead);
        }
        crate::propagate_transforms(&mut self.world);
        if let Some(copy) = carry.copy.take() {
            // A copy: the step is the copy itself, made where it now stands. Undo takes it out of the world, Redo
            // makes it there again.
            let mut command = copy.command;
            command.keep_root_places(&self.world);
            let count = command.roots().len();
            let name = copy
                .originals
                .first()
                .filter(|entity| self.world.is_alive(**entity))
                .map(|entity| name_of(&self.world, *entity))
                .unwrap_or_default();
            self.undo_stack.push_silent(Box::new(command));
            self.scene_dirty = true;
            self.toast(&copied_message(count, &name));
            info!(objects = count, "carry: copy put down");
            self.refresh_gizmo_anchor();
            return;
        }
        let mut moves: Vec<Box<dyn EditorCommand>> = Vec::new();
        for &(entity, start) in &carry.starts {
            if !self.world.is_alive(entity) {
                continue;
            }
            let end = self.world.get::<Transform>(entity).copied().unwrap_or(start);
            if end.translation != start.translation || end.rotation != start.rotation || end.scale != start.scale {
                moves.push(Box::new(SetTransformCmd::new(entity.index(), start, end)));
            }
        }
        let lead = &carry.objects[0];
        let from = lead.anchor + lead.arm;
        let to = model_of(&self.world, lead.entity).map_or(from, |m| m.w_axis.truncate());
        let message = if carry.objects.len() == 1 {
            format!("Moved {} {:.1} m.", name_of(&self.world, lead.entity), from.distance(to))
        } else {
            format!("Moved {} objects.", carry.objects.len())
        };
        if !moves.is_empty() {
            self.undo_stack
                .push_silent(Box::new(CommandGroup::named("Move on Ground".to_owned(), moves)));
            self.scene_dirty = true;
        }
        self.toast(&message);
        self.refresh_gizmo_anchor();
    }

    /// Esc: put everything back where it was. Nothing changed, nothing to undo.
    pub(crate) fn cancel_carry(&mut self) {
        let Some(mut carry) = self.editing.carry.take() else {
            return;
        };
        if let Some(mut copy) = carry.copy.take() {
            // A copy in hand never entered the history: it leaves the world, and what was selected before is
            // selected again (unless something else was selected meanwhile, which then stays).
            use crate::editor_commands::EditorCommand as _;
            let mut now: Vec<Entity> = self.selection.as_slice().to_vec();
            now.sort_unstable_by_key(|e| e.index());
            let still_the_copy = now == carry.selection;
            copy.command.undo(&mut self.world, &mut self.selection.primary);
            self.selection.retain_alive(&self.world);
            if still_the_copy {
                let world = &self.world;
                self.selection.set_many(copy.originals.iter().copied().filter(|e| world.is_alive(*e)));
                if let Some(primary) = copy.primary.filter(|e| self.selection.contains(*e)) {
                    self.selection.primary = Some(primary);
                }
            }
            self.after_selection_change();
            crate::propagate_transforms(&mut self.world);
            if let Some(ui) = self.ui_manager.as_mut() {
                ui.set_status_text("The copy was removed.");
            }
            info!("carry: copy removed");
            self.refresh_gizmo_anchor();
            return;
        }
        for &(entity, start) in &carry.starts {
            if self.world.is_alive(entity)
                && let Some(t) = self.world.get_mut::<Transform>(entity)
            {
                *t = start;
            }
        }
        crate::propagate_transforms(&mut self.world);
        if let Some(ui) = self.ui_manager.as_mut() {
            ui.set_status_text("Put back where it was.");
        }
        info!("carry: put back");
        self.refresh_gizmo_anchor();
    }

    /// A key pressed while carrying: Esc puts it back, Enter puts it down. True when the key was the carry's.
    pub(crate) fn carry_key(&mut self, code: winit::keyboard::KeyCode) -> bool {
        use winit::keyboard::KeyCode as K;
        if !self.carrying() {
            return false;
        }
        match code {
            K::Escape => self.cancel_carry(),
            K::Enter | K::NumpadEnter => self.put_down_carry(),
            _ => return false,
        }
        true
    }

    /// A mouse event while carrying: the left button puts it down, the wheel turns it (not while the right button flies
    /// the camera: the wheel is the camera's then). True when the event was the carry's.
    pub(crate) fn carry_mouse_event(&mut self, event: &winit::event::WindowEvent) -> bool {
        use winit::event::{ElementState, MouseButton, WindowEvent};
        if !self.carrying() {
            return false;
        }
        match event {
            WindowEvent::MouseInput { state: ElementState::Pressed, button: MouseButton::Left, .. } => {
                // A click on a panel, a menu or the toolbar is theirs, and the thing stays in hand.
                if !self.carry_pointer_in_view() {
                    return false;
                }
                self.put_down_carry();
                true
            }
            WindowEvent::MouseWheel { delta, .. } => self.carry_pointer_in_view() && self.carry_wheel(*delta),
            _ => false,
        }
    }

    /// The pointer is over the 3D view itself, not a panel or a control floating on it (a script's pointer always is:
    /// it is given in the window's own fractions).
    fn carry_pointer_in_view(&self) -> bool {
        self.editing.script_pointer
            || self.ui_manager.as_ref().is_none_or(somnium_ui::UiManager::pointer_over_viewport)
    }

    /// Turn the carried things by whole notches of the wheel. False while the camera has the wheel.
    pub(crate) fn carry_wheel(&mut self, delta: winit::event::MouseScrollDelta) -> bool {
        if self.ui_manager.as_ref().is_some_and(somnium_ui::UiManager::viewport_camera_active) {
            return false;
        }
        let Some(carry) = self.editing.carry.as_mut() else {
            return false;
        };
        carry.turns += wheel_notches(delta, &mut carry.pixels);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Nothing jumps: picked up with the cursor's point 3 m from the crate, the crate stays 3 m from it, and moves by
    /// exactly as much as that point moves.
    #[test]
    fn the_lead_keeps_its_distance_from_the_cursor_point() {
        let anchor = glam::Vec3::new(10.0, 2.0, -4.0);
        let start_point = glam::Vec3::new(13.0, 1.5, -4.0);
        assert_eq!(lead_plan(anchor, start_point, start_point, 0.0), glam::Vec2::new(10.0, -4.0), "it jumped at the start");
        let moved = lead_plan(anchor, start_point, start_point + glam::Vec3::new(5.0, 9.0, -2.5), 0.0);
        assert!(moved.abs_diff_eq(glam::Vec2::new(15.0, -6.5), 1e-5));
        // On a 1 m grid the place is rounded, seen from above only.
        assert_eq!(lead_plan(anchor, start_point, start_point + glam::Vec3::new(0.4, 0.0, 0.6), 1.0), glam::Vec2::new(10.0, -3.0));
    }

    /// A stall and two crates keep their plan: each the same distance and direction from the stall seen from above,
    /// and turned with it when the wheel turns.
    #[test]
    fn several_objects_keep_their_plan_and_turn_together() {
        let stall = glam::Vec3::new(0.0, 5.0, 0.0);
        let crates = [glam::Vec3::new(2.0, 5.3, 0.0), glam::Vec3::new(-1.0, 4.0, 3.0)];
        let lead = glam::Vec2::new(40.0, 20.0);
        for anchor in crates {
            let plan = follower_plan(anchor, stall, lead, glam::Quat::IDENTITY);
            assert!(plan.abs_diff_eq(lead + glam::Vec2::new(anchor.x, anchor.z), 1e-5));
        }
        let quarter = glam::Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let turned = follower_plan(crates[0], stall, lead, quarter);
        // +X turned a quarter about up is -Z.
        assert!(turned.abs_diff_eq(lead + glam::Vec2::new(0.0, -2.0), 1e-5));
        assert!((turned.distance(lead) - 2.0).abs() < 1e-5, "the crate left its distance from the stall");
    }

    /// A script names keys the way the guide writes them, and points in fractions of the window.
    #[test]
    fn a_script_names_its_keys_and_points() {
        use winit::keyboard::KeyCode as K;
        assert_eq!(script_key("g"), Some(K::KeyG));
        assert_eq!(script_key("Esc"), Some(K::Escape));
        assert_eq!(script_key(" enter "), Some(K::Enter));
        assert_eq!(script_key("f2"), Some(K::F2));
        assert_eq!(script_key("gg"), None);
        assert_eq!(script_point("0.5, 0.25"), Some((0.5, 0.25)));
        assert_eq!(script_point("0.5"), None);
        assert_eq!(script_modifiers("ctrl+shift").map(|m| (m.ctrl, m.shift, m.alt)), Some((true, true, false)));
        assert_eq!(script_modifiers("none").map(|m| (m.ctrl, m.shift, m.alt)), Some((false, false, false)));
        assert!(script_modifiers("hyper").is_none());
    }

    /// Putting a copy down says what was copied, by name when it is one thing.
    #[test]
    fn a_copy_put_down_is_named() {
        assert_eq!(copied_message(1, "gym_crate_a"), "Copied gym_crate_a.");
        assert_eq!(copied_message(1, ""), "Copied 1 object.");
        assert_eq!(copied_message(3, "gym_crate_a"), "Copied 3 objects.");
    }

    /// One notch is the bar's snap angle, or 15 degrees when it is off; a touchpad's pixels add up to notches.
    #[test]
    fn a_wheel_notch_is_the_snap_angle() {
        assert_eq!(wheel_turn_degrees(0.0), 15.0);
        assert_eq!(wheel_turn_degrees(45.0), 45.0);
        assert_eq!(wheel_turn_degrees(f32::NAN), 15.0);
        let mut pixels = 0.0;
        assert_eq!(wheel_notches(winit::event::MouseScrollDelta::LineDelta(0.0, 1.0), &mut pixels), 1);
        assert_eq!(wheel_notches(winit::event::MouseScrollDelta::LineDelta(0.0, -2.0), &mut pixels), -2);
        assert_eq!(wheel_notches(winit::event::MouseScrollDelta::LineDelta(0.0, 0.3), &mut pixels), 1);
        let pixel = |y| winit::event::MouseScrollDelta::PixelDelta(winit::dpi::PhysicalPosition::new(0.0, y));
        assert_eq!(wheel_notches(pixel(30.0), &mut pixels), 0);
        assert_eq!(wheel_notches(pixel(30.0), &mut pixels), 1);
        assert!((pixels - 10.0).abs() < 1e-9);
    }
}
