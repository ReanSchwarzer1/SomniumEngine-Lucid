//! Autosave and crash recovery — CONTROL-J, craft defect C11.
//!
//! Autosaves go to `<content root>/.somnium/autosave/`, beside the preview
//! cache and the asset index, because they are session state rather than
//! content: an autosave must never appear in the Content Drawer, and a folder
//! the drawer already ignores is the honest place for it.
//!
//! The policy here is pure so it can be tested without a clock, a filesystem
//! or a scene. The interesting decisions — when an autosave is *due*, and when
//! one on disk is worth offering back — are exactly the ones that are painful
//! to verify by hand.

use std::path::{Path, PathBuf};

/// The folder autosaves live in, relative to the content root.
pub const AUTOSAVE_DIR: &str = ".somnium/autosave";

/// Why an autosave is being written.
///
/// Named rather than a bool because the two have different *file names* — a
/// pre-Play snapshot is the one you want back after a play session goes wrong,
/// and an interval save would otherwise overwrite it within the minute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutosaveReason {
    /// The interval elapsed.
    Interval,
    /// Play is about to start.
    BeforePlay,
}

impl AutosaveReason {
    /// The file this reason writes to.
    #[must_use]
    pub const fn file_name(self) -> &'static str {
        match self {
            Self::Interval => "autosave.somnium",
            Self::BeforePlay => "before-play.somnium",
        }
    }
}

/// Where an autosave goes, given the content root.
#[must_use]
pub fn autosave_path(content_root: &Path, reason: AutosaveReason) -> PathBuf {
    content_root.join(AUTOSAVE_DIR).join(reason.file_name())
}

/// The note beside an autosave that names the scene it is a copy of (`autosave.scene` beside `autosave.somnium`):
/// one line, the scene's path, or empty for a scene never saved to a file.
///
/// Without it an autosave cannot be matched to anything. Until 27 Nov 2026 recovery compared every autosave with a
/// fixed `scene.somnium` that no project has, so an autosave left by any scene was offered at every start, for
/// good (Sunscar's editor offered one from 20 Nov a week later).
#[must_use]
pub fn scene_note_path(autosave: &Path) -> PathBuf {
    autosave.with_extension("scene")
}

/// Write the note naming the scene an autosave copies (see [`scene_note_path`]). A new file renamed over the old,
/// so a hard-linked copy of the folder keeps its own.
pub fn note_scene(autosave: &Path, scene: Option<&Path>) -> Result<(), String> {
    let line = scene.map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    crate::save_game::atomic_write(&scene_note_path(autosave), line.as_bytes())
}

/// The scene a note names: `None` with no note (an older build's autosave, which is never offered), `Some(None)`
/// for a scene never saved to a file.
fn noted_scene(autosave: &Path) -> Option<Option<PathBuf>> {
    let line = std::fs::read_to_string(scene_note_path(autosave)).ok()?;
    let line = line.trim();
    Some((!line.is_empty()).then(|| PathBuf::from(line)))
}

/// When the work a scene holds was last saved: its file or its edit layer (`<scene>.edits.json`, what Ctrl+S
/// writes for a scene a generator owns), whichever is newer. `None` when neither exists.
fn scene_saved_secs(scene: &Path) -> Option<u64> {
    let layer = modified_secs(&crate::scene_edits::layer_path(scene));
    modified_secs(scene).max(layer)
}

/// Tracks when the next interval autosave is due.
///
/// Deliberately *not* a timer thread. Autosave has to serialise the world, so
/// it can only happen on the frame thread anyway, and a `due` predicate the
/// frame loop asks is both simpler and testable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutosaveClock {
    /// Seconds between saves. `0` disables autosave entirely.
    pub interval_secs: f32,
    /// Seconds elapsed since the last save.
    since_last: f32,
    /// Whether the scene has changed since the last autosave.
    dirty: bool,
}

impl Default for AutosaveClock {
    fn default() -> Self {
        Self {
            interval_secs: 300.0,
            since_last: 0.0,
            dirty: false,
        }
    }
}

impl AutosaveClock {
    /// A clock with the given interval.
    #[must_use]
    pub fn new(interval_secs: f32) -> Self {
        Self {
            interval_secs,
            since_last: 0.0,
            dirty: false,
        }
    }

    /// Advance by `dt` seconds, and report whether a save is due.
    ///
    /// Due means: autosave is on, the interval has elapsed, **and** something
    /// has changed. The last clause is what stops an idle editor rewriting the
    /// same file every five minutes forever — the file's timestamp is what
    /// crash recovery compares, and a stream of identical saves would make it
    /// meaningless.
    pub fn tick(&mut self, dt: f32, scene_dirty: bool) -> bool {
        self.dirty |= scene_dirty;
        if self.interval_secs <= 0.0 {
            return false;
        }
        self.since_last += dt.max(0.0);
        if self.since_last < self.interval_secs || !self.dirty {
            return false;
        }
        self.since_last = 0.0;
        self.dirty = false;
        true
    }

    /// Note that a save happened for some other reason, so the interval
    /// restarts from now.
    pub fn saved(&mut self) {
        self.since_last = 0.0;
        self.dirty = false;
    }

    /// Change the interval, keeping elapsed time. Used when the setting moves.
    pub fn set_interval(&mut self, interval_secs: f32) {
        self.interval_secs = interval_secs.max(0.0);
    }
}

/// An autosave found on disk that is worth offering back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recovery {
    /// The autosave file.
    pub path: PathBuf,
    /// Why it was written.
    pub reason: AutosaveReason,
    /// Its modification time, seconds since the Unix epoch.
    pub saved_unix_secs: u64,
    /// The scene it is a copy of; `None` for a scene never saved to a file.
    pub scene: Option<PathBuf>,
}

/// Decide whether an autosave should be offered, given both timestamps.
///
/// Offered only when the autosave is **newer** than the scene it shadows. An
/// autosave older than the last real save is work the person already committed
/// by hand, and offering it back would invite them to overwrite good work with
/// stale work — the failure mode that makes people distrust crash recovery.
///
/// A scene that does not exist at all (`None`) means every autosave is newer
/// than nothing, which is right: an unsaved scene is exactly the case
/// recovery is for.
#[must_use]
pub fn should_offer(autosave_secs: u64, scene_secs: Option<u64>) -> bool {
    match scene_secs {
        None => true,
        Some(scene) => autosave_secs > scene,
    }
}

/// Look for a recoverable autosave beside `content_root`.
///
/// Each autosave is held against the scene its note names ([`scene_note_path`]): offered only when it is newer
/// than that scene's last save, file or edit layer. One with no note is never offered: nothing says what it copies.
///
/// The pre-Play snapshot wins a tie, because if both exist at the same second
/// the one taken deliberately before a risky operation is the one worth having.
#[must_use]
pub fn find_recovery(content_root: &Path) -> Option<Recovery> {
    [AutosaveReason::BeforePlay, AutosaveReason::Interval]
        .into_iter()
        .filter_map(|reason| {
            let path = autosave_path(content_root, reason);
            let saved_unix_secs = modified_secs(&path)?;
            let scene = noted_scene(&path)?;
            let scene_secs = scene.as_deref().and_then(scene_saved_secs);
            should_offer(saved_unix_secs, scene_secs).then_some(Recovery {
                path,
                reason,
                saved_unix_secs,
                scene,
            })
        })
        .max_by_key(|recovery| recovery.saved_unix_secs)
}

/// A file's modification time in seconds since the Unix epoch, or `None` if it
/// does not exist or the filesystem will not say.
#[must_use]
pub fn modified_secs(path: &Path) -> Option<u64> {
    std::fs::metadata(path)
        .ok()?
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|elapsed| elapsed.as_secs())
}

/// Delete both autosaves and their notes. Called after a clean manual save, and when the editor that wrote them
/// closes cleanly, because at either moment there is nothing left to recover.
pub fn clear(content_root: &Path) {
    for reason in [AutosaveReason::Interval, AutosaveReason::BeforePlay] {
        let path = autosave_path(content_root, reason);
        let _ = std::fs::remove_file(scene_note_path(&path));
        let _ = std::fs::remove_file(path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The interval only fires when something actually changed. An idle editor
    /// rewriting the same file every five minutes would make the timestamp
    /// crash recovery compares meaningless.
    #[test]
    fn an_idle_editor_never_autosaves() {
        let mut clock = AutosaveClock::new(60.0);
        for _ in 0..200 {
            assert!(!clock.tick(1.0, false));
        }
    }

    #[test]
    fn a_dirty_scene_saves_once_per_interval() {
        let mut clock = AutosaveClock::new(60.0);
        for _ in 0..59 {
            assert!(!clock.tick(1.0, true));
        }
        assert!(clock.tick(1.0, true), "the sixtieth second is due");
        // …and the counter restarts rather than firing every frame after.
        assert!(!clock.tick(1.0, true));
    }

    /// A change during the interval still counts, even if the scene is clean
    /// again by the time the interval elapses — an edit and an undo is still
    /// a session worth recovering.
    #[test]
    fn a_change_anywhere_in_the_interval_arms_the_save() {
        let mut clock = AutosaveClock::new(10.0);
        assert!(!clock.tick(1.0, true));
        for _ in 0..8 {
            assert!(!clock.tick(1.0, false));
        }
        assert!(clock.tick(1.0, false), "the change earlier still counts");
    }

    #[test]
    fn a_zero_interval_disables_autosave() {
        let mut clock = AutosaveClock::new(0.0);
        for _ in 0..1_000 {
            assert!(!clock.tick(60.0, true));
        }
    }

    #[test]
    fn a_manual_save_restarts_the_interval() {
        let mut clock = AutosaveClock::new(10.0);
        for _ in 0..9 {
            clock.tick(1.0, true);
        }
        clock.saved();
        assert!(!clock.tick(1.0, false), "the countdown started again");
    }

    /// The rule that makes recovery trustworthy: never offer work older than
    /// what the person already saved by hand.
    #[test]
    fn an_autosave_older_than_the_scene_is_not_offered() {
        assert!(should_offer(200, Some(100)), "newer is offered");
        assert!(!should_offer(100, Some(200)), "older is not");
        assert!(!should_offer(100, Some(100)), "identical is not");
        assert!(
            should_offer(1, None),
            "an unsaved scene is what this is for"
        );
    }

    /// The two reasons write to different files, so an interval save cannot
    /// overwrite the snapshot taken deliberately before Play.
    #[test]
    fn the_two_reasons_do_not_overwrite_each_other() {
        let root = Path::new("content");
        assert_ne!(
            autosave_path(root, AutosaveReason::Interval),
            autosave_path(root, AutosaveReason::BeforePlay)
        );
        assert!(
            autosave_path(root, AutosaveReason::Interval)
                .to_string_lossy()
                .contains(".somnium"),
            "autosaves live where the drawer does not look"
        );
    }

    /// End to end against a real filesystem: two autosaves and a scene, and
    /// the newest of the ones worth offering wins.
    #[test]
    fn recovery_finds_the_newest_offerable_autosave() {
        let root = std::env::temp_dir().join(format!(
            "somnium_autosave_{}_{}",
            std::process::id(),
            line!()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join(AUTOSAVE_DIR)).unwrap();
        let scene = root.join("scene.somnium");
        std::fs::write(&scene, b"{}").unwrap();

        // No autosaves yet.
        assert_eq!(find_recovery(&root), None);

        // One written after the scene is offered.
        std::thread::sleep(std::time::Duration::from_millis(1100));
        let interval = autosave_path(&root, AutosaveReason::Interval);
        std::fs::write(&interval, b"{}").unwrap();
        note_scene(&interval, Some(&scene)).unwrap();
        let found = find_recovery(&root).expect("an autosave newer than the scene");
        assert_eq!(found.path, interval);
        assert_eq!(found.reason, AutosaveReason::Interval);
        assert_eq!(found.scene.as_deref(), Some(scene.as_path()));

        // Clearing removes both, notes and all.
        clear(&root);
        assert_eq!(find_recovery(&root), None);
        assert!(!scene_note_path(&interval).exists());
        let _ = std::fs::remove_dir_all(root);
    }

    /// 27 Nov 2026, Sunscar's editor: an autosave from 20 Nov was offered at every start, because recovery held it
    /// against a `scene.somnium` no project has. One with no note naming its scene is never offered, and one older
    /// than the scene's edit layer (Ctrl+S on a generated world writes only the layer) is not either.
    #[test]
    fn an_autosave_is_held_against_the_scene_it_copies() {
        let root = std::env::temp_dir().join(format!("somnium_autosave_{}_{}", std::process::id(), line!()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join(AUTOSAVE_DIR)).unwrap();
        let scene = root.join("world.scene.json");
        std::fs::write(&scene, b"{}").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1100));
        let before_play = autosave_path(&root, AutosaveReason::BeforePlay);
        std::fs::write(&before_play, b"{}").unwrap();

        // An older build's autosave: no note, no offer, however new it is.
        assert_eq!(find_recovery(&root), None, "an autosave with no note is not offered");

        // Noted, newer than the scene: offered.
        note_scene(&before_play, Some(&scene)).unwrap();
        assert!(find_recovery(&root).is_some());

        // The layer saved after it: the work is in the layer, nothing to offer.
        std::thread::sleep(std::time::Duration::from_millis(1100));
        std::fs::write(crate::scene_edits::layer_path(&scene), b"{}").unwrap();
        assert_eq!(find_recovery(&root), None, "older than the edit layer");

        // A scene never saved to a file: the autosave is all there is.
        note_scene(&before_play, None).unwrap();
        let found = find_recovery(&root).expect("an unsaved scene's autosave");
        assert_eq!(found.scene, None);
        let _ = std::fs::remove_dir_all(root);
    }
}
