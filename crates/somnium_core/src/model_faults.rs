//! Models that did not load this run: the files the importer refused ("Skipped a broken model") and the files whose
//! imported meshes could not be restored (a scene names a file that is not there, or one that cannot be read). The
//! editor's status bar and a toast that stays, the standalone game's screen and the run's last log line all carry the
//! two counts, so a world with pieces missing is never taken for a whole one. (Sunscar, 12 Nov 2026: for two hours no
//! far mesh loaded, every launch's log said so about 220 times, and every still of that time was passed as proof.)

use std::collections::BTreeSet;

use somnium_ui::message::{NodeHandle, TextMessage};
use somnium_ui::widget::WidgetBuilder;
use somnium_ui::widgets::text::TextBuilder;
use somnium_ui::{GameUiFrame, UiCanvas};

/// How many files the screen's line names.
const NAMED: usize = 3;
/// Frames without a new fault before the editor's toast: a ring of far tiles fails over many frames, and the one
/// toast that stays should carry the whole count.
const SETTLED_FRAMES: u32 = 120;

#[derive(Default)]
pub(crate) struct ModelFaults {
    /// Files the importer refused, each once.
    skipped: BTreeSet<String>,
    /// Files whose imported meshes could not be restored, each once.
    unrestored: BTreeSet<String>,
    /// The counts [`Self::tick`] last reported.
    told: (usize, usize),
    /// Frames since the counts last changed, while a toast is owed.
    settling: Option<u32>,
}

impl ModelFaults {
    pub(crate) fn skipped(&mut self, file: String) {
        self.skipped.insert(file);
    }

    pub(crate) fn unrestored(&mut self, source: &str) {
        if !self.unrestored.contains(source) {
            self.unrestored.insert(source.to_owned());
        }
    }

    /// Broken models skipped and models not loaded so far this run, in files.
    pub(crate) fn counts(&self) -> (usize, usize) {
        (self.skipped.len(), self.unrestored.len())
    }

    /// Once a frame. First: the counts, on the frame they changed (for the log and the status bar). Second: the
    /// screen's line, once the counts have stood still for [`SETTLED_FRAMES`] (for the toast).
    pub(crate) fn tick(&mut self) -> (Option<(usize, usize)>, Option<String>) {
        let now = self.counts();
        if now != self.told {
            self.told = now;
            self.settling = Some(0);
            return (Some(now), None);
        }
        match self.settling.as_mut() {
            Some(frames) if *frames >= SETTLED_FRAMES => {
                self.settling = None;
                (None, self.line())
            }
            Some(frames) => {
                *frames += 1;
                (None, None)
            }
            None => (None, None),
        }
    }

    /// The screen's one line; `None` while nothing is missing.
    pub(crate) fn line(&self) -> Option<String> {
        let counts = self.counts();
        (counts != (0, 0)).then(|| line(counts, self.skipped.iter().chain(&self.unrestored).map(String::as_str)))
    }
}

/// "2 broken models skipped, 220 models not loaded".
pub(crate) fn counted((skipped, unrestored): (usize, usize)) -> String {
    let s = |n: usize| if n == 1 { "" } else { "s" };
    format!("{skipped} broken model{} skipped, {unrestored} model{} not loaded", s(skipped), s(unrestored))
}

/// The counts, the first [`NAMED`] files by name and how many more. It says "failed", which is what makes the
/// editor's toast stay until it is dismissed (`LogSeverity::infer`).
fn line<'a>(counts: (usize, usize), files: impl Iterator<Item = &'a str>) -> String {
    let names: Vec<&str> = files.map(|file| file.rsplit(['/', '\\']).next().unwrap_or(file)).collect();
    let more = match names.len().saturating_sub(NAMED) {
        0 => String::new(),
        more => format!(" and {more} more"),
    };
    let named = names.iter().take(NAMED).copied().collect::<Vec<_>>().join(", ");
    format!("Models failed to load: {}: {named}{more} (the log names each)", counted(counts))
}

/// The line on the standalone game's screen, bottom left. The editor has its status bar and a toast instead.
#[derive(Default)]
pub(crate) struct Notice {
    card: Option<(UiCanvas, NodeHandle, glam::Vec2)>,
    shown: String,
}

impl Notice {
    pub(crate) fn draw(&mut self, frame: &mut GameUiFrame, text: &str) {
        let (canvas, node, laid) = self.card.get_or_insert_with(|| {
            let mut canvas = UiCanvas::new(1280.0, 720.0);
            let font = canvas.font_id();
            let root = canvas.ui().root();
            let node = TextBuilder::new(WidgetBuilder::new())
                .with_text("")
                .with_font_id(font)
                .with_font_size(18.0)
                .with_color([255, 110, 80, 255])
                .build();
            let node = canvas.ui_mut().add_node(node, root);
            (canvas, node, glam::Vec2::ZERO)
        });
        let window = frame.window();
        let scale = window.scale_factor() as f32;
        let size = glam::Vec2::new(window.inner_size().width.max(1) as f32, window.inner_size().height.max(1) as f32) / scale;
        if size != *laid {
            canvas.place_node(*node, somnium_ui::types::Rect::new(16.0, (size.y - 44.0).max(0.0), (size.x - 32.0).max(64.0), 32.0));
            *laid = size;
        }
        if self.shown != text {
            canvas.ui_mut().send(TextMessage::set_text(*node, text.to_owned()));
            text.clone_into(&mut self.shown);
        }
        frame.draw(canvas);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn what_did_not_load_is_counted_once_a_file_and_said_when_it_changes_then_once_it_settles() {
        let mut faults = ModelFaults::default();
        assert_eq!(faults.tick(), (None, None), "a whole world says nothing");
        assert_eq!(faults.line(), None);
        faults.skipped("assets/trees/holm_oak_b.gltf".into());
        faults.unrestored("scenes/terrain_tiles/far/t_-17_0.gltf");
        faults.unrestored("scenes/terrain_tiles/far/t_-17_0.gltf");
        assert_eq!(faults.tick(), (Some((1, 1)), None), "the counts on the frame they change, each file once");
        for _ in 0..SETTLED_FRAMES {
            assert_eq!(faults.tick(), (None, None), "no toast while more may follow");
        }
        let (changed, settled) = faults.tick();
        assert_eq!(changed, None);
        assert_eq!(
            settled.as_deref(),
            Some("Models failed to load: 1 broken model skipped, 1 model not loaded: holm_oak_b.gltf, t_-17_0.gltf (the log names each)")
        );
        assert_eq!(faults.tick(), (None, None), "one toast a burst");
        for n in 0..3 {
            faults.unrestored(&format!("far\\t_{n}.gltf"));
        }
        assert_eq!(faults.tick().0, Some((1, 4)));
        assert_eq!(
            faults.line().as_deref(),
            Some("Models failed to load: 1 broken model skipped, 4 models not loaded: holm_oak_b.gltf, t_0.gltf, t_1.gltf and 2 more (the log names each)")
        );
        assert_eq!(counted((0, 0)), "0 broken models skipped, 0 models not loaded");
    }
}
