//! Metaphor helpers: Help pages, content inventory, type icons (Phase 26-C/D).

use crate::icons::IconId;
use std::path::{Path, PathBuf};

pub const HELP_WELCOME: &str = include_str!("../../../docs/editor/welcome.md");
pub const HELP_VIEWPORT: &str = include_str!("../../../docs/editor/viewport.md");
pub const HELP_SHORTCUTS: &str = include_str!("../../../docs/editor/shortcuts.md");
pub const HELP_DRAWER: &str = include_str!("../../../docs/editor/content_drawer.md");
pub const HELP_ABOUT: &str = include_str!("../../../docs/editor/about.md");
pub const HELP_OUTLINER: &str = include_str!("../../../docs/editor/outliner.md");
pub const HELP_TERRAIN: &str = include_str!("../../../docs/editor/terrain.md");
pub const HELP_WATER: &str = include_str!("../../../docs/editor/water.md");
pub const HELP_LIGHTING: &str = include_str!("../../../docs/editor/lighting.md");
pub const HELP_MORROWIND: &str = include_str!("../../../docs/editor/morrowind.md");
pub const HELP_SCRIPTING: &str = include_str!("../../../docs/editor/scripting.md");
pub const HELP_STORY: &str = include_str!("../../../docs/editor/story.md");
pub const HELP_WHATS_NEW: &str = include_str!("../../../docs/editor/whats_new.md");

/// The id of the What's New page. Ids are stable; `help_order` says where each one sits in the list.
pub const WHATS_NEW_PAGE: u8 = 12;

pub fn help_page(id: u8) -> &'static str {
    match id {
        12 => HELP_WHATS_NEW,
        1 => HELP_VIEWPORT,
        2 => HELP_SHORTCUTS,
        3 => HELP_DRAWER,
        4 => HELP_ABOUT,
        5 => HELP_OUTLINER,
        6 => HELP_TERRAIN,
        7 => HELP_WATER,
        8 => HELP_LIGHTING,
        9 => HELP_SCRIPTING,
        10 => HELP_MORROWIND,
        11 => HELP_STORY,
        _ => HELP_WELCOME,
    }
}

pub fn help_titles() -> &'static [&'static str] {
    &[
        "Welcome",
        "Viewport",
        "Shortcuts",
        "Content Drawer",
        "About",
        "Outliner",
        "Terrain",
        "Water",
        "Lighting",
        "Scripting",
        "Designer Tools",
        "Story",
        "What's New",
    ]
}

/// The page ids in the order the overlay lists them: What's New first, then the rest as before.
pub fn help_order() -> &'static [u8] {
    &[WHATS_NEW_PAGE, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
}

/// FNV-1a, 64 bit, over the text with line ends normalised, so the same words give the same number on every
/// machine and every checkout.
pub fn text_hash(text: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.bytes().filter(|byte| *byte != b'\r') {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// The hash of the compiled-in What's New page, as it is stored.
pub fn whats_new_hash() -> String {
    format!("{:016x}", text_hash(HELP_WHATS_NEW))
}

/// Whether the person has not yet been told about this text. No stored value, or a different one, means not yet.
pub fn whats_new_unseen(stored: Option<&str>, current: &str) -> bool {
    stored != Some(current)
}

/// The hash held in a `whats_new_seen.json`, if the file reads.
pub fn parse_seen_hash(bytes: &[u8]) -> Option<String> {
    let value: serde_json::Value = serde_json::from_slice(bytes).ok()?;
    value.get("hash")?.as_str().map(str::to_owned)
}

/// The text of a `whats_new_seen.json` that holds this hash.
pub fn seen_hash_file(hash: &str) -> String {
    serde_json::json!({ "hash": hash }).to_string()
}

/// Where the seen hash lives: the editor's per-user folder.
pub fn whats_new_seen_path() -> Option<PathBuf> {
    std::env::var_os("APPDATA").map(|p| PathBuf::from(p).join("SomniumEngine/whats_new_seen.json"))
}

/// Write beside the file and rename over it, so a crash never leaves half a file.
pub fn write_seen_hash(file: &Path, hash: &str) -> std::io::Result<()> {
    if let Some(folder) = file.parent() {
        std::fs::create_dir_all(folder)?;
    }
    let partial = file.with_extension("json.partial");
    std::fs::write(&partial, seen_hash_file(hash))?;
    std::fs::rename(&partial, file)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HelpBlock {
    Heading(String),
    Paragraph(String),
    Bullet(String),
}

pub fn help_blocks(id: u8) -> Vec<HelpBlock> {
    parse_help_markdown(help_page(id))
}

fn strip_md(s: &str) -> String {
    s.replace("**", "")
}

pub fn parse_help_markdown(src: &str) -> Vec<HelpBlock> {
    let mut out = Vec::new();
    let mut para = String::new();
    let flush_para = |para: &mut String, out: &mut Vec<HelpBlock>| {
        let t = para.trim();
        if !t.is_empty() {
            out.push(HelpBlock::Paragraph(strip_md(t)));
        }
        para.clear();
    };
    let mut diagram = false;
    for line in src.lines() {
        let t = line.trim();
        if t == "```mermaid" {
            flush_para(&mut para, &mut out);
            diagram = true;
            continue;
        }
        if diagram {
            if t == "```" {
                diagram = false;
            }
            continue;
        }
        if t.is_empty() {
            flush_para(&mut para, &mut out);
        } else if let Some(rest) = t.strip_prefix("## ") {
            flush_para(&mut para, &mut out);
            out.push(HelpBlock::Heading(strip_md(rest)));
        } else if let Some(rest) = t.strip_prefix("# ") {
            flush_para(&mut para, &mut out);
            out.push(HelpBlock::Heading(strip_md(rest)));
        } else if let Some(rest) = t.strip_prefix("- ") {
            flush_para(&mut para, &mut out);
            out.push(HelpBlock::Bullet(strip_md(rest)));
        } else {
            if !para.is_empty() {
                para.push(' ');
            }
            para.push_str(t);
        }
    }
    flush_para(&mut para, &mut out);
    out
}

#[derive(Clone, Debug)]
pub struct ContentEntry {
    pub path: PathBuf,
    pub name: String,
    pub is_dir: bool,
    pub icon: IconId,
    pub is_engine: bool,
    /// Immutable database identity; absent only for virtual Engine entries.
    pub asset_id: Option<somnium_asset::database::AssetId>,
    /// Precomputed facts shown by the drawer tooltip.
    pub tooltip: String,
}

impl From<somnium_asset::database::AssetRecord> for ContentEntry {
    fn from(record: somnium_asset::database::AssetRecord) -> Self {
        let is_dir = record.kind == somnium_asset::database::AssetKind::Folder;
        let tooltip = record.tooltip();
        Self {
            icon: icon_for_path(&record.absolute_path, is_dir),
            path: record.absolute_path,
            name: record.name,
            is_dir,
            is_engine: false,
            asset_id: Some(record.id),
            tooltip,
        }
    }
}

pub fn icon_for_path(path: &Path, is_dir: bool) -> IconId {
    if is_dir {
        return IconId::Folder;
    }
    match path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
        .as_str()
    {
        "gltf" | "glb" => IconId::Mesh,
        "png" | "jpg" | "jpeg" | "exr" | "hdr" | "tga" => IconId::Texture,
        "json" => IconId::Json,
        "wgsl" | "hlsl" | "glsl" => IconId::Shader,
        "ttf" | "otf" => IconId::Font,
        "wav" | "ogg" | "mp3" => IconId::Audio,
        "md" => IconId::License,
        // Phase 16-D: a `.luau` file is content, and the drawer is where an
        // author finds it. `.rs` was already here; `.luau` is the one they
        // will actually double-click.
        "rs" | "luau" | "lua" => IconId::Script,
        "somnium" => IconId::Scene,
        _ => IconId::Unknown,
    }
}

/// Recursively list `assets/` (project /Game) plus optional virtual /Engine.
pub fn list_content(
    root: &Path,
    show_engine: bool,
    filter: &str,
    current: &Path,
) -> Vec<ContentEntry> {
    let mut out = Vec::new();
    let filter = filter.to_ascii_lowercase();
    let walk = if current.as_os_str().is_empty() {
        root.to_path_buf()
    } else {
        current.to_path_buf()
    };
    if let Ok(rd) = std::fs::read_dir(&walk) {
        let mut ents: Vec<_> = rd.flatten().collect();
        ents.sort_by_key(|e| {
            (
                !e.path().is_dir(),
                e.file_name().to_string_lossy().to_ascii_lowercase(),
            )
        });
        for e in ents {
            let path = e.path();
            if path.file_name().and_then(|n| n.to_str()) == Some("bc7") {
                continue;
            }
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            if name.starts_with('.') {
                continue;
            }
            if !filter.is_empty() && !name.to_ascii_lowercase().contains(&filter) && !path.is_dir()
            {
                continue;
            }
            let is_dir = path.is_dir();
            out.push(ContentEntry {
                icon: icon_for_path(&path, is_dir),
                is_dir,
                is_engine: false,
                asset_id: None,
                tooltip: String::new(),
                name,
                path,
            });
        }
    }
    if show_engine && (current.as_os_str().is_empty() || current == root) {
        out.extend(virtual_engine_content(&filter));
    }
    out
}

/// Built-in entries are virtual and never require a filesystem query.
pub fn virtual_engine_content(filter: &str) -> Vec<ContentEntry> {
    let filter = filter.to_ascii_lowercase();
    let mut out = Vec::new();
    for (name, icon) in [
        ("Cube", IconId::Cube),
        ("Sphere", IconId::Sphere),
        ("Plane", IconId::Plane),
        ("Cylinder", IconId::Cylinder),
        ("Inter-Regular.ttf", IconId::Font),
    ] {
        if filter.is_empty() || name.to_ascii_lowercase().contains(&filter) {
            out.push(ContentEntry {
                path: PathBuf::from("/Engine").join(name),
                name: name.into(),
                is_dir: false,
                icon,
                is_engine: true,
                asset_id: None,
                tooltip: "Built-in Engine asset".into(),
            });
        }
    }
    out
}

pub fn icon_for_entity_name(name: &str) -> IconId {
    let n = name.to_ascii_lowercase();
    if n.contains("light") || n.contains("sun") {
        IconId::DirectionalLight
    } else if n.contains("terrain") && n.contains("voxel") {
        IconId::VoxelTerrain
    } else if n.contains("terrain") {
        IconId::Terrain
    } else if n.contains("water") {
        IconId::Water
    } else if n.contains("boat") || n.contains("vessel") {
        IconId::Vessel
    } else if n.contains("post") {
        IconId::PostFx
    } else if n.contains("particle") {
        IconId::Particle
    } else if n.contains("camera") {
        IconId::Camera
    } else {
        IconId::Cube
    }
}

#[allow(dead_code)]
pub fn create_icon(kind: crate::editor_event::CreateKind) -> IconId {
    use crate::editor_event::CreateKind::*;
    match kind {
        Cube | Blockout => IconId::Cube,
        Sphere => IconId::Sphere,
        Plane => IconId::Plane,
        Cylinder => IconId::Cylinder,
        DirectionalLight => IconId::DirectionalLight,
        PointLight => IconId::PointLight,
        SpotLight => IconId::SpotLight,
        RectLight => IconId::PointLight,
        DiscLight => IconId::PointLight,
        TubeLight => IconId::SpotLight,
        AudioEmitter | ShorelineAudio => IconId::Audio,
        Spline => IconId::Mesh,
        Particle => IconId::Particle,
        Terrain | EmptyTerrain => IconId::Terrain,
        VoxelTerrain => IconId::VoxelTerrain,
        Lake | Ocean | Sea | River => IconId::Vessel,
        UiCanvas => IconId::Window,
        // CONTROL-L: the environment is the sun, so it takes the sun icon.
        Environment => IconId::DirectionalLight,
    }
}

pub fn tonemap_index(label: &str) -> usize {
    match label {
        "ACES" => 1,
        "Reinhard" => 2,
        "Photographic" => 3,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    #[test]
    fn help_pages_are_nonempty() {
        assert!(HELP_WELCOME.contains("Somnium"));
        assert!(HELP_SHORTCUTS.contains("Ctrl+Space"));
        assert_eq!(help_titles().len(), 13);
        assert_eq!(help_page(WHATS_NEW_PAGE), HELP_WHATS_NEW);
        assert_eq!(help_titles()[WHATS_NEW_PAGE as usize], "What's New");
        assert_eq!(help_page(11), HELP_STORY);
        assert!(HELP_STORY.contains("Game View") && HELP_STORY.contains("game metres"));
        assert_eq!(help_page(10), HELP_MORROWIND);
        assert!(
            help_blocks(10)
                .iter()
                .all(|block| !format!("{block:?}").contains("flowchart LR"))
        );
        assert_eq!(help_page(4), HELP_ABOUT);
        assert_eq!(help_page(9), HELP_SCRIPTING);
        // Phase 16-D: the two rules an author most needs and is most
        // likely to be surprised by.
        assert!(HELP_SCRIPTING.contains("visibility rule"));
        assert!(HELP_SCRIPTING.contains("only while Play is running"));
        assert!(HELP_WATER.contains("RT Reflect"));
        assert!(HELP_WATER.contains("RT Refraction"));
        // TSUSHIMA-J: both pages document a refusal a person will otherwise
        // read as a broken control. A brush that places nothing and a bar that
        // hides a group are the two things most recently reported as bugs.
        assert!(HELP_TERRAIN.contains("Min layer"));
        assert!(HELP_TERRAIN.contains("Foliage card"));
        assert!(HELP_VIEWPORT.contains("When the bar runs out of room"));
        assert!(HELP_LIGHTING.contains("Disc Light"));
        assert!(HELP_LIGHTING.contains("Tube Light"));
        assert!(HELP_LIGHTING.contains("4×4×4"));
        let blocks = parse_help_markdown("# Title\n\nHello **world**.\n- item\n");
        assert_eq!(
            blocks,
            vec![
                HelpBlock::Heading("Title".into()),
                HelpBlock::Paragraph("Hello world.".into()),
                HelpBlock::Bullet("item".into()),
            ]
        );
    }

    #[test]
    fn whats_new_is_the_first_page_and_reads_as_blocks() {
        let order = help_order();
        assert_eq!(order.len(), help_titles().len());
        assert_eq!(order[0], WHATS_NEW_PAGE);
        let mut sorted = order.to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..help_titles().len() as u8).collect::<Vec<_>>());
        let blocks = help_blocks(WHATS_NEW_PAGE);
        assert_eq!(blocks[0], HelpBlock::Heading("What's new in the editor".into()));
        assert!(blocks.iter().any(|block| matches!(block, HelpBlock::Bullet(_))));
        assert!(!HELP_WHATS_NEW.contains('!'), "no exclamation marks");
    }

    #[test]
    fn the_text_hash_is_stable_and_ignores_line_ends() {
        // FNV-1a 64 of the empty text and of "a" are published values.
        assert_eq!(text_hash(""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(text_hash("a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(text_hash("one\r\ntwo\r\n"), text_hash("one\ntwo\n"));
        assert_ne!(text_hash("one"), text_hash("two"));
        assert_eq!(whats_new_hash().len(), 16);
        assert_eq!(whats_new_hash(), format!("{:016x}", text_hash(HELP_WHATS_NEW)));
    }

    #[test]
    fn the_notice_shows_on_first_run_and_after_a_change_only() {
        let now = whats_new_hash();
        assert!(whats_new_unseen(None, &now), "no stored file is a first run");
        assert!(whats_new_unseen(Some("0000000000000000"), &now));
        assert!(!whats_new_unseen(Some(&now), &now));
    }

    #[test]
    fn the_seen_file_round_trips_and_a_bad_one_reads_as_unseen() {
        let hash = whats_new_hash();
        assert_eq!(parse_seen_hash(seen_hash_file(&hash).as_bytes()), Some(hash.clone()));
        assert_eq!(parse_seen_hash(b"not json"), None);
        assert_eq!(parse_seen_hash(b"{}"), None);
        let folder = std::env::temp_dir().join(format!("somnium_whats_new_{}", std::process::id()));
        let _ = fs::remove_dir_all(&folder);
        let file = folder.join("nested").join("whats_new_seen.json");
        write_seen_hash(&file, "aaaa").unwrap();
        write_seen_hash(&file, &hash).unwrap();
        assert_eq!(parse_seen_hash(&fs::read(&file).unwrap()), Some(hash));
        assert!(!file.with_extension("json.partial").exists());
        let _ = fs::remove_dir_all(&folder);
    }

    #[test]
    fn bc7_and_dotfiles_are_hidden() {
        let tmp = std::env::temp_dir().join(format!("somnium_content_{}", std::process::id()));
        let assets = tmp.join("assets");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(assets.join("meshes")).unwrap();
        fs::create_dir_all(assets.join("terrain").join("bc7")).unwrap();
        fs::write(assets.join("meshes").join("cube.gltf"), "x").unwrap();
        fs::write(assets.join(".hidden"), "x").unwrap();
        fs::write(assets.join("terrain").join("bc7").join("pack.bin"), "x").unwrap();
        let listed = list_content(&assets, false, "", &PathBuf::new());
        assert!(listed.iter().any(|e| e.name == "meshes"));
        assert!(
            listed
                .iter()
                .all(|e| e.name != "bc7" && !e.name.starts_with('.'))
        );
        let nested = list_content(&assets, false, "", &assets.join("terrain"));
        assert!(nested.iter().all(|e| e.name != "bc7"));
        let engine = list_content(&assets, true, "cube", &PathBuf::new());
        assert!(engine.iter().any(|e| e.is_engine && e.name == "Cube"));
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn icon_for_gltf_is_mesh() {
        assert_eq!(
            icon_for_path(std::path::Path::new("foo.gltf"), false),
            IconId::Mesh
        );
        assert_eq!(
            icon_for_entity_name("Directional Light"),
            IconId::DirectionalLight
        );
        assert_eq!(tonemap_index("ACES"), 1);
    }
}

/// Phase 27-G. The empty state a panel shows instead of a blank rectangle.
///
/// Every field is required on purpose. `phase_27.md` §9.7-4 asks for a mark, one
/// sentence and one action, and §13 (Plain Speech) rules out "Oops" and the
/// exclamation mark. A panel that has nothing to show still has to say what
/// *would* be here and what to do about it, or the user cannot tell an empty
/// list from a broken one.
#[derive(Clone, Copy, Debug)]
pub struct EmptyState {
    pub icon: IconId,
    /// What would be here. Title Case, no trailing period.
    pub headline: &'static str,
    /// One sentence, sentence case, ending in a period.
    pub body: &'static str,
    /// The single action that fixes it. Sentence case, no period.
    pub action: &'static str,
}

/// The shipped empty states, one per panel that can be empty.
pub mod empty {
    use super::{EmptyState, IconId};

    pub const OUTLINER: EmptyState = EmptyState {
        icon: IconId::Scene,
        headline: "No Entities",
        body: "This scene is empty. Add something to start building.",
        action: "Use Create in the toolbar",
    };

    pub const DETAILS: EmptyState = EmptyState {
        icon: IconId::Select,
        headline: "No Selection",
        body: "Select an entity in the viewport or the Outliner to edit it.",
        action: "Click an entity",
    };

    pub const CONTENT: EmptyState = EmptyState {
        icon: IconId::Folder,
        headline: "Nothing Here",
        body: "This folder has no assets yet.",
        action: "Import a model from the File menu",
    };

    /// Distinct from `CONTENT`: a filter that matches nothing is a different
    /// situation from a folder that is genuinely empty, and offering "import a
    /// model" to someone who mistyped a search would be wrong.
    pub const CONTENT_FILTERED: EmptyState = EmptyState {
        icon: IconId::Search,
        headline: "No Matches",
        body: "No asset in this folder matches the filter.",
        action: "Clear the search box",
    };

    pub const LOG: EmptyState = EmptyState {
        icon: IconId::OutputLog,
        headline: "Nothing Logged",
        body: "Script output and import messages appear here.",
        action: "Press Play to run the scene",
    };
}

/// A type filter chip in the Content Browser.
///
/// Phase 27-G. Deliberately a closed set mapped onto the icon the entry already
/// resolves to, so a chip cannot disagree with the tile it filters: both answer
/// "what kind of thing is this" from `icon_for_path`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContentFilterKind {
    All,
    Folders,
    Models,
    Textures,
    Scripts,
    Scenes,
    Audio,
    Materials,
}

impl ContentFilterKind {
    pub const ALL: [ContentFilterKind; 8] = [
        ContentFilterKind::All,
        ContentFilterKind::Folders,
        ContentFilterKind::Models,
        ContentFilterKind::Materials,
        ContentFilterKind::Textures,
        ContentFilterKind::Scripts,
        ContentFilterKind::Scenes,
        ContentFilterKind::Audio,
    ];

    /// The chip's glyph (Phase ALYX-F).
    pub const fn icon(self) -> IconId {
        match self {
            ContentFilterKind::All => IconId::GridView,
            ContentFilterKind::Folders => IconId::Folder,
            ContentFilterKind::Models => IconId::Mesh,
            ContentFilterKind::Materials => IconId::Material,
            ContentFilterKind::Textures => IconId::Texture,
            ContentFilterKind::Scripts => IconId::Script,
            ContentFilterKind::Scenes => IconId::Scene,
            ContentFilterKind::Audio => IconId::Audio,
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            ContentFilterKind::All => "All",
            ContentFilterKind::Folders => "Folders",
            ContentFilterKind::Models => "Models",
            ContentFilterKind::Textures => "Textures",
            ContentFilterKind::Scripts => "Scripts",
            ContentFilterKind::Scenes => "Scenes",
            ContentFilterKind::Audio => "Audio",
            ContentFilterKind::Materials => "Materials",
        }
    }

    /// Whether an entry belongs in this chip.
    pub fn accepts(self, entry: &ContentEntry) -> bool {
        use IconId as I;
        match self {
            ContentFilterKind::All => true,
            ContentFilterKind::Folders => entry.is_dir,
            _ if entry.is_dir => false,
            ContentFilterKind::Models => entry.icon == I::Mesh,
            ContentFilterKind::Textures => entry.icon == I::Texture,
            ContentFilterKind::Scripts => entry.icon == I::Script,
            ContentFilterKind::Scenes => entry.icon == I::Scene,
            ContentFilterKind::Audio => entry.icon == I::Audio,
            ContentFilterKind::Materials => entry.icon == I::Material,
        }
    }
}

/// Tile size in the Content Browser.
///
/// Three steps rather than a continuous slider: the icon atlas has two cuts, so
/// a free-form size would spend most of its range resampling one of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContentDensity {
    Compact,
    Comfortable,
    Large,
}

impl ContentDensity {
    pub const ALL: [ContentDensity; 3] = [
        ContentDensity::Compact,
        ContentDensity::Comfortable,
        ContentDensity::Large,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            ContentDensity::Compact => "Compact",
            ContentDensity::Comfortable => "Comfortable",
            ContentDensity::Large => "Large",
        }
    }

    /// (tile width, tile height, icon size).
    pub const fn metrics(self) -> (f32, f32, f32) {
        match self {
            // Below the 40 px large-cut threshold, so these sample the 32 px cut.
            ContentDensity::Compact => (104.0, 84.0, 32.0),
            ContentDensity::Comfortable => (136.0, 112.0, 56.0),
            ContentDensity::Large => (160.0, 172.0, 120.0),
        }
    }

    pub fn next(self) -> Self {
        match self {
            ContentDensity::Compact => ContentDensity::Comfortable,
            ContentDensity::Comfortable => ContentDensity::Large,
            ContentDensity::Large => ContentDensity::Compact,
        }
    }
}

/// Back/forward history for the Content Browser.
///
/// Standard browser semantics, which are easy to get subtly wrong: navigating
/// somewhere new **truncates** the forward stack, and going back then forward
/// must land exactly where you started.
#[derive(Clone, Debug, Default)]
pub struct ContentHistory {
    entries: Vec<String>,
    cursor: usize,
}

impl ContentHistory {
    pub fn new(root: String) -> Self {
        Self {
            entries: vec![root],
            cursor: 0,
        }
    }

    pub fn current(&self) -> &str {
        self.entries
            .get(self.cursor)
            .map(|s| s.as_str())
            .unwrap_or("")
    }

    /// Navigate somewhere new. A no-op when it is where we already are, so
    /// re-clicking the current folder does not stack duplicate entries.
    pub fn push(&mut self, path: String) {
        if self.current() == path {
            return;
        }
        self.entries.truncate(self.cursor + 1);
        self.entries.push(path);
        self.cursor = self.entries.len() - 1;
    }

    pub fn can_go_back(&self) -> bool {
        self.cursor > 0
    }

    pub fn can_go_forward(&self) -> bool {
        self.cursor + 1 < self.entries.len()
    }

    pub fn back(&mut self) -> Option<&str> {
        if !self.can_go_back() {
            return None;
        }
        self.cursor -= 1;
        Some(self.current())
    }

    pub fn forward(&mut self) -> Option<&str> {
        if !self.can_go_forward() {
            return None;
        }
        self.cursor += 1;
        Some(self.current())
    }
}

#[cfg(test)]
mod browser_workflow_tests {
    use super::*;
    use std::path::PathBuf;

    fn entry(name: &str, is_dir: bool) -> ContentEntry {
        let path = PathBuf::from(name);
        ContentEntry {
            icon: icon_for_path(&path, is_dir),
            is_dir,
            is_engine: false,
            asset_id: None,
            tooltip: String::new(),
            name: name.to_string(),
            path,
        }
    }

    #[test]
    fn the_all_chip_accepts_everything_and_folders_only_folders() {
        let items = [
            entry("models", true),
            entry("ship.glb", false),
            entry("rock.png", false),
            entry("boot.luau", false),
        ];
        for e in &items {
            assert!(ContentFilterKind::All.accepts(e), "All rejected {}", e.name);
        }
        let folders: Vec<_> = items
            .iter()
            .filter(|e| ContentFilterKind::Folders.accepts(e))
            .collect();
        assert_eq!(folders.len(), 1);
        assert_eq!(folders[0].name, "models");
    }

    #[test]
    fn a_type_chip_never_matches_a_folder() {
        // A folder called "textures" must not appear under the Textures chip.
        let dir = entry("textures", true);
        for kind in ContentFilterKind::ALL {
            if matches!(kind, ContentFilterKind::All | ContentFilterKind::Folders) {
                continue;
            }
            assert!(!kind.accepts(&dir), "{kind:?} matched a folder");
        }
    }

    #[test]
    fn type_chips_route_by_the_same_answer_the_tile_shows() {
        // The chip and the tile icon must never disagree.
        for (name, kind) in [
            ("ship.glb", ContentFilterKind::Models),
            ("rock.png", ContentFilterKind::Textures),
            ("boot.luau", ContentFilterKind::Scripts),
            ("level.somnium", ContentFilterKind::Scenes),
            ("hit.wav", ContentFilterKind::Audio),
        ] {
            let e = entry(name, false);
            assert!(kind.accepts(&e), "{name} did not match {kind:?}");
            for other in ContentFilterKind::ALL {
                if other == kind || other == ContentFilterKind::All {
                    continue;
                }
                assert!(!other.accepts(&e), "{name} also matched {other:?}");
            }
        }
    }

    #[test]
    fn density_steps_stay_ordered_and_cycle() {
        let mut seen = ContentDensity::Compact;
        let mut sizes = Vec::new();
        for _ in 0..3 {
            sizes.push(seen.metrics());
            seen = seen.next();
        }
        assert_eq!(seen, ContentDensity::Compact, "next() must cycle");
        assert!(
            sizes[0].0 < sizes[1].0 && sizes[1].0 < sizes[2].0,
            "widths ascend"
        );
        assert!(
            sizes[0].2 < sizes[1].2 && sizes[1].2 < sizes[2].2,
            "icons ascend"
        );
    }

    #[test]
    fn compact_tiles_stay_under_the_large_icon_cut() {
        // Above 40 px a glyph samples the 96 px cut; a compact tile should not,
        // or the atlas does three times the work for a smaller picture.
        assert!(ContentDensity::Compact.metrics().2 <= 40.0);
    }

    #[test]
    fn history_moves_back_and_forward_to_the_same_places() {
        let mut h = ContentHistory::new(String::new());
        h.push("models".into());
        h.push("models/ships".into());
        assert_eq!(h.current(), "models/ships");

        assert_eq!(h.back(), Some("models"));
        assert_eq!(h.back(), Some(""));
        assert!(!h.can_go_back(), "the root is the end of the line");
        assert_eq!(h.back(), None);

        assert_eq!(h.forward(), Some("models"));
        assert_eq!(h.forward(), Some("models/ships"));
        assert!(!h.can_go_forward());
        assert_eq!(h.forward(), None);
    }

    #[test]
    fn navigating_somewhere_new_truncates_the_forward_stack() {
        // The classic browser-history mistake: going back, then somewhere else,
        // must discard what was ahead rather than leaving it reachable.
        let mut h = ContentHistory::new(String::new());
        h.push("models".into());
        h.push("models/ships".into());
        h.back();
        h.push("terrain".into());
        assert!(!h.can_go_forward(), "models/ships must no longer be ahead");
        assert_eq!(h.current(), "terrain");

        // The trail behind is root -> models -> terrain: the branch that was
        // discarded is gone, but everything before the branch point remains.
        assert_eq!(h.back(), Some("models"));
        assert_eq!(h.back(), Some(""));
        assert!(!h.can_go_back());
    }

    #[test]
    fn re_entering_the_current_folder_does_not_stack_duplicates() {
        let mut h = ContentHistory::new(String::new());
        h.push("models".into());
        h.push("models".into());
        h.back();
        assert_eq!(h.current(), "", "one back should reach the root");
    }
}
