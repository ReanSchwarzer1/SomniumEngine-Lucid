//! A scene's places: the spots someone wants to reach by name ("Lucara", "Vela Canyon, north rim").
//!
//! `<scene>.places.json` sits beside a scene and lists them. The editor's Command Palette (Ctrl+P) shows each as a
//! "Go to" row and takes the camera there. The tools that build a place add their own rows; a scene with no such file
//! has no rows.
//!
//! ```json
//! {"places": [{"name": "Monargent piazza", "group": "town", "x": -15482.0, "z": 331.5, "yaw": 315,
//!              "note": "the chapel steps", "built": true}]}
//! ```
//!
//! `x`, `z`: scene metres. `y`: the eye's height, for a place in the air; without it the eye stands
//! [`EYE_HEIGHT`] over the ground. `yaw`: degrees as a walking player counts them, 0 facing +z and 90 facing +x;
//! absent or null keeps the way the camera faces. `built: false` is a place planned and not there yet. Any other
//! field is the tools' own and is ignored. A row without a name, an `x` or a `z` is left out; the rest still show.

use serde::Deserialize;
use std::path::{Path, PathBuf};

/// The eye's height over the ground at a place that names no `y`, metres.
pub const EYE_HEIGHT: f32 = 1.7;

/// One place.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Place {
    /// Its name: the row's text.
    pub name: String,
    /// What kind of place it is ("town", "canyon"): the row's hint.
    #[serde(default)]
    pub group: String,
    /// Scene metres.
    pub x: f32,
    /// Scene metres.
    pub z: f32,
    /// The eye's height, for a place in the air.
    #[serde(default)]
    pub y: Option<f32>,
    /// Degrees as a walking player counts them: 0 faces +z, 90 faces +x.
    #[serde(default)]
    pub yaw: Option<f32>,
    /// What to look at there.
    #[serde(default)]
    pub note: String,
    /// False for a place that is planned and not in the world yet.
    #[serde(default = "yes")]
    pub built: bool,
}

fn yes() -> bool {
    true
}

impl Place {
    /// The editor camera's yaw for this place's (the camera counts 0 along +x and 90 along +z), or `None` to keep the
    /// camera's own.
    #[must_use]
    pub fn camera_yaw(&self) -> Option<f32> {
        self.yaw.map(|yaw| 90.0 - yaw)
    }
}

/// `<scene>.places.json`, beside the scene.
#[must_use]
pub fn places_path(scene: &Path) -> PathBuf {
    let mut name = scene.as_os_str().to_owned();
    name.push(".places.json");
    PathBuf::from(name)
}

/// The places beside `scene` and how many rows were left out as unreadable. No file is no places.
///
/// # Errors
/// A file that is there and is not a places list at all.
pub fn read(scene: &Path) -> Result<(Vec<Place>, usize), String> {
    let path = places_path(scene);
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok((Vec::new(), 0)),
        Err(error) => return Err(format!("{}: {error}", path.display())),
    };
    parse(&text).map_err(|error| format!("{}: {error}", path.display()))
}

/// Row by row: many hands add to this file, and one row with a mistyped number must not hide the others.
fn parse(text: &str) -> Result<(Vec<Place>, usize), String> {
    let document: serde_json::Value =
        serde_json::from_str(text.trim_start_matches('\u{feff}')).map_err(|error| error.to_string())?;
    let rows = document
        .get("places")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "it has no \"places\" list".to_owned())?;
    let places: Vec<Place> = rows
        .iter()
        .filter_map(|row| serde_json::from_value::<Place>(row.clone()).ok())
        .filter(|place| !place.name.is_empty() && place.x.is_finite() && place.z.is_finite())
        .collect();
    let left_out = rows.len() - places.len();
    Ok((places, left_out))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The file as Sunscar's sessions write it: their own fields are ignored, a null yaw keeps the camera's, a planned
    /// place says so, and a row with a mistyped number is left out without taking the others with it.
    #[test]
    fn a_places_file_is_read_row_by_row() {
        let text = r#"{"format": "sunscar-places-index", "version": 1, "status": "in use",
          "places": [
           {"name": "Monargent piazza (Play start)", "group": "town", "x": -15482.0, "z": 331.5, "yaw": 315,
            "note": "the chapel steps", "source": "notes/REPORTS.md:1437", "tried": true},
           {"name": "Vinehall", "group": "town", "x": -12825.6, "z": 2592.1, "yaw": null, "note": "the cellars",
            "tried": false, "gnomon": [-2225.4, 9392.1]},
           {"name": "The city", "group": "city site", "x": 3200, "z": 8400, "built": false},
           {"name": "A view from the air", "x": 0, "y": 800.0, "z": 0},
           {"name": "mistyped", "x": "east", "z": 3.0},
           {"group": "no name", "x": 1.0, "z": 2.0}]}"#;
        let (places, left_out) = parse(text).unwrap();
        assert_eq!(left_out, 2);
        assert_eq!(places.iter().map(|p| p.name.as_str()).collect::<Vec<_>>(), [
            "Monargent piazza (Play start)",
            "Vinehall",
            "The city",
            "A view from the air"
        ]);
        let piazza = &places[0];
        assert_eq!((piazza.x, piazza.z, piazza.group.as_str(), piazza.built), (-15482.0, 331.5, "town", true));
        // A player's yaw of 315 looks between +z and -x: the camera's -225.
        assert_eq!(piazza.camera_yaw(), Some(-225.0));
        assert_eq!(places[1].camera_yaw(), None);
        assert!(!places[2].built && places[2].note.is_empty());
        assert_eq!(places[3].y, Some(800.0));
        assert!(parse("[1, 2]").is_err() && parse("{").is_err());
    }

    #[test]
    fn a_scene_without_a_places_file_has_none() {
        let scene = std::env::temp_dir().join("somnium-no-such-scene-for-places.scene.json");
        assert_eq!(read(&scene), Ok((Vec::new(), 0)));
        assert!(places_path(&scene).to_string_lossy().ends_with(".scene.json.places.json"));
    }
}
