# What's new in the editor

Newest first. Each line is something you can do now that you could not do before.

## 28 November 2026

- **Shift+D makes a copy and puts it in your hand.** Select a thing, press Shift+D, and its copy follows the pointer over the ground, just as G does. Click or Enter puts the copy down. Esc removes it. The copy is one undo step.
- **Shift+D again makes the next copy.** With a copy in your hand, Shift+D puts it down and hands you another, so a row of crates is Shift+D, move, Shift+D, move.
- **Copies are numbered:** a copy of "crate" is "crate_copy", the next "crate_copy2", then "crate_copy3". Before, a copy of a copy was "crate_copy_copy".
- **Undo after Duplicate (Ctrl+D) says "Undid: Duplicate".** Before, it said "Undid: Paste".

## 27 November 2026, evening

- **G moves a thing over the ground.** Select it, press G, and it follows the pointer over whatever lies under it: the ground, a roof, a rock. Click or Enter puts it down. Esc puts it back where it was. The whole move is one undo step.
- **While you carry it:** hold Shift and it tilts to the slope. Turn the mouse wheel and it turns, one snap step a notch. Hold Ctrl and its place snaps to the grid (or stops snapping, if Snap is already on).
- **G carries several things at once:** select them, then press G.
- **The Content Drawer shows its folders at once.** Before, it said "Nothing Here" until the whole project had been read, which takes minutes in a large world. Now the folders come straight from the disk, and the line under them says so while the project is still being read. Search and previews follow when the read is done.
- **A read that fails is said.** The line under the folders says why, and the editor tries again by itself.
- **The editor opens each model and picture once.** It keeps what it found, so later reads of the project do not open them again.
- **Right-click a folder > Open opens that folder.** Before, it showed "Nothing Here".
- **Align to Surface (Shift+End) tilts a thing to a hillside.** Before, it tilted to a roof or a rock and never to the ground itself.
- **A key you move in Preferences > Keyboard is moved everywhere.** Before, the view still ran the old key as well: Scale moved to another key went on scaling on S.

## 27 November 2026, afternoon

- **Saving is honest.** Ctrl+S says how many objects and removed plants it saved. If you sculpted or painted ground, it says that the ground is not kept and the scene stays marked unsaved until Save Scene As.
- **Your last twenty saves are kept** beside the scene, in the folder .somnium_backups, so a bad save can be taken back.
- **Save before New or Open stops if the save did not work.** The scene stays open instead of being thrown away.
- **Undo and Redo say what they did,** for example "Undid: Drop to Ground". Ctrl+Shift+Z redoes, as well as Ctrl+Y.
- **F frames a thing by its size:** a house from afar, a pebble up close.
- **Ctrl+click adds or removes a thing from the selection in the view. Shift+click adds.** Either on empty ground keeps the selection.
- **Paste (Ctrl+V) puts the copy where the original stands,** selected, instead of inside whatever was selected. Duplicate (Ctrl+D) says what it copied.
- **The ground layers have their real names.** With a terrain selected, the paint tools list the layers by the names of that region's own set.
- **Help > What's New** is this page. **Help > This Project's Guide** shows the project's own guide.

## 27 November 2026, morning

- **Search boxes keep every key.** Typing in Ctrl+P, the Outliner search and the other search boxes no longer loses letters such as t, r and s, and no longer triggers shortcuts.
- **The game's weather is in Ctrl+P.** Type "weather" to list the game's weather states and events and pick one.
- **GPU Grass has a switch:** View > Pipeline > GPU Grass.
- **Esc no longer closes the editor.**
- **Move to Recycle Bin asks first** in the Content Drawer.
- **A world that cannot read its cells says so,** with a message and a line in the Output Log that names the file.
- **A menu item that cannot be used says why** when you click it.
