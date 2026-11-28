# When something goes wrong: logs, crash reports and broken models

Lucid only. This page covers what the engine leaves behind when it stops, and how it treats a model it cannot load.

## Log files

The log goes to `logs/editor.log` under the project, or `logs/game.log` for a game run on its own. It is plain text.
The two runs before are kept as `.1.log` and `.2.log`, and one run writes at most 256 MB. `SOMNIUM_LOG_FILE=0` keeps
the file off.

## Crash reports

- **A panic** writes `logs/crash-<unix seconds>.txt` under the project (or, where that cannot be written,
  `%LOCALAPPDATA%/<program>/logs`): what failed and where, the thread, the program's build, the GPU and its driver,
  the system and a backtrace. A panic on the main thread ends the program, and a message box says where the report
  is. A worker thread's panic is caught and survived: it leaves a report and no box, at most eight a run.
- **An error that stops the engine before its window opens** (a scene that will not load, for example) gets the same
  box, with the error.
- **A native crash** (an access violation in a driver, in Jolt's C++ or in Luau, which never reaches Rust's panic
  handling) writes `logs/crash-<secs>.dmp`, a minidump for a debugger, and `logs/crash-<secs>.txt` with the exception,
  the module and offset it happened in, the build and the GPU. Then the same box. The report is written in process, so
  a crash that wrecked the heap may leave less.

No box is shown where nobody would see it: a background window, a capture run, CI, `SOMNIUM_NO_DIALOGS=1`, or a run
whose output a tool has taken. `SOMNIUM_CRASH_REPORT=0` leaves panics and native crashes to the game.

## Broken models

Every glTF is checked before anything reaches the GPU: each buffer's size against its `byteLength`, buffer views and
accessors that stay in bounds, indices that stay inside the vertices, and images that decode. A model that fails is
refused: the log says so once and a toast appears.

The models that did not load in a run (the files the importer refused, and the files a scene names that are missing or
unreadable) are counted in three places, so a world with pieces missing is never taken for a whole one: the editor's
status bar (with a toast that stays), the game's screen, and the run's last log line.

**Level > Check Meshes** checks the surface of every model the loaded objects draw: closed, open only at its base
(fine where it sits in the ground), or open on a side or top (hollow from behind). Rocks and cliffs come first; the
list is written beside the scene.

To check files before you commit them, use the `gltf_check` example. It reads each file as the engine's importer
does; `--full` loads each file through the importer itself, images and all (slower).

```sh
cargo run -p somnium_asset --example gltf_check -- [--full] <file or folder>...
```

It prints one `BROKEN <file>: <why>` line per broken file, then a count, and exits with 1 when any file is broken.
