//! When the program stops on an error it leaves a report to send, and says where (ENGINE_ROADMAP_PRODUCTION items 5
//! and 35: TSF's game `src/crash_report.rs`, made the engine's, so Lucid's games have one without code of their own).
//!
//! - **A panic** writes `logs/crash-<unix seconds>.txt` under the project, or where that cannot be written,
//!   `%LOCALAPPDATA%/<program>/logs`: what failed and where, the thread, the program's build, the GPU and its driver,
//!   the system and a backtrace. A panic on the main thread ends the program, and a message box says where the report
//!   is. A worker's panic is caught and survived (`somnium_jobs`): it leaves a report and no box, at most
//!   [`MOST_REPORTS`] a run, since a job can fail every frame.
//! - **An error that stops [`crate::Engine::run`]** before its window opens (a scene that will not load, a palette
//!   entry out of range) gets the same box, with the error. Before, it reached only a console that closed with it.
//! - **The log** also goes to `logs/editor.log` (`logs/game.log` for a game alone) under the project, plain text, the
//!   two runs before kept as `.1.log` and `.2.log`, at most [`LOG_CAP`] bytes a run. `SOMNIUM_LOG_FILE=0` keeps it off.
//!
//! No box where nobody watches it: a background window, a capture run, CI, `SOMNIUM_NO_DIALOGS=1`, or a run whose
//! output a tool has taken (a pipe or a file). `SOMNIUM_CRASH_REPORT=0` leaves panics to a game's own hook.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, Ordering};

/// Most crash reports one run writes (a worker's panic can repeat every frame); the main thread's is always written.
const MOST_REPORTS: u32 = 8;
/// Most bytes one run writes to its log file; past it, one line says the log stops.
const LOG_CAP: u64 = 256 * 1024 * 1024;
/// Runs kept beside the current one: `editor.1.log`, `editor.2.log`.
const LOGS_KEPT: u32 = 2;

static INSTALLED: OnceLock<()> = OnceLock::new();
static REPORTS: AtomicU32 = AtomicU32::new(0);
/// This run's log file, for a report and a box to point at.
static LOG_PATH: OnceLock<PathBuf> = OnceLock::new();

/// A log file that stops at [`LOG_CAP`] bytes, with one line saying so.
pub(crate) struct CappedFile {
    file: std::fs::File,
    written: u64,
}

impl Write for CappedFile {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if self.written >= LOG_CAP {
            return Ok(bytes.len());
        }
        self.written += bytes.len() as u64;
        self.file.write_all(bytes)?;
        if self.written >= LOG_CAP {
            self.file.write_all(b"\n(the log stops here: this run reached its size cap)\n")?;
        }
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.file.flush()
    }
}

/// The file this run's log goes to: `<project>/logs/editor.log`, or `game.log` for a game alone (`player_mode`), the
/// runs before moved to `.1.log` and `.2.log`. `None` without a project, with `SOMNIUM_LOG_FILE=0`, or where the folder
/// cannot be written.
pub(crate) fn open_log(project: Option<&Path>, player_mode: bool) -> Option<CappedFile> {
    if std::env::var("SOMNIUM_LOG_FILE").as_deref() == Ok("0") {
        return None;
    }
    let (file, path) = open_log_in(&project?.join("logs"), if player_mode { "game" } else { "editor" })?;
    let _ = LOG_PATH.set(path);
    Some(file)
}

fn open_log_in(logs: &Path, stem: &str) -> Option<(CappedFile, PathBuf)> {
    std::fs::create_dir_all(logs).ok()?;
    let numbered = |n: u32| if n == 0 { logs.join(format!("{stem}.log")) } else { logs.join(format!("{stem}.{n}.log")) };
    for n in (1..=LOGS_KEPT).rev() {
        let _ = std::fs::rename(numbered(n - 1), numbered(n));
    }
    let path = numbered(0);
    let file = std::fs::File::create(&path).ok()?;
    Some((CappedFile { file, written: 0 }, path))
}

/// Install the panic hook, once, from the main thread: every panic leaves a report, and one on this thread a box.
pub(crate) fn install(program: &str, project: Option<&Path>) {
    if std::env::var("SOMNIUM_CRASH_REPORT").as_deref() == Ok("0") || INSTALLED.set(()).is_err() {
        return;
    }
    let program = program.to_owned();
    let folders = report_folders(&program, project);
    let main = std::thread::current().id();
    let usual = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        usual(info);
        let on_main = std::thread::current().id() == main;
        if REPORTS.fetch_add(1, Ordering::Relaxed) >= MOST_REPORTS && !on_main {
            return;
        }
        let payload = info.payload();
        let message = payload
            .downcast_ref::<&str>()
            .map(|s| (*s).to_owned())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "(no message)".to_owned());
        let at = info.location().map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column())).unwrap_or_default();
        let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
        let thread = std::thread::current();
        let facts = Facts {
            program: &program,
            secs,
            message: &message,
            at: &at,
            thread: thread.name().unwrap_or("unnamed"),
            on_main,
            build: build(),
            gpu: somnium_renderer::context::gpu_in_use().unwrap_or("none yet (it stopped before the GPU started)"),
            log: LOG_PATH.get().map_or_else(|| "none".to_owned(), |p| p.display().to_string()),
        };
        let text = facts.report(&std::backtrace::Backtrace::force_capture().to_string());
        let written = folders.iter().find_map(|dir| write_report(dir, secs, &text));
        let saved = written.as_ref().map_or_else(|| "its report could not be saved".to_owned(), |p| format!("report {}", p.display()));
        if on_main {
            tracing::error!("{program} stopped: {message} ({at}); {saved}");
            if watched() {
                tell(&program, &crash_text(&program, written.as_deref()));
            }
        } else {
            tracing::error!("a worker of {program} failed and was stopped; the program goes on: {message} ({at}); {saved}");
        }
    }));
}

/// An error stopped [`crate::Engine::run`]: the log has it, and a box says it where someone watches.
pub(crate) fn could_not_start(program: &str, error: &dyn std::fmt::Display) {
    tracing::error!("{program} could not start: {error}");
    if watched() {
        let log = LOG_PATH.get().map_or_else(String::new, |p| format!("\n\nThe log is in:\n{}", p.display()));
        tell(program, &format!("{program} could not start.\n\n{error}{log}"));
    }
}

/// Where reports go: the project's `logs`, then `%LOCALAPPDATA%/<program>/logs` (a game under Program Files cannot
/// write beside itself).
fn report_folders(program: &str, project: Option<&Path>) -> Vec<PathBuf> {
    let local = std::env::var_os("LOCALAPPDATA").map(|dir| PathBuf::from(dir).join(folder_name(program)).join("logs"));
    project.map(|p| p.join("logs")).into_iter().chain(local).collect()
}

/// A program's title as a folder name: what comes before " — " (the editor's "— Authoring"), without the characters a
/// Windows folder cannot hold.
fn folder_name(program: &str) -> String {
    let name: String = program
        .split(" — ")
        .next()
        .unwrap_or(program)
        .chars()
        .filter(|c| !matches!(c, '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*') && !c.is_control())
        .collect();
    let name = name.split_whitespace().collect::<Vec<_>>().join(" ");
    if name.is_empty() { "Somnium".to_owned() } else { name }
}

fn write_report(dir: &Path, secs: u64, text: &str) -> Option<PathBuf> {
    std::fs::create_dir_all(dir).ok()?;
    // Two panics in one second (a worker's, then the main thread's) keep both.
    let path = (0..16)
        .map(|n| if n == 0 { dir.join(format!("crash-{secs}.txt")) } else { dir.join(format!("crash-{secs}-{n}.txt")) })
        .find(|p| !p.exists())?;
    std::fs::write(&path, text).ok()?;
    Some(path)
}

/// The program file, its size and when it was built (modified): which build crashed.
fn build() -> String {
    let Ok(exe) = std::env::current_exe() else { return "unknown".to_owned() };
    let meta = std::fs::metadata(&exe).ok();
    let size = meta.as_ref().map_or_else(String::new, |m| format!(", {} bytes", m.len()));
    let built = meta
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or_else(String::new, |d| format!(", built {}", utc(d.as_secs())));
    format!("{}{size}{built}", exe.display())
}

/// What a report says.
struct Facts<'a> {
    program: &'a str,
    secs: u64,
    message: &'a str,
    at: &'a str,
    thread: &'a str,
    on_main: bool,
    build: String,
    gpu: &'a str,
    log: String,
}

impl Facts<'_> {
    fn report(&self, backtrace: &str) -> String {
        let threads = std::thread::available_parallelism().map_or(0, std::num::NonZeroUsize::get);
        format!(
            "{} stopped with an error.\n\
             When: {} ({} seconds since 1970)\n\
             What: {}\n\
             Where: {}\n\
             Thread: {} ({})\n\
             Program: {}\n\
             Engine: somnium_core {}\n\
             GPU: {}\n\
             System: {} {}, {threads} threads\n\
             Log: {}\n\n\
             {backtrace}\n",
            self.program,
            utc(self.secs),
            self.secs,
            self.message,
            self.at,
            self.thread,
            if self.on_main { "the main thread: the program stopped" } else { "a worker: the program went on" },
            self.build,
            env!("CARGO_PKG_VERSION"),
            self.gpu,
            std::env::consts::OS,
            std::env::consts::ARCH,
            self.log,
        )
    }
}

/// `secs` since 1970 as `YYYY-MM-DD HH:MM UTC` (Howard Hinnant's days-to-civil).
fn utc(secs: u64) -> String {
    let days = i64::try_from(secs / 86_400).unwrap_or(0);
    let (hour, minute) = ((secs % 86_400) / 3_600, (secs % 3_600) / 60);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02} {hour:02}:{minute:02} UTC")
}

fn crash_text(program: &str, report: Option<&Path>) -> String {
    match report {
        Some(path) => format!(
            "{program} stopped because of an error.\n\nA report was saved to:\n{}\n\nPlease send it with a note of what you \
             were doing when it happened.",
            path.display()
        ),
        None => format!("{program} stopped because of an error, and its report could not be saved."),
    }
}

/// Whether a person may see a box: not in a background window, a capture run or CI, not with `SOMNIUM_NO_DIALOGS=1`,
/// and not when a tool took the output.
fn watched() -> bool {
    quiet_runs_allow(|name| std::env::var(name).ok()) && output_is_seen()
}

/// The environment half of [`watched`], over `var` (the environment, or a test's).
fn quiet_runs_allow(var: impl Fn(&str) -> Option<String>) -> bool {
    let set = |name: &str| var(name).is_some_and(|v| !v.trim().is_empty());
    let quiet = ["SOMNIUM_BACKGROUND_WINDOW", "SOMNIUM_CAPTURE_PNG", "SOMNIUM_CAPTURE_UI_PNG", "SOMNIUM_CAPTURE_QUIT", "CI"]
        .iter()
        .any(|name| set(name));
    !quiet && var("SOMNIUM_NO_DIALOGS").as_deref() != Some("1")
}

/// Whether the program's standard error reaches a person: a console, or none at all (a windowed player's build, where
/// only a box is seen). A pipe or a file is a tool's.
#[cfg(windows)]
fn output_is_seen() -> bool {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetStdHandle(which: u32) -> *mut core::ffi::c_void;
        fn GetFileType(handle: *mut core::ffi::c_void) -> u32;
    }
    const STD_ERROR_HANDLE: u32 = -12i32 as u32;
    const FILE_TYPE_CHAR: u32 = 2;
    // SAFETY: plain Win32 queries of this process's own standard error handle; nothing is written or closed.
    unsafe {
        let handle = GetStdHandle(STD_ERROR_HANDLE);
        handle.is_null() || handle as isize == -1 || GetFileType(handle) == FILE_TYPE_CHAR
    }
}

#[cfg(not(windows))]
fn output_is_seen() -> bool {
    use std::io::IsTerminal;
    std::io::stderr().is_terminal()
}

fn tell(program: &str, text: &str) {
    let _ = rfd::MessageDialog::new()
        .set_level(rfd::MessageLevel::Error)
        .set_title(program)
        .set_description(text)
        .set_buttons(rfd::MessageButtons::Ok)
        .show();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_report_says_what_where_which_build_and_gpu() {
        let facts = Facts {
            program: "Rimeborn: Sunscar",
            secs: 1_700_000_000,
            message: "index out of bounds",
            at: "src/lib.rs:10:5",
            thread: "main",
            on_main: true,
            build: "F:/game/sunscar.exe, 57860096 bytes, built 2023-11-14 22:13 UTC".to_owned(),
            gpu: "NVIDIA GeForce RTX 4090 (Vulkan, NVIDIA 581.57)",
            log: "F:/game/logs/editor.log".to_owned(),
        };
        let text = facts.report("0: main");
        for part in [
            "Rimeborn: Sunscar stopped",
            "2023-11-14 22:13 UTC",
            "index out of bounds",
            "src/lib.rs:10:5",
            "the main thread: the program stopped",
            "sunscar.exe",
            "RTX 4090",
            "logs/editor.log",
            env!("CARGO_PKG_VERSION"),
            "0: main",
        ] {
            assert!(text.contains(part), "{part} missing from:\n{text}");
        }
        let worker = Facts { on_main: false, thread: "somnium-job-3", ..facts };
        assert!(worker.report("").contains("a worker: the program went on"));
    }

    #[test]
    fn utc_reads_as_a_calendar() {
        assert_eq!(utc(0), "1970-01-01 00:00 UTC");
        assert_eq!(utc(951_782_400), "2000-02-29 00:00 UTC", "a leap day");
        assert_eq!(utc(1_700_000_000), "2023-11-14 22:13 UTC");
        assert_eq!(utc(1_793_963_297), "2026-11-06 11:08 UTC");
    }

    #[test]
    fn a_box_waits_for_a_person_and_never_blocks_a_tool() {
        let env = |pairs: &'static [(&'static str, &'static str)]| {
            move |name: &str| pairs.iter().find(|(n, _)| *n == name).map(|(_, v)| (*v).to_owned())
        };
        assert!(quiet_runs_allow(env(&[])));
        assert!(quiet_runs_allow(env(&[("SOMNIUM_BACKGROUND_WINDOW", "")])), "Edit Sunscar.cmd clears it");
        assert!(!quiet_runs_allow(env(&[("SOMNIUM_BACKGROUND_WINDOW", "1600x900")])));
        assert!(!quiet_runs_allow(env(&[("SOMNIUM_CAPTURE_PNG", "E:/x.png")])));
        assert!(!quiet_runs_allow(env(&[("CI", "true")])));
        assert!(!quiet_runs_allow(env(&[("SOMNIUM_NO_DIALOGS", "1")])));
    }

    #[test]
    fn a_log_keeps_the_two_runs_before_and_a_report_never_overwrites_one() {
        let dir = std::env::temp_dir().join(format!("somnium-crash-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        for run in 0..4 {
            let (mut file, path) = open_log_in(&dir, "editor").expect("opened");
            writeln!(file, "run {run}").unwrap();
            assert_eq!(path, dir.join("editor.log"));
        }
        let read = |name: &str| std::fs::read_to_string(dir.join(name)).unwrap_or_default();
        assert_eq!((read("editor.log"), read("editor.1.log"), read("editor.2.log")), ("run 3\n".into(), "run 2\n".into(), "run 1\n".into()));
        assert!(!dir.join("editor.3.log").exists(), "two kept");
        let first = write_report(&dir, 42, "a").unwrap();
        let second = write_report(&dir, 42, "b").unwrap();
        assert_ne!(first, second);
        assert_eq!(std::fs::read_to_string(first).unwrap(), "a");
        assert_eq!(folder_name("Rimeborn: Sunscar — Authoring"), "Rimeborn Sunscar");
        assert_eq!(folder_name("??"), "Somnium");
        let _ = std::fs::remove_dir_all(dir);
    }
}
