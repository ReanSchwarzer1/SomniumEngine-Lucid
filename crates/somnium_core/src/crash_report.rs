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
//! - **A native crash** (an access violation in a driver, Jolt's C++ or Luau, which never reaches Rust's panic hook and
//!   left nothing) writes `logs/crash-<secs>.dmp`, a minidump with every thread's stack for a debugger and the program's
//!   PDB, and `logs/crash-<secs>.txt`: the exception in words, the module and offset it happened in, the build, the GPU.
//!   Then the same box, and Windows' own handling. In process, so a crash that wrecked the heap may leave less.
//!
//! No box where nobody watches it: a background window, a capture run, CI, `SOMNIUM_NO_DIALOGS=1`, or a run whose
//! output a tool has taken (a pipe or a file). `SOMNIUM_CRASH_REPORT=0` leaves panics and native crashes to the game.

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
/// The program's name and its report folders, for a native crash's filter (which can take no arguments).
static NATIVE: OnceLock<(String, Vec<PathBuf>)> = OnceLock::new();
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
    let _ = NATIVE.set((program.clone(), folders.clone()));
    native::install();
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

/// A native exception's code in words.
fn exception_name(code: u32) -> &'static str {
    match code {
        0xC000_0005 => "an access violation (memory the program does not own)",
        0xC000_0006 => "a page that could not be read in (a file or a drive went away)",
        0xC000_001D => "an illegal instruction",
        0xC000_0094 => "an integer divided by zero",
        0xC000_0095 => "an integer overflow",
        0xC000_00FD => "a stack overflow",
        0xC000_0374 => "heap corruption",
        0xC000_0409 => "a stack buffer overrun (a fast fail)",
        0x8000_0003 => "a breakpoint",
        _ => "a native exception",
    }
}

/// What a native crash's report says.
struct NativeFacts<'a> {
    program: &'a str,
    secs: u64,
    code: u32,
    /// Where it happened: the module's file, and the offset into it.
    at: String,
    /// For an access violation: read, write or execute, and the address it touched.
    touched: Option<(usize, usize)>,
    thread: u32,
    build: String,
    gpu: &'a str,
    log: String,
    dump: Option<&'a Path>,
}

impl NativeFacts<'_> {
    fn report(&self) -> String {
        let touched = self.touched.map_or_else(String::new, |(kind, address)| {
            let how = match kind {
                0 => "reading",
                1 => "writing",
                8 => "running code at",
                _ => "touching",
            };
            format!(" ({how} {address:#x})")
        });
        let dump = self.dump.map_or_else(
            || "none: it could not be written".to_owned(),
            |p| format!("{} (open it in a debugger with the program's PDB beside it)", p.display()),
        );
        format!(
            "{} stopped with a native crash.\n\
             When: {} ({} seconds since 1970)\n\
             What: {}{touched}, code {:#010x}\n\
             Where: {}\n\
             Thread: {}\n\
             Program: {}\n\
             Engine: somnium_core {}\n\
             GPU: {}\n\
             System: {} {}\n\
             Log: {}\n\
             Dump: {dump}\n",
            self.program,
            utc(self.secs),
            self.secs,
            exception_name(self.code),
            self.code,
            self.at,
            self.thread,
            self.build,
            env!("CARGO_PKG_VERSION"),
            self.gpu,
            std::env::consts::OS,
            std::env::consts::ARCH,
            self.log,
        )
    }
}

/// The unhandled-exception filter (Windows): a minidump and a report for the first native crash, then whatever filter
/// was there before, then Windows' own handling.
#[cfg(windows)]
mod native {
    use super::{NATIVE, NativeFacts, build, watched, write_report};
    use std::ffi::c_void;
    use std::path::{Path, PathBuf};
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[repr(C)]
    struct ExceptionRecord {
        code: u32,
        flags: u32,
        record: *mut ExceptionRecord,
        address: *mut c_void,
        parameters: u32,
        information: [usize; 15],
    }

    #[repr(C)]
    pub(super) struct ExceptionPointers {
        record: *mut ExceptionRecord,
        context: *mut c_void,
    }

    /// `MINIDUMP_EXCEPTION_INFORMATION`, which dbghelp.h declares under `#pragma pack(4)`.
    #[repr(C, packed(4))]
    struct DumpException {
        thread_id: u32,
        pointers: *mut ExceptionPointers,
        client_pointers: i32,
    }

    type Filter = unsafe extern "system" fn(*mut ExceptionPointers) -> i32;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn SetUnhandledExceptionFilter(filter: Option<Filter>) -> Option<Filter>;
        fn GetCurrentProcess() -> *mut c_void;
        fn GetCurrentProcessId() -> u32;
        fn GetCurrentThreadId() -> u32;
        fn GetModuleHandleExW(flags: u32, address: *const c_void, module: *mut *mut c_void) -> i32;
        fn GetModuleFileNameW(module: *mut c_void, name: *mut u16, size: u32) -> u32;
        fn RaiseException(code: u32, flags: u32, count: u32, arguments: *const usize);
    }

    #[link(name = "dbghelp")]
    unsafe extern "system" {
        fn MiniDumpWriteDump(
            process: *mut c_void,
            pid: u32,
            file: *mut c_void,
            kind: u32,
            exception: *const DumpException,
            user: *const c_void,
            callback: *const c_void,
        ) -> i32;
    }

    #[link(name = "user32")]
    unsafe extern "system" {
        fn MessageBoxW(window: *mut c_void, text: *const u16, caption: *const u16, kind: u32) -> i32;
    }

    /// `MiniDumpNormal` with thread info and unloaded modules: every thread's stack, a few MB, small enough to send.
    const DUMP_KIND: u32 = 0x0000_1000 | 0x0000_0020;
    const EXCEPTION_CONTINUE_SEARCH: i32 = 0;
    /// `GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT`.
    const MODULE_FROM_ADDRESS: u32 = 0x4 | 0x2;

    static PREVIOUS: OnceLock<Option<Filter>> = OnceLock::new();
    static ENTERED: AtomicBool = AtomicBool::new(false);

    pub(super) fn install() {
        // SAFETY: registers a process-wide filter with the signature Windows calls it with; the previous one is kept
        // and called after ours.
        let previous = unsafe { SetUnhandledExceptionFilter(Some(filter)) };
        let _ = PREVIOUS.set(previous);
    }

    /// Raise an access violation, as a driver's or C++'s would arrive (the minidump's proof run).
    pub(crate) fn raise_access_violation() {
        let arguments = [1usize, 0];
        // SAFETY: raises a structured exception with two arguments in a live array; nothing catches it but the filter.
        unsafe { RaiseException(0xC000_0005, 0, 2, arguments.as_ptr()) };
    }

    /// The module an address lies in, as its file and the offset into it.
    fn module_of(address: *mut c_void) -> String {
        let mut module: *mut c_void = std::ptr::null_mut();
        // SAFETY: Win32 queries with valid out-pointers and a sized buffer; the module's refcount is left unchanged.
        let found = unsafe { GetModuleHandleExW(MODULE_FROM_ADDRESS, address, &mut module) } != 0;
        if !found || module.is_null() {
            return format!("{address:p} (in no module the program loaded)");
        }
        let mut name = [0u16; 520];
        let length = unsafe { GetModuleFileNameW(module, name.as_mut_ptr(), name.len() as u32) } as usize;
        let file = String::from_utf16_lossy(&name[..length.min(name.len())]);
        format!("{file} + {:#x}", (address as usize).wrapping_sub(module as usize))
    }

    fn write_dump(dir: &Path, secs: u64, pointers: *mut ExceptionPointers) -> Option<PathBuf> {
        use std::os::windows::io::AsRawHandle;
        std::fs::create_dir_all(dir).ok()?;
        let path = dir.join(format!("crash-{secs}.dmp"));
        let file = std::fs::File::create(&path).ok()?;
        // SAFETY: plain thread id query.
        let thread_id = unsafe { GetCurrentThreadId() };
        let exception = DumpException { thread_id, pointers, client_pointers: 0 };
        // SAFETY: a live file handle, this process, and the exception pointers Windows handed the filter.
        let written = unsafe {
            MiniDumpWriteDump(
                GetCurrentProcess(),
                GetCurrentProcessId(),
                file.as_raw_handle().cast(),
                DUMP_KIND,
                &exception,
                std::ptr::null(),
                std::ptr::null(),
            )
        } != 0;
        written.then_some(path)
    }

    unsafe extern "system" fn filter(pointers: *mut ExceptionPointers) -> i32 {
        // The first crash only: a fault inside this filter must not come back into it.
        if !ENTERED.swap(true, Ordering::SeqCst) {
            report(pointers);
        }
        match PREVIOUS.get().copied().flatten() {
            // SAFETY: the filter that was installed before ours, called as Windows would have.
            Some(previous) => unsafe { previous(pointers) },
            None => EXCEPTION_CONTINUE_SEARCH,
        }
    }

    fn report(pointers: *mut ExceptionPointers) {
        let Some((program, folders)) = NATIVE.get() else { return };
        // SAFETY: Windows hands the filter valid exception pointers for the duration of the call.
        let record = unsafe { pointers.as_ref().and_then(|p| p.record.as_ref()) };
        let (code, address, touched) = match record {
            Some(r) => {
                let touched = (r.code == 0xC000_0005 && r.parameters >= 2).then(|| (r.information[0], r.information[1]));
                (r.code, r.address, touched)
            }
            None => (0, std::ptr::null_mut(), None),
        };
        let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
        let dump = folders.iter().find_map(|dir| write_dump(dir, secs, pointers));
        // SAFETY: plain thread id query.
        let thread = unsafe { GetCurrentThreadId() };
        let facts = NativeFacts {
            program,
            secs,
            code,
            at: module_of(address),
            touched,
            thread,
            build: build(),
            gpu: somnium_renderer::context::gpu_in_use().unwrap_or("none yet (it stopped before the GPU started)"),
            log: super::LOG_PATH.get().map_or_else(|| "none".to_owned(), |p| p.display().to_string()),
            dump: dump.as_deref(),
        };
        let text = facts.report();
        let written = folders.iter().find_map(|dir| write_report(dir, secs, &text));
        eprintln!("{program} stopped with a native crash; report {}", written.as_ref().map_or_else(|| "not written".to_owned(), |p| p.display().to_string()));
        if watched() {
            let message = super::crash_text(program, written.as_deref());
            let wide = |s: &str| s.encode_utf16().chain(Some(0)).collect::<Vec<u16>>();
            let (text, caption) = (wide(&message), wide(program));
            // SAFETY: NUL-terminated UTF-16 strings that outlive the call; no owner window; error icon, topmost.
            unsafe { MessageBoxW(std::ptr::null_mut(), text.as_ptr(), caption.as_ptr(), 0x10 | 0x1_0000 | 0x4_0000) };
        }
    }
}

#[cfg(not(windows))]
mod native {
    pub(super) fn install() {}
    pub(crate) fn raise_access_violation() {
        std::process::abort();
    }
}

/// Raise a native access violation, as a driver's would arrive: the minidump's proof run (`SOMNIUM_EDITOR_SCRIPT`).
pub(crate) fn raise_native_crash() {
    native::raise_access_violation();
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
    fn a_native_crash_report_names_the_exception_the_module_and_the_dump() {
        let dump = Path::new("F:/game/logs/crash-42.dmp");
        let facts = NativeFacts {
            program: "Rimeborn: Sunscar",
            secs: 1_700_000_000,
            code: 0xC000_0005,
            at: "C:\\Windows\\System32\\nvoglv64.dll + 0x1a2b".to_owned(),
            touched: Some((1, 0x10)),
            thread: 7,
            build: "F:/game/sunscar.exe".to_owned(),
            gpu: "NVIDIA GeForce RTX 5080 Laptop GPU (Vulkan, NVIDIA 617.42)",
            log: "F:/game/logs/editor.log".to_owned(),
            dump: Some(dump),
        };
        let text = facts.report();
        for part in ["native crash", "an access violation", "writing 0x10", "0xc0000005", "nvoglv64.dll + 0x1a2b", "crash-42.dmp", "RTX 5080"] {
            assert!(text.contains(part), "{part} missing from:\n{text}");
        }
        assert_eq!(exception_name(0xC000_00FD), "a stack overflow");
        assert_eq!(exception_name(0x1234), "a native exception");
        let unwritten = NativeFacts { dump: None, touched: None, ..facts }.report();
        assert!(unwritten.contains("Dump: none") && !unwritten.contains("writing"));
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
