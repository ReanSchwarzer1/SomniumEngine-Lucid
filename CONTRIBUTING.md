# Contributing to Lucid

Thanks for your interest. Lucid is the open-world fork of Somnium Engine, a from-scratch Rust engine and a research
project. Issues, discussion and pull requests are welcome.

## Which repository?

There are two engines with a shared past:

- **[Somnium Engine](https://github.com/ReanSchwarzer1/SomniumEngine)**, the original, serves *The Somnus Fracture*
  (small first-person levels). It alone has packed game data (`somnium_vfs`).
- **Lucid** (this repository) serves open worlds: Sunscar and Rimeborn. GPU grass, weather, crash reports and model
  validation started here, and its world streaming goes further.

Open the pull request where the code you change lives. A fix that both engines need goes to one of them first and is
ported to the other by hand: the trees have diverged far enough (Lucid uses reversed depth, for one) that a
cherry-pick rarely applies cleanly. Say in the description if you know the other engine has the same bug.

Day-to-day work lands on the `sunscar` branch and reaches `main` in batches, so look at `sunscar` before you start
on something.

## Getting set up

Requirements:
- **Windows 10 or 11** with a Vulkan GPU. It is the only platform built and tested.
- **Rust 1.88** (pinned in [`rust-toolchain.toml`](rust-toolchain.toml)).
- A **C++ toolchain** (MSVC Build Tools) for the Jolt physics bridge and Luau.

```sh
cargo build --workspace
cargo test  --workspace --lib      # what CI runs
cargo run   -p hello_engine        # the editor
```

On Windows a test run sometimes fails to link with `LNK1104` because another process holds a freshly built file
(OneDrive, an antivirus). That is not a code error: run again with `-j 1`, or test one crate at a time
(`cargo test -p somnium_core --lib -j 1`).

## Before opening a pull request

- **Warnings:** CI builds with `-D warnings`, so the workspace must compile warning-free.
- **Tests:** `cargo test --workspace --lib` must pass. Add tests for new CPU-side logic where practical. GPU paths are
  not covered in CI (the runner has no GPU), so say how you checked them.
- **Lint:** `cargo clippy --workspace --all-targets` runs in CI as advice; fix what you reasonably can.
- **Formatting:** do not run `cargo fmt` over existing files. The engine is hand-wrapped near 120 columns and has no
  `rustfmt.toml`, so rustfmt would rewrite hundreds of unrelated lines and bury your change. Format only files you
  create, and match the lines around your edit everywhere else.
- **Gates:** `python tools/ghostfence/run.py` runs every gate (`--fast` skips the test row).

## Project conventions

1. **Study references, don't copy them.** Somnium reimplements patterns from other engines from scratch. When a
   subsystem is informed by a reference, cite the specific file in [`ATTRIBUTION.md`](ATTRIBUTION.md) (and in a source
   comment). Never paste third-party source.
2. **Don't commit `example_repo/`.** It holds local reference repositories and is gitignored (see the note in
   [`README.md`](README.md)). The one tracked exception is the vendored Jolt Physics subtree, which the build compiles.
3. **Keep the living docs current.** Architectural changes should be reflected in [`context.md`](context.md) (and
   `ATTRIBUTION.md` if a new reference was used). New authoring UI also needs a page or section in
   [`docs/editor/`](docs/editor/).
4. **Find out what already exists before building it.** Read [`context.md`](context.md) for the current architecture,
   then [`dev records/README.md`](dev%20records/README.md) for which phases have landed and where their evidence lives,
   and [`docs/history.md`](docs/history.md) for what Lucid added after the fork. A phase plan is not proof that a
   feature exists, and older plans preserve the tree they were written against.
5. **Match the surrounding style.** New code should read like the code around it: naming, comment density, and
   idiom.
6. **Per-frame code never scans the whole world.** Query by component (`iter_with` / `entities_with` / `first_with`),
   rebuild caches only when `World::change_signature` changes, and write a component only when its value changes (a
   write every frame defeats every cache that reads it). `crates/somnium_core/tests/full_scan_ratchet.rs` fails on a
   new `.entities()` scan in the engine or any game, and at run time `cpu_watchdog` logs a warning naming the frame
   stage whenever a scene goes CPU-bound. Read `editor.log` before profiling.
7. **New public fields can break games.** A game that builds an engine struct literally, without
   `..Default::default()`, stops compiling when a field is added. Check the games built on the engine before adding a
   field to a public struct.
8. **Claims need measurements.** A frame-time claim comes from matched before-and-after `.somtime` runs at the same
   camera; a change inside the noise band is reported as noise.

## A note on AI assistance

This project is developed with AI pair-programming assistance (Claude Code). Contributions made with similar tooling
are welcome. Hold them to the same bar: understand the code you submit, make sure it builds and is tested, and keep
the attribution rules above.

## License

By contributing, you agree that your contributions are dual-licensed under **MIT OR Apache-2.0**, matching the project
license.
