#!/usr/bin/env bash
# Lucid's landing check: the engine in THIS checkout, and the games that build against it, stay whole. Run it before
# landing (for example after merging `sunscar` into `dev_rimborn`, before pushing).
#
#   bash tools/check_games.sh            # from any Lucid checkout or worktree
#   JOBS=1 bash tools/check_games.sh     # if OneDrive locks the linker (LNK1104); it also retries once by itself
#   TSF=1 bash tools/check_games.sh      # also TSF's tests (see below)
#   SUNSCAR=<dir> CHECK_TARGET=<dir>     # where Sunscar is (default F:/Somnium Games/rimeborn-fp) and where its
#                                        # check build goes (default E:/target-check-games, else target/check-games)
#
# What it runs:
# - the engine: somnium_core's tests, somnium_ui's, every shader through naga (no GPU needed), the whole workspace;
# - Sunscar against this checkout: its manifest names the engine by absolute path (SomniumEngine-sunscar), so the
#   check writes a copy whose engine paths point here and builds and tests that, in its own target dir;
# - TSF only with TSF=1: it builds against SomniumEngine-tsf, its own fork, so it says nothing about this checkout
#   (the developer, 18 Oct: no engine or Rimeborn change may interfere with TSF; the forks keep that true).
#   Read-only for TSF, and checked: TSF's own tests write into its folder (the tutorials-seen save in saves/, the
#   asset index cache, a document fixture under evidence/), so the fixture test is skipped, the other two files are
#   backed up before TSF's tests and put back byte for byte after, and any other TSF file the run changes fails the
#   check by name. Its ignored audits (they write evidence) stay ignored;
# - the isometric Rimeborn (postponed) only when it sits in this checkout's games/, where its relative engine paths
#   resolve.
# Exit 0 only if everything that ran passed.
set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd -W 2>/dev/null || pwd)
JOBS="${JOBS:-4}"
SUNSCAR="${SUNSCAR:-F:/Somnium Games/rimeborn-fp}"
if [ -z "${CHECK_TARGET:-}" ]; then
    if [ -d E:/ ]; then CHECK_TARGET=E:/target-check-games; else CHECK_TARGET=target/check-games; fi
fi
status=0
run() {
    echo "== $1"
    shift
    local log
    log=$(mktemp)
    local ok=1
    "$@" >"$log" 2>&1 || ok=0
    if [ "$ok" -eq 0 ] && grep -q "1104" "$log"; then
        # LNK1104: OneDrive held a file the linker wanted (memory note: use -j 1). Once more, single job.
        local args=() prev=""
        for a in "$@"; do if [ "$prev" = "-j" ]; then args+=("1"); else args+=("$a"); fi; prev="$a"; done
        echo "   (linker lock; retrying with -j 1)"
        ok=1
        "${args[@]}" >"$log" 2>&1 || ok=0
    fi
    if [ "$ok" -eq 1 ]; then
        grep -E "^test result" "$log" | head -3
    else
        grep -E "^error|FAILED|panicked|^test result" "$log" | head -40
        echo "!! FAILED: $*"
        status=1
    fi
    rm -f "$log"
}
run "engine core" cargo test -p somnium_core -j "$JOBS" --quiet
run "engine ui" cargo test -p somnium_ui --lib -j "$JOBS" --quiet
run "engine shaders (naga, no GPU)" cargo test -p somnium_renderer --test shaders_validate -j "$JOBS" --quiet
run "engine workspace builds" cargo check --workspace --all-targets -j "$JOBS" --quiet

# Sunscar, against this checkout's engine.
if [ -f "$SUNSCAR/Cargo.toml" ]; then
    copy="$CHECK_TARGET/sunscar-manifest"
    mkdir -p "$copy"
    sed -e "s|path = \"[^\"]*/crates/|path = \"$ROOT/crates/|" \
        -e "s|path = \"[^\"]*/examples/|path = \"$ROOT/examples/|" "$SUNSCAR/Cargo.toml" > "$copy/Cargo.toml"
    printf '\n[lib]\npath = "%s/src/lib.rs"\n\n[[bin]]\nname = "sunscar"\npath = "%s/src/main.rs"\n' \
        "$SUNSCAR" "$SUNSCAR" >> "$copy/Cargo.toml"
    for f in Cargo.lock rust-toolchain.toml; do [ -f "$SUNSCAR/$f" ] && cp "$SUNSCAR/$f" "$copy/$f"; done
    if grep -q "path = \"$ROOT/crates/somnium_core\"" "$copy/Cargo.toml"; then
        run "Sunscar (against this checkout)" cargo test --manifest-path "$copy/Cargo.toml" \
            --target-dir "$CHECK_TARGET/sunscar" -j "$JOBS" --quiet
    else
        echo "!! FAILED: could not point Sunscar's manifest at this checkout ($copy/Cargo.toml)"
        status=1
    fi
else
    echo "== Sunscar: no manifest at $SUNSCAR (skipped; SUNSCAR=<dir> to point at it)"
fi

# TSF, opt-in.
TSF_DIR=games/TheSomnusFracture
if [ "${TSF:-0}" = 1 ] && [ -f "$TSF_DIR/Cargo.toml" ]; then
    KEEP=(saves/player-tutorials.somsave assets/.somnium/asset-index.json Cargo.lock)
    backup=$(mktemp -d)
    for f in "${KEEP[@]}"; do
        if [ -e "$TSF_DIR/$f" ]; then mkdir -p "$backup/$(dirname "$f")" && cp -p "$TSF_DIR/$f" "$backup/$f"; fi
    done
    marker="$backup/.marker"
    touch "$marker"
    run "TSF" cargo test --manifest-path "$TSF_DIR/Cargo.toml" --target-dir target -j "$JOBS" --quiet -- \
        --skip private_component_document_draft_and_conflicts_share_live_schemas
    changed=$(find "$TSF_DIR" -type f -newer "$marker" -not -path "*/target/*" 2>/dev/null \
        | grep -v -F -e "$TSF_DIR/saves/player-tutorials.somsave" -e "$TSF_DIR/assets/.somnium/asset-index.json" \
            -e "$TSF_DIR/Cargo.lock")
    if [ -n "$changed" ]; then
        # Something else in TSF changed while its tests ran: a TSF session (or the developer) is working in it. Its
        # writes may include the two files below, so nothing is put back; look before landing.
        echo "!! FAILED: TSF files changed during the check (a TSF session may be working; nothing was put back):"
        echo "$changed" | head -20
        status=1
    else
        for f in "${KEEP[@]}"; do
            [ -e "$backup/$f" ] || continue
            if ! cmp -s "$backup/$f" "$TSF_DIR/$f"; then
                echo "   (put back TSF $f, which its own tests rewrote)"
            fi
            cp -p "$backup/$f" "$TSF_DIR/$f"
        done
    fi
    rm -rf "$backup"
else
    echo "== TSF: skipped (it builds against SomniumEngine-tsf, not this checkout; TSF=1 runs its tests anyway)"
fi

# The isometric Rimeborn (postponed), only where its relative engine paths resolve.
if [ -f games/Rimeborn/Cargo.toml ]; then
    run "Rimeborn (isometric)" cargo test --manifest-path games/Rimeborn/Cargo.toml --target-dir target -j "$JOBS" --quiet
else
    echo "== Rimeborn (isometric): not in games/ (postponed; skipped)"
fi
[ "$status" -eq 0 ] && echo "== the engine and its games whole" || echo "== SOMETHING BROKE: fix it before landing"
exit "$status"
