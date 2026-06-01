#!/usr/bin/env bash
# Kill any leftover vLLM processes and report GPU memory state.
#
# Usage:
#   scripts/clean_vllm.sh           # graceful (TERM → 5s → KILL)
#   scripts/clean_vllm.sh --force   # straight to KILL, no grace period
#   scripts/clean_vllm.sh --quiet   # suppress per-step output (still shows summary)
#
# What it kills (in order):
#   - `vllm serve …` driver processes
#   - VLLM::EngineCore* (incl. zombie/<defunct>)
#   - VLLM::Worker_TP*  (the orphan workers that survive a parent kill)
#   - multiprocessing.resource_tracker
#   - serve_deepseek_v4_flash.sh wrappers (so they don't try to relaunch)
#
# What it does NOT kill:
#   - Random Python processes; matching is done on vllm-specific patterns.
#
# Stale KFD references with PID = "UNKNOWN" holding 0 B VRAM are harmless;
# the driver garbage-collects them on the next allocation. Non-zero VRAM
# held by an UNKNOWN PID means the kernel still has a page table for a
# process that exited; usually `sleep 5` or the next vllm launch clears
# it. A driver reset (`echo 1 > /sys/class/drm/cardN/device/reset`) is
# the only hard fix but is rarely needed in practice.

set -uo pipefail

FORCE=0
QUIET=0
for arg in "$@"; do
    case "$arg" in
        --force|-f) FORCE=1 ;;
        --quiet|-q) QUIET=1 ;;
        -h|--help)
            sed -n '2,/^set -uo/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

log() { [[ "$QUIET" -eq 1 ]] || echo "[clean_vllm] $*"; }

# Pattern that matches real vLLM *runtime* processes but NOT this script or
# the shell that launched it. We intentionally avoid matching every command
# line that merely contains the substring "vllm" (e.g. `/app/vllm_deepseek/…`)
# so the cleaner doesn't list / try to kill itself.
VLLM_RUNTIME_RE='VLLM::|/vllm( |$)| vllm serve|/vllm serve|multiprocessing\.resource_tracker'

# Pretty-print currently-alive vLLM runtime processes (skip $$ / $PPID).
list_vllm_procs() {
    # NB: $6 is `comm` (executable name); drop the awk/grep/sed/ps the
    # cleaner itself spawns, otherwise the regex string in their argv
    # makes them match themselves.
    ps -eo pid,ppid,pcpu,etime,stat,comm,args --no-headers \
        | awk -v me=$$ -v par=$PPID -v re="$VLLM_RUNTIME_RE" \
              '$1 != me && $1 != par \
               && $6 !~ /^(awk|grep|sed|ps|bash|sh)$/ \
               && $0 ~ re { print }'
}

# ---------------------------------------------------------------------------
# Pre-state
# ---------------------------------------------------------------------------
log "before: vllm-related processes:"
if [[ "$QUIET" -eq 0 ]]; then
    out="$(list_vllm_procs)"
    if [[ -z "$out" ]]; then echo "  (none)"; else echo "$out"; fi
fi

# ---------------------------------------------------------------------------
# Kill phase
# ---------------------------------------------------------------------------
PATTERNS=(
    'serve_deepseek_v4_flash\.sh'
    'vllm serve'
    'VLLM::EngineCore'
    'VLLM::Worker'
    'multiprocessing\.resource_tracker'
)

kill_pattern() {
    local sig="$1" pat="$2"
    local pids
    pids="$(pgrep -f "$pat" || true)"
    [[ -z "$pids" ]] && return 0
    log "  $sig  $pat  -> pids: $(echo $pids | tr '\n' ' ')"
    # shellcheck disable=SC2086
    kill -"$sig" $pids 2>/dev/null || true
}

if [[ "$FORCE" -eq 0 ]]; then
    log "sending SIGTERM…"
    for pat in "${PATTERNS[@]}"; do kill_pattern TERM "$pat"; done

    # Wait up to 5 s for graceful exit.
    for i in 1 2 3 4 5; do
        any_alive=0
        for pat in "${PATTERNS[@]}"; do
            if pgrep -f "$pat" >/dev/null 2>&1; then any_alive=1; break; fi
        done
        [[ "$any_alive" -eq 0 ]] && break
        sleep 1
    done
fi

log "sending SIGKILL to anything still alive…"
for pat in "${PATTERNS[@]}"; do kill_pattern KILL "$pat"; done

# Reap zombies whose parent shells are still around (kill the parents).
sleep 2
zombies="$(ps -eo pid,stat,comm | awk '$2 ~ /Z/ && $3 ~ /VLLM/ {print $1}' || true)"
if [[ -n "$zombies" ]]; then
    log "reaping zombie parents for: $zombies"
    for z in $zombies; do
        ppid=$(awk '/^PPid:/{print $2}' /proc/$z/status 2>/dev/null || echo "")
        [[ -n "$ppid" && "$ppid" != "1" ]] && kill -9 "$ppid" 2>/dev/null || true
    done
fi

# ---------------------------------------------------------------------------
# Post-state
# ---------------------------------------------------------------------------
sleep 2
log "after: vllm-related processes:"
remaining="$(list_vllm_procs)"
if [[ -z "$remaining" ]]; then
    log "  (none) ✓"
else
    echo "$remaining"
    log "WARNING: processes still alive (may be in uninterruptible D-state)"
fi

# ---------------------------------------------------------------------------
# /dev/shm cleanup
# ---------------------------------------------------------------------------
# vLLM's shm_broadcast IPC creates `/dev/shm/psm_<hex>` POSIX shared-memory
# segments and only `shm_unlink()`s them on clean shutdown. SIGKILL'd workers
# leave them behind — they're sparse files so the actual RAM cost is tiny,
# but the directory entries pile up forever (we've seen 30+ accumulate over
# weeks). Delete only the ones that NO live process is mmap'ing right now;
# that way we never yank a segment out from under an unrelated tenant.
log "scanning /dev/shm for orphan psm_* segments…"
shm_total=0; shm_freed=0; shm_kept=0; shm_bytes=0
if compgen -G '/dev/shm/psm_*' >/dev/null 2>&1; then
    # Build the set of psm_* files that ARE mapped by some live process by
    # grepping all /proc/*/maps once. This is portable (no fuser/lsof
    # dependency) and fast enough for hundreds of segments. /proc/*/maps
    # lines look like:
    #   7f3c...-7f3c... rw-s 00000000 00:18 1234 /dev/shm/psm_abcd1234
    # so a fixed-string grep for "/dev/shm/psm_" picks them all up.
    in_use="$(grep -hF '/dev/shm/psm_' /proc/*/maps 2>/dev/null \
              | awk '{print $NF}' | sort -u || true)"
    for f in /dev/shm/psm_*; do
        shm_total=$((shm_total + 1))
        if printf '%s\n' "$in_use" | grep -Fxq "$f"; then
            shm_kept=$((shm_kept + 1))
            log "  KEEP   $(basename "$f") (mapped by a live process)"
        else
            sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
            shm_bytes=$((shm_bytes + sz))
            rm -f "$f" && shm_freed=$((shm_freed + 1))
        fi
    done
fi
if [[ "$shm_total" -gt 0 ]]; then
    log "  psm_*: $shm_total found, deleted $shm_freed (declared $((shm_bytes / 1024 / 1024)) MiB), kept $shm_kept in-use"
else
    log "  psm_*: (none found) ✓"
fi

# ---------------------------------------------------------------------------
# /dev/shm semaphore cleanup (sem.loky-* and sem.mp-*)
# ---------------------------------------------------------------------------
# Two other classes of POSIX objects also leak when workers are SIGKILL'd:
#
#   sem.loky-<PID>-<random>  : semaphores from joblib's `loky` process pool
#                              (pulled in transitively by some sklearn/HF
#                              pre-processing). The owning PID is right in
#                              the filename, so we can tell whether the
#                              owner is still alive.
#
#   sem.mp-<random>          : semaphores from Python's `multiprocessing`
#                              module (e.g. Lock / Event / Semaphore objects
#                              shared between EngineCore and workers).
#                              No PID in the name — fall back to the same
#                              `/proc/*/maps` test we use for psm_*.
#
# Each is only 32 bytes so the RAM impact is negligible, but they accrete
# across every crashed run (we've seen entries from April still hanging
# around in May) and pollute the namespace. Same safety rule as for psm_*:
# only delete when we can prove nothing live is using it.

log "scanning /dev/shm for orphan sem.loky-* / sem.mp-* …"
loky_total=0; loky_freed=0; loky_kept=0
if compgen -G '/dev/shm/sem.loky-*' >/dev/null 2>&1; then
    for f in /dev/shm/sem.loky-*; do
        loky_total=$((loky_total + 1))
        # filename: sem.loky-<PID>-<random>
        pid="$(basename "$f" | awk -F- '{print $2}')"
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            loky_kept=$((loky_kept + 1))
        else
            rm -f "$f" && loky_freed=$((loky_freed + 1))
        fi
    done
fi
if [[ "$loky_total" -gt 0 ]]; then
    log "  sem.loky-*: $loky_total found, deleted $loky_freed (owner PID gone), kept $loky_kept (owner alive)"
else
    log "  sem.loky-*: (none found) ✓"
fi

mp_total=0; mp_freed=0; mp_kept=0
if compgen -G '/dev/shm/sem.mp-*' >/dev/null 2>&1; then
    in_use_mp="$(grep -hF '/dev/shm/sem.mp-' /proc/*/maps 2>/dev/null \
                 | awk '{print $NF}' | sort -u || true)"
    for f in /dev/shm/sem.mp-*; do
        mp_total=$((mp_total + 1))
        if printf '%s\n' "$in_use_mp" | grep -Fxq "$f"; then
            mp_kept=$((mp_kept + 1))
        else
            rm -f "$f" && mp_freed=$((mp_freed + 1))
        fi
    done
fi
if [[ "$mp_total" -gt 0 ]]; then
    log "  sem.mp-*: $mp_total found, deleted $mp_freed (unmapped), kept $mp_kept (mapped)"
else
    log "  sem.mp-*: (none found) ✓"
fi

# ---------------------------------------------------------------------------
# GPU memory summary
# ---------------------------------------------------------------------------
if command -v rocm-smi >/dev/null 2>&1; then
    log "GPU VRAM usage (post-clean):"
    rocm-smi --showmeminfo vram 2>&1 \
        | awk '/VRAM Total Used Memory/ {
            mb = $NF / 1024 / 1024
            mark = (mb > 200) ? " ⚠"  : ""
            printf "  %-10s %8.1f MiB%s\n", $1, mb, mark
          }'

    log "KFD process table (UNKNOWN entries w/ non-zero VRAM = stale driver refs):"
    rocm-smi --showpids 2>&1 \
        | awk '/^[0-9]/ {
            mb = $4 / 1024 / 1024
            mark = (mb > 100) ? " ⚠ stale" : ""
            printf "  pid=%-10s gpu=%s  vram=%8.1f MiB%s\n",
                   $1, $3, mb, mark
          }' \
        | sed 's/^/  /'
fi

exit 0
