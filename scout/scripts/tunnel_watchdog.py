"""Long-running watchdog that owns the local meet stack.

Manages two child processes:

  1. ``cloudflared tunnel --url http://localhost:3000`` — exposes the
     local meet server to the public internet so Recall.ai can fetch
     ``bot.html``. Free quick-tunnels (the ``trycloudflare.com``
     hostnames) are unstable: Cloudflare can revoke the hostname at any
     time, causing a DNS NXDOMAIN. We monitor for that.
  2. ``node server.js`` (the upstream runway-characters-meet meet
     server). We start it via ``node`` directly to avoid the npm
     wrapper-hang we hit twice today.

Loop, every PROBE_INTERVAL_S:
  - HEAD the public tunnel URL with a short timeout.
  - If 3 consecutive probes fail (or DNS resolution fails immediately),
    treat the tunnel as dead → restart cloudflared, capture the new URL,
    rewrite ``meet/.env`` PUBLIC_URL, restart the meet server.

This replaces the manual sequence we keep doing (kill cloudflared, kill
meet, npx cloudflared, sed .env, restart meet). Run it once and forget:

    venv/bin/python -m scout.scripts.tunnel_watchdog

(Stop with Ctrl-C; both children are SIGTERM'd on shutdown.)

It will also:
  - kill any pre-existing cloudflared / node-server.js on startup so
    you never end up with two competing tunnels;
  - print clear status lines so you can see exactly what's happening.

Notes on what it deliberately does NOT do:
  - It doesn't manage Scout's uvicorn process — that's stable on
    localhost-only and never needs a tunnel.
  - It doesn't try to migrate a live brainstorm session through a
    tunnel rotation. If a brainstorm is mid-flight when Cloudflare
    evicts the URL, the existing Recall bots will fail (they cached the
    old hostname). The fix is to hit "End" on the brainstorm UI and
    start a new one — the new bots will use the new tunnel.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Optional

import requests
from dotenv import dotenv_values, set_key

ROOT = Path(__file__).resolve().parents[2]
MEET_DIR = ROOT / "meet"
LOG_DIR = ROOT / "scout" / "data" / "watchdog"

PROBE_INTERVAL_S = int(os.environ.get("WATCHDOG_PROBE_INTERVAL_S", "30"))
PROBE_TIMEOUT_S = float(os.environ.get("WATCHDOG_PROBE_TIMEOUT_S", "8"))
FAILURES_BEFORE_RESTART = int(os.environ.get("WATCHDOG_FAILURES", "3"))
MEET_PORT = int(os.environ.get("WATCHDOG_MEET_PORT", "3000"))
MEET_LOCAL_HEALTH_URL = f"http://localhost:{MEET_PORT}/"
TUNNEL_STARTUP_TIMEOUT_S = int(os.environ.get("WATCHDOG_TUNNEL_TIMEOUT_S", "45"))

log = logging.getLogger("watchdog")


# ---------------------------------------------------------------------------
# Process helpers
# ---------------------------------------------------------------------------


def _kill_pids_on_port(port: int) -> None:
    """SIGKILL anything bound to ``port``. macOS-friendly."""
    try:
        proc = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True, check=False,
        )
        for pid in proc.stdout.split():
            with suppress(Exception):
                os.kill(int(pid), signal.SIGKILL)
                log.info("killed pid %s on port %d", pid, port)
    except Exception as e:
        log.warning("lsof failed: %s", e)


def _kill_existing_cloudflared() -> None:
    try:
        subprocess.run(
            ["pkill", "-f", "cloudflared.*tunnel.*--url"],
            check=False, capture_output=True,
        )
    except Exception as e:
        log.warning("pkill cloudflared failed: %s", e)


def _start_cloudflared(local_port: int = MEET_PORT, *, detach: bool = False) -> tuple[subprocess.Popen, str]:
    """Start cloudflared. Block until we see its assigned URL or
    TUNNEL_STARTUP_TIMEOUT_S elapses. Return (process, url).

    Always writes child stdout/stderr to ``LOG_DIR/cloudflared.log``
    via dup'd file descriptors. We then tail that file to find the
    URL. This works identically in foreground and ``detach`` modes —
    in detach mode we additionally start the child in its own session
    so it outlives the Python process; the file FD remains valid.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "cloudflared.log"
    # Truncate so the URL-tail loop only sees the new run.
    log_path.write_text("")
    log.info("starting cloudflared tunnel → localhost:%d (detach=%s)", local_port, detach)
    log.info("  cloudflared logs → %s", log_path)
    log_fh = open(log_path, "ab", buffering=0)
    proc = subprocess.Popen(
        ["npx", "--yes", "cloudflared@latest", "tunnel", "--url", f"http://localhost:{local_port}"],
        stdout=log_fh,                # child writes straight to the file FD
        stderr=subprocess.STDOUT,
        cwd=str(MEET_DIR),
        start_new_session=detach,     # detach from our session/PG
    )
    log_fh.close()                    # parent doesn't need it open
    url = _tail_for_url(log_path, proc, timeout_s=TUNNEL_STARTUP_TIMEOUT_S)
    if not url:
        with suppress(Exception):
            proc.terminate()
        raise TimeoutError("cloudflared did not produce a tunnel URL in time")
    log.info("cloudflared assigned URL: %s", url)
    if not detach:
        _follow_log_in_background(log_path, "cloudflared")
    return proc, url


def _tail_for_url(path: Path, proc: subprocess.Popen, *, timeout_s: int) -> Optional[str]:
    """Block-read ``path`` until we spot a trycloudflare.com URL or
    ``proc`` exits or the timeout elapses."""
    url_re = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
    deadline = time.time() + timeout_s
    seen = ""
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"cloudflared exited early (rc={proc.returncode}) before producing a URL"
            )
        try:
            seen = path.read_text(errors="ignore")
        except Exception:
            seen = ""
        m = url_re.search(seen)
        if m:
            return m.group(0)
        time.sleep(0.4)
    return None


def _follow_log_in_background(path: Path, label: str) -> None:
    """Foreground convenience: tail a log file and print new lines to
    our stdout with a label. Daemon thread; dies when watchdog exits."""
    import threading
    def _pump() -> None:
        try:
            with path.open("r") as fh:
                fh.seek(0, os.SEEK_END)   # only show NEW output
                while True:
                    line = fh.readline()
                    if not line:
                        time.sleep(0.5)
                        continue
                    print(f"[{label}] {line.rstrip()}", flush=True)
        except Exception:
            pass
    t = threading.Thread(target=_pump, daemon=True)
    t.start()


def _start_meet(env: dict, *, detach: bool = False) -> subprocess.Popen:
    """Run meet/server.js via node directly (no npm wrapper — avoids
    the silent-hang we hit twice). Pass the rotated env in."""
    _kill_pids_on_port(MEET_PORT)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "meet.log"
    log_path.write_text("")
    log.info("starting meet server (node server.js, port %d, detach=%s)", MEET_PORT, detach)
    log.info("  meet logs → %s", log_path)
    log_fh = open(log_path, "ab", buffering=0)
    proc = subprocess.Popen(
        ["node", "server.js"],
        cwd=str(MEET_DIR),
        env={**os.environ, **{k: v for k, v in env.items() if v is not None}},
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=detach,
    )
    log_fh.close()
    if not detach:
        _follow_log_in_background(log_path, "meet")
    # Wait briefly for it to listen.
    deadline = time.time() + 15
    while time.time() < deadline:
        with suppress(Exception):
            r = requests.get(MEET_LOCAL_HEALTH_URL, timeout=2)
            if r.status_code < 500:
                log.info("meet server is listening on :%d", MEET_PORT)
                return proc
        if proc.poll() is not None:
            raise RuntimeError(
                f"meet server exited early (rc={proc.returncode})"
            )
        time.sleep(0.5)
    raise TimeoutError("meet server did not start listening within 15s")


# (Helper drain/follow functions live above next to _start_cloudflared.)


# ---------------------------------------------------------------------------
# .env shuffling
# ---------------------------------------------------------------------------


def _read_meet_env() -> dict:
    p = MEET_DIR / ".env"
    if not p.exists():
        raise FileNotFoundError(f"meet/.env not found at {p}")
    return dict(dotenv_values(p))


def _write_public_url(url: str) -> None:
    p = MEET_DIR / ".env"
    set_key(str(p), "PUBLIC_URL", url, quote_mode="never")
    log.info("wrote PUBLIC_URL=%s into meet/.env", url)


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def _probe_tunnel(url: str) -> bool:
    # Step 1: DNS. NXDOMAIN means Cloudflare evicted the hostname — no
    # point continuing.
    host = url.split("//", 1)[-1].split("/", 1)[0]
    try:
        socket.gethostbyname(host)
    except socket.gaierror as e:
        log.warning("DNS probe failed for %s: %s", host, e)
        return False
    # Step 2: HTTP. Anything 2xx/3xx/4xx means the tunnel is reaching
    # something — we only treat connection errors / timeouts as failure.
    try:
        r = requests.head(url, timeout=PROBE_TIMEOUT_S, allow_redirects=False)
        if r.status_code < 500:
            return True
        log.warning("HTTP probe got 5xx (%d) from %s", r.status_code, url)
        return False
    except requests.RequestException as e:
        log.warning("HTTP probe failed for %s: %s", url, e)
        return False


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class StackManager:
    """Owns one cloudflared and one meet server. Methods are safe to
    call repeatedly — they tear down the prior procs first."""

    def shutdown(self) -> None:
        for proc, name in [(self.meet_proc, "meet"), (self.cf_proc, "cloudflared")]:
            if proc and proc.poll() is None:
                log.info("terminating %s (pid=%d)", name, proc.pid)
                with suppress(Exception):
                    proc.terminate()
                with suppress(Exception):
                    proc.wait(timeout=5)
                if proc.poll() is None:
                    with suppress(Exception):
                        proc.kill()
        self.cf_proc = None
        self.meet_proc = None

    def __init__(self, *, detach: bool = False) -> None:
        self.cf_proc: Optional[subprocess.Popen] = None
        self.meet_proc: Optional[subprocess.Popen] = None
        self.tunnel_url: Optional[str] = None
        self.detach = detach

    def boot(self) -> None:
        """Tear down anything pre-existing and bring up a fresh stack."""
        log.info("booting stack…")
        _kill_pids_on_port(MEET_PORT)
        _kill_existing_cloudflared()
        time.sleep(0.5)

        self.cf_proc, self.tunnel_url = _start_cloudflared(MEET_PORT, detach=self.detach)
        env = _read_meet_env()
        env["PUBLIC_URL"] = self.tunnel_url
        _write_public_url(self.tunnel_url)
        self.meet_proc = _start_meet(env, detach=self.detach)
        log.info("stack ready · tunnel=%s · meet=:%d", self.tunnel_url, MEET_PORT)

    def rotate(self) -> None:
        """Tear down + start fresh. Used when the tunnel goes bad."""
        log.warning("ROTATING stack (tunnel went bad)")
        self.shutdown()
        time.sleep(1.0)
        self.boot()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe-interval", type=int, default=PROBE_INTERVAL_S,
        help=f"Seconds between health probes (default {PROBE_INTERVAL_S}).",
    )
    parser.add_argument(
        "--failures-before-restart", type=int, default=FAILURES_BEFORE_RESTART,
        help=f"Consecutive failed probes before rotating (default {FAILURES_BEFORE_RESTART}).",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Just (re)boot the stack, write the URL, and exit. Skips the loop.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    stack = StackManager(detach=args.once)
    try:
        stack.boot()
    except Exception as e:
        log.exception("initial boot failed: %s", e)
        return 1

    if args.once:
        log.info(
            "--once mode: stack detached. cloudflared pid=%s, meet pid=%s. "
            "Logs at %s. NOTE: there is no auto-recovery now — if "
            "Cloudflare evicts the URL, restart the watchdog.",
            stack.cf_proc.pid if stack.cf_proc else "?",
            stack.meet_proc.pid if stack.meet_proc else "?",
            LOG_DIR,
        )
        return 0

    log.info(
        "entering watch loop · probe every %ds · rotate after %d consecutive failures",
        args.probe_interval, args.failures_before_restart,
    )

    consecutive_failures = 0
    try:
        while True:
            time.sleep(args.probe_interval)
            url = stack.tunnel_url or ""
            if not url:
                log.warning("no tunnel URL — forcing rotate")
                stack.rotate()
                consecutive_failures = 0
                continue

            ok = _probe_tunnel(url)
            # Also check the local meet server isn't dead.
            meet_dead = stack.meet_proc is not None and stack.meet_proc.poll() is not None
            if ok and not meet_dead:
                if consecutive_failures > 0:
                    log.info("tunnel recovered — back to OK")
                consecutive_failures = 0
                continue

            consecutive_failures += 1
            log.warning(
                "probe failed (%d/%d) · meet_dead=%s",
                consecutive_failures, args.failures_before_restart, meet_dead,
            )
            if consecutive_failures >= args.failures_before_restart or meet_dead:
                try:
                    stack.rotate()
                except Exception as e:
                    log.exception("rotate failed: %s — will keep retrying", e)
                consecutive_failures = 0
    except KeyboardInterrupt:
        log.info("Ctrl-C — shutting down stack")
    finally:
        stack.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
