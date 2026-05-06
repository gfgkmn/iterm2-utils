#!/usr/bin/env python3

import asyncio
import json
import os
import re
import subprocess
from functools import partial
from pathlib import Path

import aiohttp.web
import iterm2
import paramiko
import pyperclip


# ─────────────────────────────────────────────────────────────────────
# Registry-based CC pane detection.
#
# Replaces the old screen-scrape-only detect_session_type / cwd
# extraction.  Ground-truth sources (filesystem + process tree),
# regex screen-scrape kept as last-resort fallback only.
#
#   1. ~/.claude/sessions/<PID>.json  — every running CC writes one
#      with {pid, sessionId, cwd, ...}.
#   2. ~/.claude/cc-state/<UUID>.json — bridge statusline writes
#      {tmux_session, session_name, transcript_path, ...}.
#   3. ps -ax -o pid,ppid           — full PID tree (one syscall).
#   4. tmux list-clients / list-panes — for tmux-hosted CC.
#
# The detection function itself returns rich data drawn from the
# registry rather than guessed from screen content.

_SESSIONS_DIR = Path.home() / ".claude" / "sessions"
_CC_STATE_DIR = Path.home() / ".claude" / "cc-state"


def _resolve_tmux_binary():
    """Find the tmux binary at module load time.

    The iTerm-spawned Python doesn't inherit the user's interactive
    PATH — `/opt/homebrew/bin' (Apple-Silicon Homebrew) and
    `/usr/local/bin' (Intel Homebrew / typical custom installs) are
    NOT on PATH for the embedded server, so `subprocess.run([\"tmux\",
    ...])' fails with FileNotFoundError, gets caught silently, and
    every tmux-aware code path collapses to a no-op.

    Resolve once: try shutil.which (works if PATH is fine), then
    common Homebrew locations.  Returns None if tmux isn't found at
    all (degrades to non-tmux-aware behavior — same as before, but
    NOW INTENTIONAL rather than silently broken)."""
    import shutil
    p = shutil.which("tmux")
    if p:
        return p
    for cand in ("/opt/homebrew/bin/tmux", "/usr/local/bin/tmux",
                 "/usr/bin/tmux"):
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


_TMUX_BIN = _resolve_tmux_binary()


def _load_sessions_registry():
    """Return dict pid -> {pid, sessionId, cwd, ...} for every alive
    CC process.  Skips PIDs whose process no longer exists (stale
    sessions/<PID>.json files left after a crash)."""
    reg = {}
    if not _SESSIONS_DIR.is_dir():
        return reg
    for f in _SESSIONS_DIR.glob("*.json"):
        try:
            d = json.loads(f.read_text())
            pid = d.get("pid")
            if not pid:
                continue
            try:
                os.kill(int(pid), 0)
            except (ProcessLookupError, PermissionError, ValueError, TypeError):
                continue
            reg[int(pid)] = d
        except Exception:
            pass
    return reg


def _load_cc_state_by_uuid():
    """Return dict sessionId -> cc-state dict (tmux_session,
    session_name, transcript_path, cwd, ...)."""
    out = {}
    if not _CC_STATE_DIR.is_dir():
        return out
    for f in _CC_STATE_DIR.glob("*.json"):
        try:
            d = json.loads(f.read_text())
            sid = d.get("session_id") or d.get("sessionId")
            if sid:
                out[sid] = d
        except Exception:
            pass
    return out


def _build_children_map():
    """Return dict ppid -> [child_pid, ...] from a single
    `ps -ax -o pid,ppid` syscall.

    macOS NOTE: do NOT use `pgrep -P <ppid>` — without a pattern
    argument it returns nothing on macOS (different from Linux).
    `ps` is portable."""
    try:
        out = subprocess.check_output(
            ["ps", "-ax", "-o", "pid,ppid"],
            text=True, timeout=2, stderr=subprocess.DEVNULL)
    except Exception:
        return {}
    children = {}
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except ValueError:
            continue
        children.setdefault(ppid, []).append(pid)
    return children


def _claude_pid_in_ancestry(start_pid, registry, max_hops=15):
    """Walk PPIDs upward from start_pid; return first ancestor in
    registry, or None.  Bounded to avoid runaway traversal."""
    if not start_pid:
        return None
    pid = int(start_pid)
    for _ in range(max_hops):
        if pid in registry:
            return pid
        try:
            out = subprocess.check_output(
                ["ps", "-o", "ppid=", "-p", str(pid)],
                text=True, timeout=1, stderr=subprocess.DEVNULL).strip()
            ppid = int(out) if out else None
        except Exception:
            return None
        if not ppid or ppid <= 1:
            return None
        pid = ppid
    return None


def _claude_pid_in_descendants(start_pid, registry, children_map,
                               max_total=80):
    """BFS down the children tree from start_pid; return first
    descendant in registry, or None.

    For tmux-pane PIDs: tmux's pane_pid is typically the shell
    wrapper (bash/zsh); CC is its direct or grandchild descendant.
    Walk-up doesn't reach it; this walks down."""
    if not start_pid:
        return None
    if start_pid in registry:
        return start_pid
    queue = list(children_map.get(start_pid, []))
    visited = {start_pid}
    while queue and len(visited) < max_total:
        pid = queue.pop(0)
        if pid in visited:
            continue
        visited.add(pid)
        if pid in registry:
            return pid
        queue.extend(children_map.get(pid, []))
    return None


def _tmux_clients_by_tty():
    """Return dict tty -> session_name from `tmux list-clients`.
    Reflects the CURRENT session each client is attached to —
    correct after `switch-client', unlike parsing the iTerm pane's
    job_args (which records only the launch command).

    Returns {} when tmux binary isn't reachable from this server
    (see `_resolve_tmux_binary')."""
    if not _TMUX_BIN:
        return {}
    try:
        out = subprocess.check_output(
            [_TMUX_BIN, "list-clients", "-F",
             "#{client_tty} #{session_name}"],
            text=True, timeout=2, stderr=subprocess.DEVNULL)
    except Exception:
        return {}
    m = {}
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            m[parts[0]] = parts[1]
    return m


def _tmux_pane_pids(session_name):
    """Return list of pane PIDs in tmux session, or []."""
    if not session_name or not _TMUX_BIN:
        return []
    try:
        out = subprocess.check_output(
            [_TMUX_BIN, "list-panes", "-t", session_name,
             "-F", "#{pane_pid}"],
            text=True, timeout=2, stderr=subprocess.DEVNULL)
    except Exception:
        return []
    pids = []
    for line in out.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return pids


def _parse_tmux_session_from_args(job_args):
    """Last-resort parse of `tmux ... -t <session>` from job_args.
    Less reliable than `_tmux_clients_by_tty` because the user may
    have done `switch-client` since launch."""
    if not job_args:
        return None
    m = re.search(r'-t\s+(\S+)', job_args)
    if m:
        return m.group(1).rstrip(';').rstrip(',')
    return None


async def claude_info_for_pane(session, registry=None, by_uuid=None,
                               children_map=None, tmux_by_tty=None):
    """Return rich CC info for SESSION, or None if not CC.

    Detection priority (ground-truth → heuristic):

      1. PID-walk UP from session.jobPid: catches CC running directly
         in the iTerm pane and CC's tool subprocesses (chrome-devtools-mcp,
         node, bash, etc.).
      2. tmux-tty lookup: when the pane's tty is attached to a tmux
         session (per `list-clients`), list its panes' PIDs and walk
         DOWN to find a CC descendant.

    Returns dict with: session_id, claude_pid, current_dir,
    session_name, transcript_path, tmux_session, detection_path.

    The cached helper-data args (registry / by_uuid / children_map /
    tmux_by_tty) are populated at call time when None — pass them in
    when iterating multiple panes (find_claude_sessions) to avoid
    repeated filesystem / subprocess work."""
    if registry is None:
        registry = _load_sessions_registry()
    if by_uuid is None:
        by_uuid = _load_cc_state_by_uuid()
    if children_map is None:
        children_map = _build_children_map()
    if tmux_by_tty is None:
        tmux_by_tty = _tmux_clients_by_tty()

    # Read all four iTerm variables in parallel via asyncio.gather.
    # Sequential `await session.async_get_variable(...)` (the previous
    # shape) costs N WebSocket roundtrips per pane; gather collapses
    # them to ~1 roundtrip's worth.  With ~8 panes, this is the
    # dominant win for `find_claude_sessions` latency.
    # `return_exceptions=True` keeps a single failed read from killing
    # the whole gather — we coerce exceptions to None below.
    results = await asyncio.gather(
        session.async_get_variable("jobName"),
        session.async_get_variable("jobPid"),
        session.async_get_variable("commandLine"),
        session.async_get_variable("tty"),
        return_exceptions=True)
    job_name = None if isinstance(results[0], BaseException) else results[0]
    job_pid_raw = None if isinstance(results[1], BaseException) else results[1]
    job_args = None if isinstance(results[2], BaseException) else results[2]
    tty = None if isinstance(results[3], BaseException) else results[3]
    try:
        job_pid = int(job_pid_raw) if job_pid_raw else None
    except (TypeError, ValueError):
        job_pid = None

    def _build_result(cc_pid, detection_path, tmux_name=None):
        d = registry[cc_pid]
        sid = d.get("sessionId")
        ccs = by_uuid.get(sid, {}) if sid else {}
        return {
            "session_id": sid,
            "claude_pid": cc_pid,
            "current_dir": d.get("cwd") or ccs.get("cwd"),
            "session_name": ccs.get("session_name"),
            "transcript_path": ccs.get("transcript_path"),
            "tmux_session": tmux_name or ccs.get("tmux_session"),
            "detection_path": detection_path,
            # Pass through the per-pane variables so the
            # `find_claude_sessions` caller doesn't need to re-read
            # them (used to cost 2 extra sequential awaits per pane).
            "job_name": job_name,
            "job_pid": job_pid,
        }

    # Path 1: PID-walk up from jobPid.
    if job_pid:
        match_pid = _claude_pid_in_ancestry(job_pid, registry)
        if match_pid:
            return _build_result(match_pid, "pid_walk")

    # Path 2: tmux session — prefer list-clients (current attach),
    # fall back to job_args (launch command).
    tmux_name = None
    if tty and tty in tmux_by_tty:
        tmux_name = tmux_by_tty[tty]
    elif job_name == "tmux":
        tmux_name = _parse_tmux_session_from_args(job_args)

    if tmux_name:
        for pane_pid in _tmux_pane_pids(tmux_name):
            cc_pid = _claude_pid_in_descendants(
                pane_pid, registry, children_map)
            if cc_pid:
                return _build_result(
                    cc_pid, f"tmux_lookup({tmux_name})", tmux_name)

    return None


def detect_session_type(lines):
    """Detect if session is in pdb, ipython, or bash"""
    session_indicators = {
        'pdb': [
            r'\(Pdb\)',
            r'\(pdb\)',
            r'--Return--',
            r'ipdb>',
            r'\(Pdb\+\+\)',
        ],
        'ipython': [
            r'In \[\d+\]:',  # IPython input prompt
            r'Out\[\d+\]:',  # IPython output prompt
            r'IPython \d+\.\d+',  # IPython version info
            r'^\s*\.\.\.:',  # IPython continuation prompt
        ],
        'claude': [
            # Primary signal: CC's `[claude]:` statusline at bottom
            # of pane.  Visible during normal idle/streaming render.
            r'\[claude\]:',
            # Permission-dialog umbrellas — when CC's TUI shows a
            # tool-permission prompt the dialog covers the bottom
            # statusline, so `[claude]:` isn't visible.  Every
            # permission prompt CC 2.x emits starts with one of these
            # two phrases (verified against the 2.1.119 binary):
            #   "Do you want to <action>?"   — proceed, make this
            #     edit, allow this connection, use this API key,
            #     allow Claude to fetch, etc.
            #   "Claude wants to <action>"   — enter/exit plan mode,
            #     fetch content from this URL, guide you through ...
            r'Do you want to ',
            r'Claude wants to ',
        ],
        'bash': [
            r'\$ $',
            r'# $',
            r'[^>]*@[^>]*:[^>]*\$ ',
            r'[^>]*:[^>]*\$ ',
            r'^yuhe\$',
            r'^gfgkmn\\>:',
        ]
    }

    # Check all lines (reversed) — some indicators like [claude]: status line
    # may not be in the last few lines
    all_lines = [line.string.strip() for line in lines]

    for line in reversed(all_lines):
        for session_type, patterns in session_indicators.items():
            for pattern in patterns:
                if re.search(pattern, line):
                    return session_type

    return 'unknown'


def parse_ssh_config():
    """Parse SSH config file and return a dictionary of host configurations"""
    config_file = os.path.expanduser('~/.ssh/config')
    if os.path.exists(config_file):
        try:
            config = paramiko.SSHConfig()
            with open(config_file) as f:
                config.parse(f)
            return config
        except Exception as e:
            print(f"Error parsing SSH config: {e}")
    return None


def is_host_in_config(host_string, config):
    """Check if a host string matches any configured host pattern"""
    if not config:
        return False

    # Lookup returns the resolved config if the host matches any pattern
    host_config = config.lookup(host_string)

    # Check if this resolved to an actual configured host
    # (paramiko always returns something, so we need to verify)
    configured_hosts = config.get_hostnames()

    # Try to match against each configured pattern
    for pattern in configured_hosts:
        if config.lookup(pattern).get('hostname') == host_config.get('hostname'):
            # Additional check: see if the pattern actually matches our string
            import fnmatch
            if fnmatch.fnmatch(host_string, pattern):
                return True

    return False


def reverse_lookup_ssh_config(machine_string, config):
    """Reverse lookup SSH config by hostname to find matching host entries"""
    if not config:
        return []

    matching_host = None

    # Get all host patterns from the config
    for host in config.get_hostnames():
        # Look up the configuration for this host pattern
        host_config = config.lookup(host)

        # Check if the hostname matches our target
        if host_config.get('hostname') == machine_string:
            matching_host = host
            break

    return matching_host


def get_hostname_from_config_when_jump(job_args, configs):
    """
    Extract the real target hostname from SSH -W command and SSH config
    For command like 'ssh -W \[192.168.53.80\]:22 caiyunjump'
    Verifies both target host and jump host match SSH config

    Args:
        job_args (str): The full SSH command string from job_args
        configs (paramiko.SSHConfig): Parsed SSH config object from paramiko

    Returns:
        str: The resolved target hostname for the %h part
    """
    if not configs:
        return None

    # Remove escaped brackets, regular brackets, and double quotes
    job_args = job_args.replace('\\[', '').replace('\\]', '')
    job_args = job_args.replace('[', '').replace(']', '').replace('"', '')
    parts = job_args.split()

    # Find the part with the target host (after -W) and jump host (last part)
    try:
        w_index = parts.index('-W')
        if w_index + 1 < len(parts):
            target_with_port = parts[w_index + 1]
            target_host = target_with_port.split(':')[0]  # Remove port part
            jump_host = parts[-1]  # Last part is jump host

            # Look through all hosts in config for target host
            for host_entry in configs._config:
                if 'config' in host_entry and 'hostname' in host_entry[
                        'config'] and 'proxyjump' in host_entry['config']:
                    # Clean up hostname for comparison
                    config_hostname = host_entry['config']['hostname'].replace(
                        '[', '').replace(']', '')
                    config_jump = host_entry['config']['proxyjump']
                    if config_hostname == target_host and config_jump == jump_host:
                        return host_entry['host'][0]

            # If no matching hostname found, return the target host itself
            return target_host

    except (ValueError, IndexError):
        return None

    return None


def determine_officel_package(line_str):
    """Determine if a line contains official package information"""
    line_str = line_str.replace('\\', '/')

    # Regex pattern to match lib/pythonX.Y/site-packages
    # This will match patterns like:
    # - lib/python3.7/site-packages
    # - lib/python3.11/site-packages
    # etc.
    site_package_pattern = r'lib/python\d+(\.\d+)?/site-packages'

    # List of packages to exclude
    excluded_packages = ['transformers', 'litellm']

    # Check if it matches the site-packages pattern
    is_site_package = bool(re.search(site_package_pattern, line_str))

    # Check if it's not in excluded packages
    is_excluded = any(pkg in line_str for pkg in excluded_packages)

    return is_site_package and not is_excluded


def final_mappping(path_str):
    """Apply final path mappings"""
    if not path_str:
        return ""
    update_path = ""
    mappings = {
        "/root/.cache/huggingface/modules/transformers_modules":
            "/root/workspace/crhavk47v38s73fnfgbg/dcformer",
        "/home/yuhe/.cache/huggingface/modules/transformers_modules":
            "/home/yuhe/dcformer",
        "/data1/hub/modules/transformers_modules":
            "/home/yuhe/dcformer",
        "/home/yuhe/.cache/huggingface/modules/transformers_modules/unify":
            "/home/yuhe/dcformer-origin",
        "/home/yuhe/.cache/huggingface/modules/transformers_modules/unify_format":
            "/home/yuhe/dcformer",
        "/data1/hub/modules/transformers_modules/global_step_594":
            "/home/yuhe/dcformer",
    }
    for old_path, new_path in mappings.items():
        if old_path in path_str:
            update_path = path_str.replace(old_path, new_path)
    if update_path:
        return update_path
    return path_str


def reconstruct_logical_lines(lines):
    """Reconstruct logical lines from wrapped physical lines using iTerm2's hard_eol property"""
    logical_lines = []
    current_line = ""

    for line in lines:
        current_line += line.string

        # If this line has a hard end-of-line, it's the end of a logical line
        if line.hard_eol:
            logical_lines.append(current_line)
            current_line = ""

    if current_line:
        logical_lines.append(current_line)

    return logical_lines


async def get_all_panes_info(connection):
    """Get information from all panes in the current tab."""
    app = await iterm2.async_get_app(connection)
    window = app.current_terminal_window

    if not window:
        return []

    tab = window.current_tab
    if not tab:
        return []

    sessions = tab.sessions
    results = []
    active_session = tab.current_session

    for pane_ind, session in enumerate(sessions):
        pane_info = await get_pane_info(connection, session)
        pane_info["pane_index"] = pane_ind

        if session.session_id == active_session.session_id:
            results.insert(0, pane_info)
        else:
            results.append(pane_info)

    return results


async def get_pane_info(connection, target_session):
    """Get information from a specific session/pane."""
    app = await iterm2.async_get_app(connection)
    window = app.current_terminal_window

    if window and target_session:
        line_info = await target_session.async_get_line_info()
        lines = await target_session.async_get_contents(
            first_line=line_info.first_visible_line_number,
            number_of_lines=line_info.mutable_area_height)

        session_type = detect_session_type(lines)
        logical_lines = reconstruct_logical_lines(lines)

        # Registry-based CC detection AND the username/hostname/job
        # variable reads in parallel.  Both groups are independent
        # reads against the same iTerm session — no ordering
        # dependency, so a single `asyncio.gather' collapses ~8
        # sequential WebSocket round-trips into one round-trip's
        # worth of latency.  Cuts `get_session_info' from ~170 ms to
        # ~80 ms; matters for the spawn-completion polling loops in
        # the bridge (`--wait-for-claude-on-pane',
        # `--delegate-append-user-after-spawn').
        gather_results = await asyncio.gather(
            claude_info_for_pane(target_session),
            target_session.async_get_variable("username"),
            target_session.async_get_variable("hostname"),
            target_session.async_get_variable("jobName"),
            target_session.async_get_variable("commandLine"),
            return_exceptions=True)
        def _ok(v):
            return None if isinstance(v, BaseException) else v
        cc_info = _ok(gather_results[0])
        username = _ok(gather_results[1])
        hostname = _ok(gather_results[2])
        job_name = _ok(gather_results[3])
        job_args = _ok(gather_results[4])

        cc_session_id = None
        cc_claude_pid = None
        cc_session_name = None
        cc_transcript_path = None
        cc_tmux_session = None
        cc_detection_path = None
        cc_current_dir = None
        if cc_info:
            session_type = 'claude'
            cc_session_id = cc_info.get("session_id")
            cc_claude_pid = cc_info.get("claude_pid")
            cc_session_name = cc_info.get("session_name")
            cc_transcript_path = cc_info.get("transcript_path")
            cc_tmux_session = cc_info.get("tmux_session")
            cc_detection_path = cc_info.get("detection_path")
            cc_current_dir = cc_info.get("current_dir")

        line = -1
        current_file = ""
        arrow_line = -1

        for logical_line in reversed(logical_lines):
            if determine_officel_package(logical_line):
                continue
            file_match = re.search(r'(File|>)\s*"?([^"=(]+)(", line |\()(\d+)\)?',
                                   logical_line)
            if file_match:
                current_file = file_match.group(2)
                line = int(file_match.group(4))
                break
            line_match = re.search(r'-> (\d*)', logical_line)
            if line_match and arrow_line == -1:
                try:
                    arrow_line = int(line_match.group(1))
                except ValueError:
                    pass

        if line == -1:
            line = arrow_line

        # username / hostname / jobName / commandLine were already
        # read above (in the gather block); reuse them here.

        # Fallback: if screen scrape didn't detect `claude' (e.g. user
        # scrolled back past the `[claude]:' prompt), but the foreground
        # process is named `claude' or a common wrapper, treat as claude.
        # Without this, send_code picks the bash branch and mangles input.
        if session_type != 'claude' and job_name:
            jn = job_name.lower()
            if jn == 'claude' or jn.startswith('claude-') or jn == 'claude-light':
                session_type = 'claude'

        # Prefer the registry-derived cwd when available (cc-state's
        # `cwd' field is CC's actual working directory, not whatever
        # the [claude]: line happens to display right now).  Falls
        # back to the screen-scrape regex / iTerm `path' var when the
        # registry didn't find a match — preserves behavior for CC
        # versions / setups the registry can't see.
        current_dir = cc_current_dir or ""
        if session_type == 'claude' and not current_dir:
            for logical_line in reversed(logical_lines):
                claude_match = re.search(
                    r'\[claude\]:\s*(?:\{[^}]*\}\s*)?[^/]*(\/.*?)[\s\x00]+\d{2}/\d{2}/\d{2}[\s\x00]+\d{2}:\d{2}:\d{2}',
                    logical_line)
                if claude_match:
                    current_dir = claude_match.group(1).strip()
                    break
            # Last-resort: iTerm's `path' var (from OSC 7) when even
            # the regex misses (e.g., permission dialog covers the
            # [claude]: statusline AND the registry didn't find a
            # match — rare).
            if not current_dir:
                try:
                    pv = await target_session.async_get_variable("path")
                    if pv and isinstance(pv, str):
                        current_dir = pv.strip()
                except Exception:
                    pass

        if current_dir == "":
            for logical_line in reversed(logical_lines):
                folder_match = re.search(
                    r"(?:^\([^\(]*\))?[^\s:]*:(\/.*?)\s*\d{4}-\d{2}-\d{2}\s*\d{2}:\d{2}:\d{2}",
                    logical_line)
                if folder_match:
                    current_dir = folder_match.group(1)
                    break

        if current_dir == "":
            if current_file:
                current_dir = os.path.dirname(current_file)
            else:
                current_dir = await target_session.async_get_variable("path")

        if job_name == "ssh":
            configs = parse_ssh_config()
            if job_args.startswith("ssh -W"):
                hostname = get_hostname_from_config_when_jump(job_args, configs)
            else:
                if job_args.find('@') != -1:
                    machine_str = job_args.split('@')[-1]
                else:
                    machine_str = job_args.split(' ')[-1]

                machine_str = machine_str.strip('"').strip("'")

                if is_host_in_config(machine_str, configs):
                    hostname = machine_str
                else:
                    hostname = reverse_lookup_ssh_config(machine_str, configs)

        if (not username or username == "") and job_name == "ssh":
            try:
                username = job_args.split(" ")[1].split("@")[0]
            except (IndexError, AttributeError):
                username = ""

        if (not hostname or hostname == "") and job_name == "ssh":
            try:
                hostname = job_args.split("@")[-1].split(":")[0]
            except (IndexError, AttributeError):
                hostname = ""

        if hostname:
            hostname = hostname.strip('"').strip("'")  # ADD THIS LINE

        return {
            "username": username,
            "hostname": hostname,
            "job_name": job_name,
            "job_args": job_args,
            "current_dir": current_dir,
            "line": line,
            "current_file": final_mappping(current_file),
            "session_type": session_type,
            "iterm_session_id": target_session.session_id,
            # Rich CC info from the registry (None when not CC or
            # when the registry didn't find a match; bridge consumers
            # treat these as authoritative when present).
            "session_id": cc_session_id,
            "claude_pid": cc_claude_pid,
            "session_name": cc_session_name,
            "transcript_path": cc_transcript_path,
            "tmux_session": cc_tmux_session,
            "cc_detection_path": cc_detection_path,
        }

    return {}


async def send_to_session(connection, session, code, is_ascii, multiline):
    """Send code to a single iTerm2 session."""
    if is_ascii:
        # Send ASCII character
        ascii_code = ord(code) if len(code) == 1 else int(code)
        await session.async_send_text(chr(ascii_code))
    else:
        # Get session info to determine type
        session_info = await get_pane_info(connection, session)
        session_type = session_info.get('session_type', 'unknown')

        if session_type == 'claude':
            # Bracketed paste for ALL claude input (single + multiline).
            # ESC[200~ … ESC[201~ tells CC's Ink/React input reader
            # "treat as one atomic paste":
            #   * Multi-line: embedded \n stay as literal newlines, not
            #     submissions (replaces the old per-line Esc+Enter dance,
            #     which broke when the 50 ms gap exceeded CC's escape-
            #     timeout and split each \n into its own turn).
            #   * Single-line: prevents partial-byte loss when CC is mid-
            #     render (atomic paste can't be partially eaten).
            # The 100 ms sleep before \r lets CC finish processing the
            # paste before the submission key arrives — without it, \r
            # sometimes lands while CC is still in a popup/autocomplete
            # state and the Enter is swallowed (prompt sits in input box).
            await session.async_send_text('\x1b[200~')
            await session.async_send_text(code)
            await session.async_send_text('\x1b[201~')
            # 300 ms (was 100 ms): the gap covers (a) CC's React re-render
            # of the input box after `\x1b[201~', and (b) any autocomplete
            # / slash-menu popup CC opens for `@<path>' or `/<cmd>' in the
            # pasted content.  When the popup is still resolving when \r
            # arrives, the popup absorbs it (selects suggestion / dismisses
            # menu) and the prompt is left sitting in the input box —
            # exactly the "needs an extra Enter" symptom users report.
            # 300 ms is a heuristic, not a guarantee; if it still races,
            # the next step is screen-scrape verify-and-retry, not a
            # bigger sleep.  Don't try Escape-before-Enter — CC reserves
            # Escape+Enter for multiline newline insertion.
            await asyncio.sleep(0.3)
            await session.async_send_text('\r')
        elif multiline:
            # For IPython sessions, use %paste magic command
            if session_type == 'ipython':
                # Copy to clipboard first
                pyperclip.copy(code)
                # Send %paste command
                await session.async_send_text('%paste\n')
                # Small delay to let IPython process the command
                await asyncio.sleep(0.2)
            else:
                # For other sessions (bash, pdb, etc.)
                pyperclip.copy(code)
                # Send Ctrl+U to clear line, then paste
                await session.async_send_text('\x15')  # Ctrl+U
                await asyncio.sleep(0.1)
                await session.async_send_text(code)
                await session.async_send_text('\n')
        else:
            # Single line - send directly
            await session.async_send_text(code)
            await session.async_send_text('\n')


async def send_code_to_iterm(code,
                             connection,
                             target_pane=None,
                             session_id=None,
                             broadcast=False,
                             is_ascii=False,
                             multiline=False):
    """Enhanced function to send code to iTerm2 with broadcast and targeting support"""
    app = await iterm2.async_get_app(connection)

    if session_id:
        # Direct session lookup — works across all windows
        session = app.get_session_by_id(session_id)
        if session:
            await send_to_session(connection, session, code, is_ascii, multiline)
            return True
        return False

    window = app.current_terminal_window

    if not window:
        return False

    tab = window.current_tab
    if not tab:
        return False

    try:
        if broadcast:
            # Broadcast to all sessions in current tab
            sessions = tab.sessions
            for session in sessions:
                await send_to_session(connection, session, code, is_ascii, multiline)
        else:
            # Send to specific pane or current session
            if target_pane is not None:
                # Select specific pane (convert from 1-based to 0-based)
                pane_index = target_pane - 1
                sessions = tab.sessions
                if 0 <= pane_index < len(sessions):
                    session = sessions[pane_index]
                else:
                    session = tab.current_session
            else:
                session = tab.current_session

            await send_to_session(connection, session, code, is_ascii, multiline)

        return True

    except Exception as e:
        print(f"Error sending code to iTerm: {e}")
        return False


async def select_pane(connection, pane_number):
    """Select a specific pane in the current tab"""
    app = await iterm2.async_get_app(connection)
    window = app.current_terminal_window

    if not window:
        return False

    tab = window.current_tab
    if not tab:
        return False

    try:
        sessions = tab.sessions
        # Convert from 1-based to 0-based indexing
        pane_index = pane_number - 1

        if 0 <= pane_index < len(sessions):
            await tab.async_select_session(sessions[pane_index])
            return True
        else:
            # If pane number is out of range, select first session
            if sessions:
                await tab.async_select_session(sessions[0])
            return False
    except Exception as e:
        print(f"Error selecting pane: {e}")
        return False


async def create_new_pane(connection):
    """Create a new pane in the current tab.
    Returns a dict with the new session's `iterm_session_id` on success,
    False on failure. Returning the id lets callers (e.g. the bridge
    spawn flow) follow up with send_code against the new pane."""
    app = await iterm2.async_get_app(connection)
    window = app.current_terminal_window

    if not window:
        return False

    tab = window.current_tab
    if not tab:
        return False

    try:
        current_session = tab.current_session
        if current_session:
            new_session = await current_session.async_split_pane(vertical=True)
            return {"iterm_session_id": new_session.session_id, "tab_id": tab.tab_id}
        return False
    except Exception as e:
        print(f"Error creating new pane: {e}")
        return False


# HTTP Request Handlers


async def handle_send_code(request, connection):
    """Enhanced handler for sending code with broadcast and targeting support"""
    try:
        data = await request.json()
        code = data.get('code', '')
        # print(f"DEBUG: Received code repr: {repr(code)}")  # Add this line
        target_pane = data.get('target_pane')
        session_id = data.get('session_id')
        broadcast = data.get('broadcast', False)
        is_ascii = data.get('is_ascii', False)
        multiline = data.get('multiline', False)

        if not code:
            return aiohttp.web.Response(text='No code provided', status=400)

        success = await send_code_to_iterm(code,
                                           connection,
                                           target_pane,
                                           session_id=session_id,
                                           broadcast=broadcast,
                                           is_ascii=is_ascii,
                                           multiline=multiline)

        if success:
            if broadcast:
                return aiohttp.web.Response(
                    text=f'Code broadcasted to all panes successfully')
            elif target_pane:
                return aiohttp.web.Response(
                    text=f'Code sent to pane {target_pane} successfully')
            else:
                return aiohttp.web.Response(text='Code sent to current pane successfully')
        else:
            return aiohttp.web.Response(text='Failed to send code', status=500)

    except Exception as e:
        return aiohttp.web.Response(text=f'Error processing request: {str(e)}',
                                    status=400)


async def handle_control(request, connection):
    """Handler for control operations like selecting panes or creating new ones"""
    try:
        data = await request.json()
        action = data.get('action')

        if action == 'select_pane':
            pane_number = data.get('pane_number')
            if pane_number is None:
                return aiohttp.web.Response(text='Pane number required', status=400)

            success = await select_pane(connection, pane_number)
            if success:
                return aiohttp.web.Response(text=f'Selected pane {pane_number}')
            else:
                return aiohttp.web.Response(text=f'Failed to select pane {pane_number}',
                                            status=500)

        elif action == 'new_pane':
            result = await create_new_pane(connection)
            if isinstance(result, dict):
                return aiohttp.web.json_response(result)
            elif result:
                return aiohttp.web.Response(text='New pane created successfully')
            else:
                return aiohttp.web.Response(text='Failed to create new pane', status=500)

        elif action == 'find_window_by_profile':
            profile_name = data.get('profile')
            app = await iterm2.async_get_app(connection)
            await app.async_refresh()
            for window in app.windows:
                for tab in window.tabs:
                    for session in tab.sessions:
                        p = await session.async_get_variable("profileName")
                        if p == profile_name:
                            tabs_info = []
                            for t in window.tabs:
                                sess = t.current_session
                                tabs_info.append({
                                    "tab_id": t.tab_id,
                                    "session_id": sess.session_id if sess else None,
                                })
                            return aiohttp.web.json_response({
                                "window_id": window.window_id,
                                "tabs": tabs_info,
                            })
            return aiohttp.web.json_response({"window_id": None})

        elif action == 'find_active_window':
            app = await iterm2.async_get_app(connection)
            await app.async_refresh()
            w = app.current_terminal_window
            return aiohttp.web.json_response({
                "window_id": w.window_id if w else None,
            })

        elif action == 'create_tab':
            profile_name = data.get('profile', 'Emacs Hotkey Window')
            window_id = data.get('window_id')
            app = await iterm2.async_get_app(connection)
            if window_id:
                window = app.get_window_by_id(window_id)
                if window:
                    tab = await window.async_create_tab(profile=profile_name)
                    session = tab.current_session
                    return aiohttp.web.json_response({
                        "session_id": session.session_id,
                        "tab_id": tab.tab_id,
                    })
            # Fallback: create new window with this profile
            window = await iterm2.Window.async_create(connection, profile=profile_name)
            tab = window.current_tab
            session = tab.current_session
            return aiohttp.web.json_response({
                "window_id": window.window_id,
                "session_id": session.session_id,
                "tab_id": tab.tab_id,
            })

        elif action == 'activate_tab':
            session_id = data.get('session_id')
            app = await iterm2.async_get_app(connection)
            session = app.get_session_by_id(session_id)
            if session:
                window, tab = app.get_window_and_tab_for_session(session)
                if tab:
                    # Select the tab within its window.
                    await tab.async_activate()
                    # Also raise the window itself.  For NORMAL windows
                    # this is enough; for HOTKEY windows (hidden by
                    # default) `async_activate()` is a no-op — iTerm
                    # reserves unhide to the OS hotkey.  We surface the
                    # hotkey-window-ness in the response so the caller
                    # can simulate the hotkey (osascript Cmd+9) on its
                    # side.  Detection via `isHotkeyWindow` window
                    # variable; verified empirically (returns "1" / "0").
                    hotkey = False
                    if window:
                        try:
                            await window.async_activate()
                        except Exception:
                            pass
                        try:
                            v = await window.async_get_variable(
                                "isHotkeyWindow")
                            hotkey = bool(v) and v != "0"
                        except Exception:
                            hotkey = False
                    return aiohttp.web.json_response(
                        {"status": "ok", "hotkey_window": hotkey})
            return aiohttp.web.json_response({"error": "Session not found"}, status=404)

        elif action == 'get_session_info':
            session_id = data.get('session_id')
            app = await iterm2.async_get_app(connection)
            await app.async_refresh()
            session = app.get_session_by_id(session_id)
            if session:
                info = await get_pane_info(connection, session)
                return aiohttp.web.json_response(info)
            return aiohttp.web.json_response({"error": "not_found"})

        elif action == 'find_claude_sessions':
            # Walk every iTerm session in every window, return the ones
            # that are running Claude Code.  Detection uses the registry
            # (~/.claude/sessions/<PID>.json + process tree + tmux state)
            # as primary source; falls back to the screen-scrape regex
            # only for panes the registry can't see (rare).
            #
            # Returns rich per-pane info INCLUDING `session_id` from
            # the registry — bridge consumers (`--build-iterm-uuid-map')
            # can map uuid → iid directly without re-walking the PID
            # tree on the elisp side.
            app = await iterm2.async_get_app(connection)
            await app.async_refresh()
            # Build registry caches ONCE per scan, share across panes
            # to avoid N redundant filesystem / subprocess passes.
            registry = _load_sessions_registry()
            by_uuid = _load_cc_state_by_uuid()
            children_map = _build_children_map()
            tmux_by_tty = _tmux_clients_by_tty()

            async def _process_pane(window, tab, session):
                """Build the per-pane result dict, or None for non-CC panes.
                Runs all per-pane async work concurrently with sibling panes
                via the gather call below."""
                try:
                    # Kick off cc_info + hostname read in parallel.
                    # cc_info handles its own internal gather of 4 vars,
                    # so adding hostname here doesn't add latency — they
                    # all share the same WebSocket round-trip window.
                    cc_info, hostname_or_exc = await asyncio.gather(
                        claude_info_for_pane(
                            session,
                            registry=registry,
                            by_uuid=by_uuid,
                            children_map=children_map,
                            tmux_by_tty=tmux_by_tty),
                        session.async_get_variable("hostname"),
                        return_exceptions=True)
                    if isinstance(cc_info, BaseException):
                        return None
                    hostname = (
                        "" if isinstance(hostname_or_exc, BaseException)
                        else (hostname_or_exc or ""))
                    if not cc_info:
                        # Registry didn't find a CC for this pane.  Fall
                        # back to the slower regex screen-scrape via
                        # get_pane_info — catches the rare cases where
                        # registry isn't authoritative.
                        info = await get_pane_info(connection, session)
                        if info.get("session_type") != 'claude':
                            return None
                        try:
                            job_pid_raw = await session.async_get_variable("jobPid")
                            job_pid = int(job_pid_raw) if job_pid_raw else None
                        except (TypeError, ValueError, Exception):
                            job_pid = None
                        return {
                            "iterm_session_id": session.session_id,
                            "iterm_window_id": window.window_id,
                            "iterm_tab_id": tab.tab_id,
                            "job_pid": job_pid,
                            "hostname": info.get("hostname", ""),
                            "current_dir": info.get("current_dir", ""),
                            "session_type": "claude",
                            "session_id": None,
                            "claude_pid": None,
                            "session_name": None,
                            "transcript_path": None,
                            "tmux_session": None,
                            "cc_detection_path": "regex_scrape",
                        }
                    # Registry hit.  cc_info already carries job_name and
                    # job_pid from its own gather — reuse them instead of
                    # re-reading (used to cost 2 extra sequential awaits
                    # per pane).
                    return {
                        "iterm_session_id": session.session_id,
                        "iterm_window_id": window.window_id,
                        "iterm_tab_id": tab.tab_id,
                        "job_pid": cc_info.get("job_pid"),
                        "job_name": cc_info.get("job_name"),
                        "hostname": hostname,
                        "current_dir": cc_info.get("current_dir") or "",
                        "session_type": "claude",
                        "session_id": cc_info.get("session_id"),
                        "claude_pid": cc_info.get("claude_pid"),
                        "session_name": cc_info.get("session_name"),
                        "transcript_path": cc_info.get("transcript_path"),
                        "tmux_session": cc_info.get("tmux_session"),
                        "cc_detection_path": cc_info.get("detection_path"),
                    }
                except Exception:
                    # Skip panes we can't introspect; don't fail the
                    # whole scan.
                    return None

            # Run one coroutine per pane, all concurrently.  iTerm's
            # WebSocket pipelines requests fine; this collapses N panes'
            # serial latency to ~max(per-pane time).
            pane_results = await asyncio.gather(*[
                _process_pane(window, tab, session)
                for window in app.windows
                for tab in window.tabs
                for session in tab.sessions
            ], return_exceptions=False)
            results = [r for r in pane_results if r is not None]
            return aiohttp.web.json_response({"sessions": results})

        elif action == 'find_active_session':
            # Return the iterm session_id of the currently-focused pane.
            # Used by the bridge to decide whether to skip auto-popup
            # when the user is already looking at the target tab.
            #
            # Note: this returns the LAST-active iTerm session when iTerm
            # itself isn't the frontmost macOS app — the bridge filters
            # via `(frame-focus-state)' on the elisp side to be sure
            # iTerm has OS focus before trusting this answer.
            app = await iterm2.async_get_app(connection)
            await app.async_refresh()
            w = app.current_terminal_window
            if w and w.current_tab and w.current_tab.current_session:
                return aiohttp.web.json_response({
                    "iterm_session_id": w.current_tab.current_session.session_id,
                    "iterm_window_id": w.window_id,
                    "iterm_tab_id": w.current_tab.tab_id,
                })
            return aiohttp.web.json_response({"iterm_session_id": None})

        elif action == 'find_pane_by_tty':
            # Reverse-lookup: given a pty path like '/dev/ttys003' (typically
            # the output of `tmux list-clients -F #{client_tty}`), return the
            # iTerm session id whose pane is sitting on that tty.
            #
            # Used by the bridge's tmux fallback resolver: when CC runs
            # inside tmux, there's no shared ancestor between the iTerm
            # pane and the CC process, so the PID-tree-walk in
            # find_claude_sessions can't bind them.  But every iTerm pane
            # has a stable `tty' variable, and `tmux list-clients' returns
            # the tty of every attached client — match them and we know
            # which iTerm pane is currently showing the tmux session.
            tty = data.get('tty')
            if not tty:
                return aiohttp.web.json_response(
                    {"error": "missing tty"}, status=400)
            app = await iterm2.async_get_app(connection)
            await app.async_refresh()
            for window in app.windows:
                for tab in window.tabs:
                    for session in tab.sessions:
                        try:
                            s_tty = await session.async_get_variable("tty")
                        except Exception:
                            s_tty = None
                        if s_tty == tty:
                            return aiohttp.web.json_response({
                                "iterm_session_id": session.session_id,
                                "iterm_window_id": window.window_id,
                                "iterm_tab_id": tab.tab_id,
                            })
            return aiohttp.web.json_response({"iterm_session_id": None})

        else:
            return aiohttp.web.Response(text='Unknown action', status=400)

    except Exception as e:
        return aiohttp.web.Response(text=f'Error processing control request: {str(e)}',
                                    status=400)


async def handle_breakpoint(request, connection):
    """Handler for breakpoint information requests"""
    try:
        break_info = await get_all_panes_info(connection)
        return aiohttp.web.json_response(break_info)
    except Exception as e:
        return aiohttp.web.Response(text=f'Error getting breakpoint info: {str(e)}',
                                    status=500)


async def handle_screen_content(request, connection):
    """Handler for getting current screen content"""
    try:
        app = await iterm2.async_get_app(connection)
        window = app.current_terminal_window

        if not window:
            return aiohttp.web.json_response({'error': 'No active window'}, status=404)

        tab = window.current_tab
        if not tab:
            return aiohttp.web.json_response({'error': 'No active tab'}, status=404)

        session = tab.current_session
        if not session:
            return aiohttp.web.json_response({'error': 'No active session'}, status=404)

        # Get screen content
        line_info = await session.async_get_line_info()
        lines = await session.async_get_contents(
            first_line=line_info.first_visible_line_number,
            number_of_lines=line_info.mutable_area_height)

        # Reconstruct logical lines (handles line wrapping)
        logical_lines = reconstruct_logical_lines(lines)
        content = '\n'.join(logical_lines)

        return aiohttp.web.json_response({'content': content})

    except Exception as e:
        return aiohttp.web.json_response(
            {'error': f'Error getting screen content: {str(e)}'}, status=500)


async def main(connection):
    """Main function to set up and run the web server"""
    app = aiohttp.web.Application()

    # Create handlers with connection binding
    breakpoint_handler = partial(handle_breakpoint, connection=connection)
    send_code_handler = partial(handle_send_code, connection=connection)
    control_handler = partial(handle_control, connection=connection)
    screen_content_handler = partial(handle_screen_content, connection=connection)

    # Set up routes
    app.router.add_get('/breakpoint', breakpoint_handler)
    app.router.add_post('/send_code', send_code_handler)
    app.router.add_post('/control', control_handler)
    app.router.add_get('/screen_content', screen_content_handler)

    # Add CORS support for development
    app.router.add_options(
        '/{path:.*}', lambda request: aiohttp.web.Response(
            headers={
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            }))

    # Start the server
    runner = aiohttp.web.AppRunner(app)
    await runner.setup()
    site = aiohttp.web.TCPSite(runner, 'localhost', 17647)
    await site.start()
    print("Enhanced iTerm2 Web Server running on http://localhost:17647")
    print("Available endpoints:")
    print("  GET  /breakpoint - Get all pane information")
    print("  POST /send_code  - Send code to panes (supports broadcast and targeting)")
    print(
        "  POST /control    - Control operations (select_pane, new_pane, find_window_by_profile, find_active_window, create_tab, activate_tab, get_session_info)"
    )
    print("  GET  /screen_content - Get current pane screen content")

    # Keep the server running
    while True:
        await asyncio.sleep(3600)


# Run the enhanced server
if __name__ == "__main__":
    iterm2.run_until_complete(main)
