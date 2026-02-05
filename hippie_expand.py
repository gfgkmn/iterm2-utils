#!/usr/bin/env python3
"""Hippie-expand style completion for iTerm2.

Similar to Emacs' hippie-expand/dabbrev-expand:
- First invocation completes the current word with the nearest match
- Repeated invocations cycle through other matches
- Works by reading the current input line prefix and finding completions

Usage:
    Assign to a hotkey (e.g., Ctrl+/ or Alt+/)
    - First press: Complete current word with nearest match
    - Repeat: Cycle through other matches
    - Type anything else: Accept current completion
"""

import asyncio
import re
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import iterm2
import iterm2.screen

DEBUG = True
EXPANSION_TIMEOUT = 2.0   # Seconds before expansion state resets
INCLUDE_SCROLLBACK = True # Include scrollback buffer in completion source
MAX_SCROLLBACK_LINES = 300 # Maximum scrollback lines to scan
USE_EMACS_BUFFER = True   # Try to get completions from Emacs buffer
EMACSCLIENT_PATH = "/opt/homebrew/bin/emacsclient"  # Path to emacsclient


@dataclass
class ExpansionState:
    """Tracks the current expansion state across invocations."""
    prefix: str = ""
    completions: List[str] = field(default_factory=list)
    current_idx: int = 0
    inserted_text: str = ""
    last_time: float = 0.0
    session_id: str = ""


# State file for persistence between invocations
STATE_FILE = "/tmp/hippie_expand_state.json"


def load_expansion_state() -> ExpansionState:
    """Load expansion state from file."""
    import json
    import os
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                data = json.load(f)
                return ExpansionState(**data)
    except Exception as e:
        debug_print(f"Failed to load state: {e}")
    return ExpansionState()


def save_expansion_state(state: ExpansionState):
    """Save expansion state to file."""
    import json
    try:
        data = {
            'prefix': state.prefix,
            'completions': state.completions,
            'current_idx': state.current_idx,
            'inserted_text': state.inserted_text,
            'last_time': state.last_time,
            'session_id': state.session_id,
        }
        with open(STATE_FILE, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        debug_print(f"Failed to save state: {e}")


def debug_print(msg: str):
    if DEBUG:
        import sys
        print(f"[DEBUG] {msg}", file=sys.stderr, flush=True)


def get_emacs_buffer_content() -> Optional[str]:
    """Get current Emacs buffer content via emacsclient."""
    import shutil

    debug_print(f"USE_EMACS_BUFFER={USE_EMACS_BUFFER}")
    if not USE_EMACS_BUFFER:
        debug_print("Emacs buffer disabled, skipping")
        return None

    # Check if emacsclient exists
    emacsclient_full = shutil.which(EMACSCLIENT_PATH)
    debug_print(f"emacsclient path: {EMACSCLIENT_PATH} -> {emacsclient_full}")
    if not emacsclient_full:
        debug_print(f"ERROR: emacsclient not found in PATH")
        debug_print(f"PATH={subprocess.os.environ.get('PATH', 'NOT SET')}")
        return None

    # Get buffer from the selected window in the selected frame
    elisp = "(with-current-buffer (window-buffer (selected-window)) (buffer-substring-no-properties (point-min) (point-max)))"
    debug_print(f"Running: {emacsclient_full} --eval '{elisp[:50]}...'")

    try:
        result = subprocess.run(
            [emacsclient_full, "--eval", elisp],
            capture_output=True,
            text=True,
            timeout=2.0
        )
        debug_print(f"emacsclient returncode={result.returncode}")
        debug_print(f"emacsclient stdout length={len(result.stdout) if result.stdout else 0}")
        debug_print(f"emacsclient stderr={result.stderr[:200] if result.stderr else '(empty)'}")

        if result.returncode != 0:
            debug_print(f"ERROR: emacsclient failed with code {result.returncode}")
            return None

        if not result.stdout or result.stdout.strip() == '""':
            debug_print("WARNING: emacsclient returned empty buffer")
            return None

        # emacsclient returns quoted string, need to unescape
        content = result.stdout.strip()
        debug_print(f"Raw output (first 200 chars): {repr(content[:200])}")
        starts_quote = content.startswith('"')
        ends_quote = content.endswith('"')
        debug_print(f"Starts with quote: {starts_quote}, Ends with quote: {ends_quote}")

        # Remove surrounding quotes
        if content.startswith('"') and content.endswith('"'):
            content = content[1:-1]
            debug_print("Stripped quotes")
        else:
            debug_print("WARNING: No quotes to strip!")

        # Unescape common escape sequences
        content = content.replace('\\n', '\n')
        content = content.replace('\\t', '\t')
        content = content.replace('\\"', '"')
        content = content.replace('\\\\', '\\')

        lines = content.split('\n')
        debug_print(f"After unescape: {len(content)} chars, {len(lines)} lines")
        debug_print(f"First 3 lines: {lines[:3]}")
        debug_print(f"Last line: {repr(lines[-1])}")
        return content

    except subprocess.TimeoutExpired:
        debug_print("ERROR: emacsclient timed out after 2s")
    except FileNotFoundError as e:
        debug_print(f"ERROR: FileNotFoundError: {e}")
    except Exception as e:
        debug_print(f"ERROR: Exception: {type(e).__name__}: {e}")

    return None


async def get_screen_content(session) -> Tuple[List[iterm2.screen.LineContents], int, int]:
    """Get the visible screen content and cursor position.

    Returns:
        (lines, cursor_row, cursor_col) - cursor position is screen-relative
    """
    contents = await session.async_get_screen_contents()
    line_info = await session.async_get_line_info()

    lines = []
    for i in range(contents.number_of_lines):
        lines.append(contents.line(i))

    # Get cursor position - need to adjust for scrollback offset
    cursor_row, cursor_col = -1, -1
    if hasattr(contents, 'cursor_coord') and contents.cursor_coord:
        abs_cursor_row = contents.cursor_coord.y
        cursor_col = contents.cursor_coord.x

        # cursor_coord.y is absolute (includes scrollback), convert to screen-relative
        first_visible = line_info.first_visible_line_number
        cursor_row = abs_cursor_row - first_visible

        debug_print(f"Cursor: abs_row={abs_cursor_row}, first_visible={first_visible}, relative_row={cursor_row}, col={cursor_col}")

    return lines, cursor_row, cursor_col


def line_to_string(line: iterm2.screen.LineContents) -> str:
    """Convert LineContents to string."""
    chars = []
    col = 0
    while True:
        try:
            cell = line.string_at(col)
            chars.append(cell if cell else ' ')
            col += 1
        except IndexError:
            break
    return ''.join(chars)


async def get_scrollback_text(session) -> List[str]:
    """Get text from scrollback buffer."""
    if not INCLUDE_SCROLLBACK:
        return []

    try:
        line_info = await session.async_get_line_info()
        scrollback_height = line_info.scrollback_buffer_height

        if scrollback_height == 0:
            return []

        lines_to_read = min(scrollback_height, MAX_SCROLLBACK_LINES)
        start_line = max(0, scrollback_height - lines_to_read)

        contents = await session.async_get_contents(start_line, lines_to_read)

        text_lines = []
        for i in range(contents.number_of_lines):
            line = contents.line(i)
            text_lines.append(line_to_string(line))

        debug_print(f" Read {len(text_lines)} lines from scrollback")
        return text_lines

    except Exception as e:
        debug_print(f" Error reading scrollback: {e}")
        return []


def get_cursor_position(lines: List[iterm2.screen.LineContents]) -> Tuple[int, int]:
    """Estimate cursor position (last non-empty position on last line with content)."""
    # Look for the last line that appears to have a prompt/input
    for row in range(len(lines) - 1, -1, -1):
        line_text = line_to_string(lines[row]).rstrip()
        if line_text:
            # Cursor is at end of this line's content
            return row, len(line_text)
    return len(lines) - 1, 0


def extract_word_prefix(lines: List[iterm2.screen.LineContents],
                        cursor_row: int, cursor_col: int) -> str:
    """Extract the word prefix at cursor position."""
    if cursor_row >= len(lines):
        return ""

    line_text = line_to_string(lines[cursor_row])

    # Find word boundary going backwards from cursor
    end = min(cursor_col, len(line_text))
    start = end

    word_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.')

    while start > 0 and line_text[start - 1] in word_chars:
        start -= 1

    prefix = line_text[start:end]
    debug_print(f" Extracted prefix: '{prefix}' from position ({cursor_row}, {start}:{end})")
    return prefix


def find_word_completions(lines: List[iterm2.screen.LineContents],
                          prefix: str,
                          cursor_row: int,
                          scrollback_text: Optional[List[str]] = None,
                          emacs_content: Optional[str] = None) -> List[str]:
    """Find all words matching prefix, ordered by proximity to cursor."""
    if not prefix:
        return []

    word_pattern = re.compile(r'[a-zA-Z_][a-zA-Z0-9_\-\.]*')
    prefix_lower = prefix.lower()

    # Collect matches with their distance from cursor
    matches: List[Tuple[str, int]] = []
    seen: Set[str] = set()
    seen.add(prefix)  # Don't include the prefix itself

    # Search Emacs buffer first (highest priority)
    if emacs_content:
        emacs_lines = emacs_content.split('\n')
        debug_print(f"Emacs buffer has {len(emacs_lines)} lines")
        all_emacs_words = []
        for row, line_text in enumerate(emacs_lines):
            for match in word_pattern.finditer(line_text):
                word = match.group()
                all_emacs_words.append(word)
                if word.lower().startswith(prefix_lower) and word not in seen:
                    seen.add(word)
                    # Emacs buffer gets highest priority (negative distance)
                    distance = -10000 + row * 10 + match.start()
                    matches.append((word, distance))
        debug_print(f"Emacs words found: {len(all_emacs_words)}, matching '{prefix}': {[w for w in all_emacs_words if w.lower().startswith(prefix_lower)][:10]}")

    # Search visible screen (higher priority = smaller distance)
    for row, line in enumerate(lines):
        line_text = line_to_string(line)
        for match in word_pattern.finditer(line_text):
            word = match.group()
            if word.lower().startswith(prefix_lower) and word not in seen:
                seen.add(word)
                # Distance: prefer words on same line, then nearby lines
                distance = abs(row - cursor_row) * 1000 + match.start()
                matches.append((word, distance))

    # Search scrollback (lower priority = larger base distance)
    if scrollback_text:
        scrollback_base = len(lines) * 1000  # Start after all visible lines
        for row, line_text in enumerate(scrollback_text):
            for match in word_pattern.finditer(line_text):
                word = match.group()
                if word.lower().startswith(prefix_lower) and word not in seen:
                    seen.add(word)
                    # Scrollback is further away, most recent first
                    distance = scrollback_base + (len(scrollback_text) - row) * 1000 + match.start()
                    matches.append((word, distance))

    # Sort by distance (nearest first)
    matches.sort(key=lambda x: x[1])

    debug_print(f"Found {len(matches)} completions for '{prefix}'")
    return [m[0] for m in matches]


async def hippie_expand(connection, session):
    """Perform hippie-expand style completion."""
    import time
    current_time = time.time()
    session_id = session.session_id

    # Load persisted state from file (for cycling across invocations)
    _expansion_state = load_expansion_state()
    debug_print(f"Loaded state: prefix='{_expansion_state.prefix}', idx={_expansion_state.current_idx}, last_time={_expansion_state.last_time}")

    # Get screen content, cursor position, and scrollback
    lines, cursor_row, cursor_col = await get_screen_content(session)
    scrollback_text = await get_scrollback_text(session)

    # Fallback if cursor position not available from API
    if cursor_row < 0 or cursor_col < 0:
        debug_print("Cursor position not available from API, using heuristic")
        cursor_row, cursor_col = get_cursor_position(lines)

    debug_print(f"Screen: {len(lines)} lines, cursor at ({cursor_row}, {cursor_col})")

    # Check if this is a continuation of previous expansion
    is_continuation = (
        _expansion_state.session_id == session_id and
        _expansion_state.completions and
        (current_time - _expansion_state.last_time) < EXPANSION_TIMEOUT
    )
    debug_print(f"is_continuation={is_continuation}, time_diff={current_time - _expansion_state.last_time:.2f}s")

    if is_continuation:
        debug_print("Continuing previous expansion")
        # Cycle to next completion
        _expansion_state.current_idx = (
            (_expansion_state.current_idx + 1) % len(_expansion_state.completions)
        )

        # Delete previous insertion
        if _expansion_state.inserted_text:
            delete_count = len(_expansion_state.inserted_text)
            backspaces = '\x7f' * delete_count  # DEL characters
            # Use async_send_text to send as USER INPUT
            await session.async_send_text(backspaces)

        # Insert new completion (suffix only)
        completion = _expansion_state.completions[_expansion_state.current_idx]
        suffix = completion[len(_expansion_state.prefix):]
        _expansion_state.inserted_text = suffix
        _expansion_state.last_time = current_time

        if suffix:
            # Use async_send_text to send as USER INPUT
            await session.async_send_text(suffix)

        debug_print(f"Cycled to: '{completion}' (idx={_expansion_state.current_idx})")

    else:
        debug_print("Starting new expansion")
        # New expansion
        prefix = extract_word_prefix(lines, cursor_row, cursor_col)

        if not prefix:
            debug_print("No prefix found")
            return

        # Get Emacs buffer content
        emacs_content = get_emacs_buffer_content()

        completions = find_word_completions(lines, prefix, cursor_row, scrollback_text, emacs_content)

        if not completions:
            debug_print("No completions found")
            return

        # Initialize state
        _expansion_state = ExpansionState(
            prefix=prefix,
            completions=completions,
            current_idx=0,
            inserted_text="",
            last_time=current_time,
            session_id=session_id
        )

        # Insert first completion (suffix only)
        completion = completions[0]
        suffix = completion[len(prefix):]
        _expansion_state.inserted_text = suffix

        if suffix:
            # Use async_send_text to send as USER INPUT
            await session.async_send_text(suffix)

        debug_print(f"First expansion: '{completion}'")

    # Save state for next invocation
    save_expansion_state(_expansion_state)
    debug_print(f"Saved state: prefix='{_expansion_state.prefix}', idx={_expansion_state.current_idx}")


async def main(connection):
    """Main entry point."""
    app = await iterm2.async_get_app(connection)

    window = app.current_terminal_window
    if not window:
        return

    session = window.current_tab.current_session
    if not session:
        return

    await hippie_expand(connection, session)


iterm2.run_until_complete(main)
