#!/usr/bin/env python3
"""Screen content completion for iTerm2.

Provides word and line completion using visible terminal content as the source,
similar to Emacs' hippie-expand/dabbrev-expand functionality.

Usage:
    - Trigger via iTerm2 hotkey
    - Type to filter completions
    - Use Up/Down or Tab/Shift-Tab to navigate
    - Enter to select, Escape to cancel
"""

import asyncio
import re
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import iterm2
import iterm2.screen

# Configuration
DEBUG = True              # Enable debug logging
DEBUG_LOG_FILE = "/tmp/screen_complete_debug.log"  # Log file path
MAX_COMPLETIONS = 20      # Maximum completions to show
MIN_WORD_LENGTH = 2       # Minimum word length to consider
INCLUDE_SCROLLBACK = True # Include scrollback buffer in completion source
MAX_SCROLLBACK_LINES = 500 # Maximum scrollback lines to scan
USE_EMACS_BUFFER = True   # Try to get completions from Emacs buffer
EMACSCLIENT_PATH = "/opt/homebrew/bin/emacsclient"  # Path to emacsclient


class CompletionMode(Enum):
    WORD = "word"
    LINE = "line"


@dataclass
class Completion:
    """A completion candidate."""
    text: str
    source_row: int  # Row where this completion was found
    source_col: int  # Column where this completion starts
    score: int = 0   # Match score for ranking


def debug_print(msg: str):
    """Print debug message if DEBUG is enabled."""
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
        debug_print(f"Raw output (first 100 chars): {content[:100]}")

        # Remove surrounding quotes
        if content.startswith('"') and content.endswith('"'):
            content = content[1:-1]
        # Unescape common escape sequences
        content = content.replace('\\n', '\n')
        content = content.replace('\\t', '\t')
        content = content.replace('\\"', '"')
        content = content.replace('\\\\', '\\')
        debug_print(f"Got {len(content)} chars from Emacs buffer")
        return content

    except subprocess.TimeoutExpired:
        debug_print("ERROR: emacsclient timed out after 2s")
    except FileNotFoundError as e:
        debug_print(f"ERROR: FileNotFoundError: {e}")
    except Exception as e:
        debug_print(f"ERROR: Exception: {type(e).__name__}: {e}")

    return None


async def should_use_alt_screen(session) -> bool:
    """Check if it's safe to use alternate screen buffer."""
    try:
        term = await session.async_get_variable("user.TERM")
        if term and any(x in term.lower() for x in ["tmux", "screen"]):
            return False
    except Exception:
        pass

    try:
        job = await session.async_get_variable("jobName")
        if job:
            job_lower = job.lower()
            if "tmux" in job_lower or job_lower == "ssh":
                return False
    except Exception:
        pass

    return True


async def get_screen_content(session) -> Tuple[List[iterm2.screen.LineContents], int, Tuple[int, int]]:
    """Get the visible screen content from the session.

    Returns:
        (lines, mutable_area_start, (cursor_row, cursor_col))
    """
    contents = await session.async_get_screen_contents()
    line_info = await session.async_get_line_info()
    lines: List[iterm2.screen.LineContents] = []
    for i in range(contents.number_of_lines):
        lines.append(contents.line(i))

    mutable_area_start = line_info.overflow + line_info.scrollback_buffer_height

    # Get cursor position - need to adjust for scrollback offset
    cursor_row, cursor_col = -1, -1
    if hasattr(contents, 'cursor_coord') and contents.cursor_coord:
        abs_cursor_row = contents.cursor_coord.y
        cursor_col = contents.cursor_coord.x

        # cursor_coord.y is absolute (includes scrollback), convert to screen-relative
        # first_visible_line_number is the absolute line number of the first visible line
        first_visible = line_info.first_visible_line_number
        cursor_row = abs_cursor_row - first_visible

        debug_print(f"Cursor: abs_row={abs_cursor_row}, first_visible={first_visible}, relative_row={cursor_row}, col={cursor_col}")

    return lines, mutable_area_start, (cursor_row, cursor_col)


async def get_scrollback_text(session) -> List[str]:
    """Get text from scrollback buffer."""
    if not INCLUDE_SCROLLBACK:
        return []

    try:
        line_info = await session.async_get_line_info()
        scrollback_height = line_info.scrollback_buffer_height

        if scrollback_height == 0:
            return []

        # Limit how much scrollback we read
        lines_to_read = min(scrollback_height, MAX_SCROLLBACK_LINES)
        start_line = max(0, scrollback_height - lines_to_read)

        # Read scrollback range
        contents = await session.async_get_contents(
            start_line,
            lines_to_read
        )

        text_lines = []
        for i in range(contents.number_of_lines):
            line = contents.line(i)
            text_lines.append(line_to_string(line))

        debug_print(f"[DEBUG] Read {len(text_lines)} lines from scrollback")
        return text_lines

    except Exception as e:
        debug_print(f"[DEBUG] Error reading scrollback: {e}")
        return []


def line_to_string(line: iterm2.screen.LineContents) -> str:
    """Convert a LineContents to a string."""
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


def extract_words(lines: List[iterm2.screen.LineContents],
                  scrollback_text: Optional[List[str]] = None,
                  emacs_content: Optional[str] = None) -> List[Completion]:
    """Extract unique words from screen content, scrollback, and Emacs buffer."""
    word_pattern = re.compile(r'[a-zA-Z_][a-zA-Z0-9_\-\.]*')
    seen: Set[str] = set()
    completions: List[Completion] = []

    # Extract from Emacs buffer first (highest priority)
    if emacs_content:
        emacs_lines = emacs_content.split('\n')
        debug_print(f"Emacs buffer has {len(emacs_lines)} lines")
        emacs_word_count = 0
        for row, line_text in enumerate(emacs_lines):
            for match in word_pattern.finditer(line_text):
                word = match.group()
                emacs_word_count += 1
                if len(word) >= MIN_WORD_LENGTH and word not in seen:
                    seen.add(word)
                    completions.append(Completion(
                        text=word,
                        source_row=-10000 + row,  # Very high priority
                        source_col=match.start()
                    ))
        debug_print(f"Emacs words extracted: {emacs_word_count} total, {len([c for c in completions if c.source_row < -1000])} unique")

    # Extract from visible screen
    for row, line in enumerate(lines):
        line_text = line_to_string(line)
        for match in word_pattern.finditer(line_text):
            word = match.group()
            if len(word) >= MIN_WORD_LENGTH and word not in seen:
                seen.add(word)
                completions.append(Completion(
                    text=word,
                    source_row=row,
                    source_col=match.start()
                ))

    # Extract from scrollback (lower priority, negative row numbers)
    if scrollback_text:
        for row, line_text in enumerate(scrollback_text):
            for match in word_pattern.finditer(line_text):
                word = match.group()
                if len(word) >= MIN_WORD_LENGTH and word not in seen:
                    seen.add(word)
                    completions.append(Completion(
                        text=word,
                        source_row=-(len(scrollback_text) - row),
                        source_col=match.start()
                    ))

    return completions


def extract_lines(lines: List[iterm2.screen.LineContents],
                  scrollback_text: Optional[List[str]] = None,
                  emacs_content: Optional[str] = None) -> List[Completion]:
    """Extract non-empty lines from screen content, scrollback, and Emacs buffer."""
    seen: Set[str] = set()
    completions: List[Completion] = []

    # Extract from Emacs buffer first (highest priority)
    if emacs_content:
        for row, line_text in enumerate(emacs_content.split('\n')):
            line_text = line_text.rstrip()
            if line_text and line_text not in seen:
                seen.add(line_text)
                completions.append(Completion(
                    text=line_text,
                    source_row=-10000 + row,
                    source_col=0
                ))

    # Extract from visible screen
    for row, line in enumerate(lines):
        line_text = line_to_string(line).rstrip()
        if line_text and line_text not in seen:
            seen.add(line_text)
            completions.append(Completion(
                text=line_text,
                source_row=row,
                source_col=0
            ))

    # Extract from scrollback
    if scrollback_text:
        for row, line_text in enumerate(scrollback_text):
            line_text = line_text.rstrip()
            if line_text and line_text not in seen:
                seen.add(line_text)
                completions.append(Completion(
                    text=line_text,
                    source_row=-(len(scrollback_text) - row),
                    source_col=0
                ))

    return completions


def fuzzy_match(pattern: str, text: str) -> Tuple[bool, int]:
    """Fuzzy match pattern against text.

    Returns (matches, score) where higher score is better.
    """
    if not pattern:
        return True, 0

    pattern_lower = pattern.lower()
    text_lower = text.lower()

    # Exact prefix match gets highest score
    if text_lower.startswith(pattern_lower):
        return True, 1000 + len(pattern)

    # Contains match
    if pattern_lower in text_lower:
        # Earlier position = higher score
        pos = text_lower.index(pattern_lower)
        return True, 500 - pos

    # Fuzzy subsequence match
    pattern_idx = 0
    score = 0
    consecutive = 0

    for i, char in enumerate(text_lower):
        if pattern_idx < len(pattern_lower) and char == pattern_lower[pattern_idx]:
            pattern_idx += 1
            consecutive += 1
            score += consecutive * 10  # Bonus for consecutive matches
            if i == 0:
                score += 50  # Bonus for matching start
        else:
            consecutive = 0

    if pattern_idx == len(pattern_lower):
        return True, score

    return False, 0


def filter_completions(completions: List[Completion],
                       filter_text: str) -> List[Completion]:
    """Filter and rank completions based on filter text."""
    if not filter_text:
        return completions[:MAX_COMPLETIONS]

    matches: List[Tuple[Completion, int]] = []
    for comp in completions:
        is_match, score = fuzzy_match(filter_text, comp.text)
        if is_match:
            comp.score = score
            matches.append((comp, score))

    # Sort by score descending
    matches.sort(key=lambda x: x[1], reverse=True)
    return [m[0] for m in matches[:MAX_COMPLETIONS]]


def _line_cell_count(line: iterm2.screen.LineContents) -> int:
    """Get the number of cells in a line."""
    offsets = getattr(line, "_LineContents__offset_of_cell", None)
    if offsets is not None:
        return max(0, len(offsets) - 1)

    count = 0
    while True:
        try:
            _ = line.string_at(count)
        except IndexError:
            break
        count += 1
    return count


def _style_to_sgr(style, extra: Optional[List[str]] = None) -> str:
    """Convert cell style to SGR escape sequence."""
    codes: List[str] = ["0"]

    if style:
        if style.bold:
            codes.append("1")
        if style.faint:
            codes.append("2")
        if style.italic:
            codes.append("3")
        if style.underline:
            codes.append("4")
        if style.inverse:
            codes.append("7")

    if extra:
        codes.extend(extra)
    return f"\033[{';'.join(codes)}m"


def build_completion_menu(completions: List[Completion],
                         selected_idx: int,
                         filter_text: str,
                         screen_height: int,
                         screen_width: int,
                         mode: CompletionMode,
                         cursor_row: int = -1,
                         cursor_col: int = -1) -> Tuple[str, Tuple[int, int, int, int]]:
    """Build the completion menu overlay as escape sequences.

    Returns:
        (menu_string, (start_row, start_col, menu_width, menu_height))
        The coordinates are needed for proper screen restoration.
    """
    if not completions:
        return "", (0, 0, 0, 0)

    parts: List[str] = ["\0337"]  # Save cursor

    # Menu dimensions
    menu_height = min(len(completions) + 5, screen_height - 2)  # +5 for header/filter/separator/footer
    menu_width = min(max(len(c.text) for c in completions) + 6, screen_width - 4)
    menu_width = max(menu_width, 30)  # Minimum width

    # Position menu near cursor if available
    if cursor_row >= 0 and cursor_col >= 0:
        # Try to position menu just below the cursor
        if cursor_row + menu_height + 1 <= screen_height:
            # Enough space below cursor
            start_row = cursor_row + 2  # 1-indexed, +1 below cursor
        else:
            # Position above cursor
            start_row = max(1, cursor_row - menu_height)

        # Horizontal: try to start at cursor, but ensure it fits
        start_col = max(1, min(cursor_col, screen_width - menu_width - 1))
    else:
        # Fallback: position at bottom-right
        start_row = screen_height - menu_height
        start_col = screen_width - menu_width - 2

    # Draw menu box
    mode_label = "Word" if mode == CompletionMode.WORD else "Line"
    header = f" {mode_label} Completion ({len(completions)}) "

    # Top border
    parts.append(f"\033[{start_row};{start_col}H")
    parts.append("\033[1;37;44m")  # Bold white on blue
    parts.append("╭" + "─" * (menu_width - 2) + "╮")

    # Header row
    parts.append(f"\033[{start_row + 1};{start_col}H")
    parts.append("│")
    parts.append(f"\033[1;33m")  # Bold yellow
    parts.append(header.center(menu_width - 2))
    parts.append("\033[1;37;44m│")

    # Filter row
    parts.append(f"\033[{start_row + 2};{start_col}H")
    parts.append("│")
    parts.append("\033[0;37;44m")  # Normal white on blue
    filter_display = f" > {filter_text}_" if filter_text else " > _"
    parts.append(filter_display.ljust(menu_width - 2))
    parts.append("\033[1;37;44m│")

    # Separator
    parts.append(f"\033[{start_row + 3};{start_col}H")
    parts.append("├" + "─" * (menu_width - 2) + "┤")

    # Completion items
    visible_items = menu_height - 5  # Subtract header, filter, separator, bottom
    for i, comp in enumerate(completions[:visible_items]):
        row = start_row + 4 + i
        parts.append(f"\033[{row};{start_col}H")
        parts.append("│")

        # Truncate text if needed
        display_text = comp.text
        if len(display_text) > menu_width - 4:
            display_text = display_text[:menu_width - 7] + "..."

        if i == selected_idx:
            parts.append("\033[1;30;47m")  # Bold black on white (selected)
        else:
            parts.append("\033[0;37;44m")  # Normal white on blue

        parts.append(f" {display_text}".ljust(menu_width - 2))
        parts.append("\033[1;37;44m│")

    # Bottom border
    bottom_row = start_row + menu_height - 1
    parts.append(f"\033[{bottom_row};{start_col}H")
    parts.append("╰" + "─" * (menu_width - 2) + "╯")

    parts.append("\033[0m\0338")  # Reset and restore cursor
    return ''.join(parts), (start_row, start_col, menu_width, menu_height)


def build_screen_restore(lines: List[iterm2.screen.LineContents],
                        menu_pos: Tuple[int, int, int, int]) -> str:
    """Build sequence to restore screen area under menu.

    Args:
        lines: Original screen content
        menu_pos: (start_row, start_col, menu_width, menu_height) from build_completion_menu
    """
    start_row, start_col, menu_width, menu_height = menu_pos
    parts: List[str] = ["\0337"]  # Save cursor

    # Restore the exact area where the menu was drawn
    for row_offset in range(menu_height):
        row = start_row + row_offset - 1  # start_row is 1-indexed, lines[] is 0-indexed
        if row < 0 or row >= len(lines):
            continue
        line = lines[row]

        # Position cursor at the start of this row's menu area
        parts.append(f"\033[{start_row + row_offset};{start_col}H")

        # Restore each cell in the menu width
        for col_offset in range(menu_width):
            col = start_col + col_offset - 1  # start_col is 1-indexed
            try:
                cell = line.string_at(col)
                style = None
                if hasattr(line, 'style_at'):
                    try:
                        style = line.style_at(col)
                    except Exception:
                        pass
                parts.append(_style_to_sgr(style))
                parts.append(cell if cell else ' ')
            except IndexError:
                parts.append("\033[0m ")

    parts.append("\033[0m\0338")
    return ''.join(parts)


def extract_word_prefix(lines: List[iterm2.screen.LineContents],
                        cursor_row: int, cursor_col: int) -> str:
    """Extract the word prefix at cursor position."""
    if cursor_row < 0 or cursor_row >= len(lines):
        debug_print(f"extract_word_prefix: invalid cursor_row={cursor_row}, lines={len(lines)}")
        return ""

    line_text = line_to_string(lines[cursor_row])

    # Debug: show full line and cursor position to help diagnose ghost text issues
    debug_print(f"Line at cursor (row={cursor_row}): '{line_text}'")
    debug_print(f"Cursor col={cursor_col}, line length={len(line_text)}")

    # Find word boundary going backwards from cursor
    end = min(cursor_col, len(line_text))
    start = end

    word_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.')

    while start > 0 and line_text[start - 1] in word_chars:
        start -= 1

    prefix = line_text[start:end]
    # Also show what's AFTER cursor (might be ghost text from zsh-autosuggestions)
    after_cursor = line_text[end:end+20] if end < len(line_text) else "(end of line)"
    debug_print(f"Extracted prefix: '{prefix}' at ({cursor_row}, {start}:{end}), after cursor: '{after_cursor}'")
    return prefix


def extract_line_prefix(lines: List[iterm2.screen.LineContents],
                        cursor_row: int, cursor_col: int) -> str:
    """Extract the line prefix (everything from line start to cursor) for LINE mode."""
    if cursor_row < 0 or cursor_row >= len(lines):
        debug_print(f"extract_line_prefix: invalid cursor_row={cursor_row}, lines={len(lines)}")
        return ""

    line_text = line_to_string(lines[cursor_row])

    # Extract from start of line to cursor position
    end = min(cursor_col, len(line_text))
    prefix = line_text[:end].lstrip()  # Strip leading whitespace for better matching

    debug_print(f"Extracted line prefix: '{prefix}' at ({cursor_row}, 0:{end})")
    return prefix


def _create_completion_key_pattern() -> iterm2.KeystrokePattern:
    """Create pattern for completion navigation keys."""
    pattern = iterm2.KeystrokePattern()
    pattern.characters = list(
        'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        '_-./:@#$%^&*()+=[]{}|\\;\'",<>?`~! '
    )
    pattern.keycodes = [
        iterm2.Keycode.ESCAPE,
        iterm2.Keycode.RETURN,
        iterm2.Keycode.TAB,
        iterm2.Keycode.UP_ARROW,
        iterm2.Keycode.DOWN_ARROW,
        iterm2.Keycode.DELETE,  # Backspace key
    ]
    return pattern


async def complete_interactive(connection, session, mode: CompletionMode):
    """Interactive completion using screen content."""
    session_id = session.session_id
    use_alt_screen = await should_use_alt_screen(session)

    # Get screen content, scrollback, and cursor position
    debug_print("=== Starting completion ===")
    lines, _, (cursor_row, cursor_col) = await get_screen_content(session)
    debug_print(f"Screen: {len(lines)} lines, cursor at ({cursor_row}, {cursor_col})")

    scrollback_text = await get_scrollback_text(session)
    debug_print(f"Scrollback: {len(scrollback_text)} lines")

    # Get Emacs buffer content
    debug_print("Fetching Emacs buffer...")
    emacs_content = get_emacs_buffer_content()
    debug_print(f"Emacs content: {len(emacs_content) if emacs_content else 0} chars")

    screen_height = len(lines)
    screen_width = 80  # Default, we'll estimate from content

    if lines:
        first_line_width = _line_cell_count(lines[0])
        if first_line_width > 0:
            screen_width = first_line_width

    # Extract completions based on mode
    if mode == CompletionMode.WORD:
        all_completions = extract_words(lines, scrollback_text, emacs_content)
    else:
        all_completions = extract_lines(lines, scrollback_text, emacs_content)

    if not all_completions:
        debug_print("No completions found")
        return

    # Count sources
    emacs_count = sum(1 for c in all_completions if c.source_row < -1000)
    screen_count = sum(1 for c in all_completions if c.source_row >= 0)
    scrollback_count = sum(1 for c in all_completions if -1000 <= c.source_row < 0)
    debug_print(f"Completions: {len(all_completions)} total (emacs={emacs_count}, screen={screen_count}, scrollback={scrollback_count})")
    if all_completions:
        debug_print(f"First 5 completions: {[c.text for c in all_completions[:5]]}")

    # Extract prefix at cursor for pre-filtering
    initial_prefix = ""
    if cursor_row >= 0 and cursor_col >= 0:
        if mode == CompletionMode.WORD:
            initial_prefix = extract_word_prefix(lines, cursor_row, cursor_col)
        else:
            # LINE mode: extract everything from line start to cursor
            initial_prefix = extract_line_prefix(lines, cursor_row, cursor_col)
        debug_print(f"Initial prefix from cursor: '{initial_prefix}'")

    # Track how much to skip when inserting (length of prefix already typed)
    prefix_len = len(initial_prefix)

    filter_pattern = _create_completion_key_pattern()

    # Track what we'll inject after menu closes
    result: Optional[str] = None

    async with iterm2.KeystrokeFilter(connection, [filter_pattern], session_id):
        async with iterm2.KeystrokeMonitor(connection, session_id) as mon:
            filter_text = initial_prefix  # Start with prefix already typed
            selected_idx = 0
            prev_menu_height = 0  # Track for clearing
            menu_pos = (0, 0, 0, 0)  # Track menu position for restore

            # Switch to alternate screen if available
            if use_alt_screen:
                await session.async_inject(b'\033[?1049h')
                # Redraw original screen
                base_parts = ["\0337"]
                for row, line in enumerate(lines):
                    for col in range(_line_cell_count(line)):
                        try:
                            cell = line.string_at(col)
                            base_parts.append(f"\033[{row + 1};{col + 1}H")
                            base_parts.append(cell if cell else ' ')
                        except IndexError:
                            break
                base_parts.append("\0338")
                await session.async_inject(''.join(base_parts).encode('utf-8'))

            try:
                while True:
                    # Filter completions
                    filtered = filter_completions(all_completions, filter_text)

                    # Calculate current menu height for clearing
                    curr_menu_height = min(len(filtered) + 5, screen_height - 2) if filtered else 6

                    # Clear previous menu area if size changed (prevents overlap)
                    if prev_menu_height > curr_menu_height and not use_alt_screen:
                        clear_parts = ["\0337"]  # Save cursor
                        for row in range(screen_height - prev_menu_height, screen_height - curr_menu_height):
                            clear_parts.append(f"\033[{row + 1};1H\033[2K")  # Clear line
                        clear_parts.append("\0338")  # Restore cursor
                        await session.async_inject(''.join(clear_parts).encode('utf-8'))
                    prev_menu_height = curr_menu_height

                    if not filtered:
                        # No matches, show empty state
                        menu, menu_pos = build_completion_menu(
                            [], 0, filter_text, screen_height, screen_width, mode,
                            cursor_row, cursor_col
                        )
                    else:
                        # Clamp selection
                        selected_idx = min(selected_idx, len(filtered) - 1)
                        selected_idx = max(0, selected_idx)

                        menu, menu_pos = build_completion_menu(
                            filtered, selected_idx, filter_text,
                            screen_height, screen_width, mode,
                            cursor_row, cursor_col
                        )

                    await session.async_inject(menu.encode('utf-8'))

                    # Get keystroke
                    keystroke = await mon.async_get()

                    if keystroke.keycode == iterm2.Keycode.ESCAPE:
                        # Cancel
                        break

                    elif keystroke.keycode == iterm2.Keycode.RETURN:
                        # Select current completion
                        if filtered and 0 <= selected_idx < len(filtered):
                            result = filtered[selected_idx].text
                        break

                    elif keystroke.keycode == iterm2.Keycode.TAB:
                        # Navigate down (or up with shift)
                        if keystroke.modifiers and iterm2.Modifier.SHIFT in keystroke.modifiers:
                            selected_idx = max(0, selected_idx - 1)
                        else:
                            if filtered:
                                selected_idx = (selected_idx + 1) % len(filtered)

                    elif keystroke.keycode == iterm2.Keycode.UP_ARROW:
                        selected_idx = max(0, selected_idx - 1)

                    elif keystroke.keycode == iterm2.Keycode.DOWN_ARROW:
                        if filtered:
                            selected_idx = min(len(filtered) - 1, selected_idx + 1)

                    elif keystroke.keycode == iterm2.Keycode.DELETE:
                        if filter_text:
                            filter_text = filter_text[:-1]
                            selected_idx = 0

                    else:
                        # Add character to filter
                        char = keystroke.characters
                        if char:
                            filter_text += char
                            selected_idx = 0

            finally:
                # Restore screen
                if use_alt_screen:
                    await session.async_inject(b'\033[?1049l')
                else:
                    # Direct restore for tmux - use actual menu position
                    restore = build_screen_restore(lines, menu_pos)
                    await session.async_inject(restore.encode('utf-8'))

    # Insert the selected completion OUTSIDE the KeystrokeFilter context
    # This ensures the injection goes directly to the terminal without interception
    if result:
        # Only insert the part after what was already typed
        suffix = result[prefix_len:] if len(result) > prefix_len else result
        debug_print(f"Inserting suffix: '{suffix}' (full: '{result}', prefix_len: {prefix_len}, initial_prefix: '{initial_prefix}')")
        if suffix:
            # Small delay to ensure terminal is ready after screen restore
            await asyncio.sleep(0.05)
            # Use async_send_text to send as USER INPUT (not async_inject which sends as terminal output)
            await session.async_send_text(suffix)


async def main(connection):
    """Main entry point."""
    app = await iterm2.async_get_app(connection)

    window = app.current_terminal_window
    if not window:
        return

    session = window.current_tab.current_session
    if not session:
        return

    # Default to word completion
    # You can change this or make it configurable
    await complete_interactive(connection, session, CompletionMode.WORD)


if __name__ == "__main__":
    iterm2.run_until_complete(main)
