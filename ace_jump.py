#!/usr/bin/env python3

import asyncio
from typing import Dict, Iterable, List, Optional, Tuple

import iterm2
import iterm2.screen


async def should_use_alt_screen(session) -> bool:
    """Check if it's safe to use alternate screen buffer.

    Returns False if in tmux or other multiplexer (use direct overlay instead).
    """
    try:
        # Check TERM variable - tmux/screen set specific values
        term = await session.async_get_variable("user.TERM")
        if term:
            term_lower = term.lower()
            # tmux and screen use these TERM values
            if any(x in term_lower for x in ["tmux", "screen"]):
                return False
    except Exception:
        pass

    try:
        # Check session name for tmux indicators
        name = await session.async_get_variable("session.name")
        if name and "tmux" in name.lower():
            return False
    except Exception:
        pass

    try:
        # Check job name - tmux shows up here
        job = await session.async_get_variable("jobName")
        if job and "tmux" in job.lower():
            return False
    except Exception:
        pass

    try:
        # Check TTY for pts which might indicate nested session
        tty = await session.async_get_variable("session.tty")
        if tty:
            # Additional heuristic could go here
            pass
    except Exception:
        pass

    # Default to using alternate screen if no multiplexer detected
    return True

# QWERTY keyboard order for hints (home row first, then numbers)
# Extended set for more single-char hints
HINT_CHARS = 'asdfghjklqwertyuiopzxcvbnm1234567890'


def generate_hints(count: int) -> List[str]:
    """Generate hint labels in QWERTY order.

    Uses single chars when possible, multi-char when needed.
    """
    if count == 0:
        return []

    base = len(HINT_CHARS)

    # Calculate minimum length needed
    length = 1
    while base ** length < count:
        length += 1

    final_hints: List[str] = []

    def generate_fixed_length(prefix: str, remaining_length: int) -> bool:
        if remaining_length == 0:
            final_hints.append(prefix)
            return len(final_hints) < count

        for char in HINT_CHARS:
            if not generate_fixed_length(prefix + char, remaining_length - 1):
                return False
        return True

    generate_fixed_length('', length)
    return final_hints[:count]


def get_hint_char_to_show(hint: str, typed_prefix: str) -> str:
    """Get the next character to show for a hint given what's been typed.

    This enables progressive disclosure - only show one char at a time
    to avoid overlaps with adjacent targets.
    """
    prefix_len = len(typed_prefix)
    if prefix_len < len(hint):
        return hint[prefix_len].upper()
    return ""


def find_char_positions(lines: List[iterm2.screen.LineContents],
                        target_char: str) -> List[Tuple[int, int]]:
    """Find all positions of target character in screen content."""
    matches: List[Tuple[int, int]] = []
    needle = target_char.lower()

    for row, line in enumerate(lines):
        col = 0
        while True:
            try:
                cell_text = line.string_at(col)
            except IndexError:
                break

            if not cell_text:
                col += 1
                continue

            first_char = cell_text[0]
            if first_char.strip() and first_char.lower() == needle:
                matches.append((row, col))
            col += 1

    return matches


def _color_to_sgr(color, prefix: str) -> Optional[str]:
    if color is None:
        return None

    try:
        if color.is_standard:
            return f"{prefix};5;{color.standard}"
        if color.is_rgb:
            rgb = color.rgb
            return f"{prefix};2;{rgb.red};{rgb.green};{rgb.blue}"
        if color.is_alternate:
            alt = color.alternate
            if alt == iterm2.screen.CellStyle.AlternateColor.DEFAULT:
                return '39' if prefix == '38' else '49'
    except Exception:
        pass
    return None


def _style_to_sgr(style, extra: Optional[List[str]] = None) -> str:
    if style is None:
        codes = ["0"]
        if extra:
            codes.extend(extra)
        return f"\033[{';'.join(codes)}m"

    codes: List[str] = ["0"]

    if style.bold:
        codes.append("1")
    if style.faint:
        codes.append("2")
    if style.italic:
        codes.append("3")
    if style.underline:
        codes.append("4")
    if style.blink:
        codes.append("5")
    if style.inverse:
        codes.append("7")
    if style.invisible:
        codes.append("8")
    if style.strikethrough:
        codes.append("9")

    try:
        fg = style.fg_color
        fg_code = _color_to_sgr(fg, '38')
        if fg_code:
            codes.append(fg_code)
    except Exception:
        pass

    try:
        bg = style.bg_color
        bg_code = _color_to_sgr(bg, '48')
        if bg_code:
            codes.append(bg_code)
    except Exception:
        pass

    if extra:
        codes.extend(extra)
    return f"\033[{';'.join(codes)}m"


async def get_screen_content(session) -> List[iterm2.screen.LineContents]:
    """Get the visible screen content from the session."""
    contents = await session.async_get_screen_contents()
    lines: List[iterm2.screen.LineContents] = []
    for i in range(contents.number_of_lines):
        lines.append(contents.line(i))
    return lines


def _line_cell_count(line: iterm2.screen.LineContents) -> int:
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


def get_matching_positions(
        positions: List[Tuple[int, int]],
        hints: List[str],
        typed_prefix: str
) -> Tuple[List[Tuple[Tuple[int, int], str]], Optional[Tuple[int, int]]]:
    """Get positions matching typed prefix and check for exact match.

    Returns:
        - List of (position, next_char_to_show) for hints matching prefix
        - If exactly one hint matches completely, returns that position as second element
    """
    matches: List[Tuple[Tuple[int, int], str]] = []
    exact_match: Optional[Tuple[int, int]] = None
    lower_prefix = typed_prefix.lower()

    for (row, col), hint in zip(positions, hints):
        hint_lower = hint.lower()
        # Only include hints that match typed prefix
        if not hint_lower.startswith(lower_prefix):
            continue

        # Check for exact match
        if hint_lower == lower_prefix:
            exact_match = (row, col)

        # Get the next character to show
        char_to_show = get_hint_char_to_show(hint, typed_prefix)
        if char_to_show:
            matches.append(((row, col), char_to_show))

    return matches, exact_match


def build_single_char_hint_map(
        matches: List[Tuple[Tuple[int, int], str]]
) -> Dict[Tuple[int, int], str]:
    """Build hint map from matches list."""
    return {pos: char for pos, char in matches}


def _build_hint_only_sequence(lines: List[iterm2.screen.LineContents],
                              hint_map: Dict[Tuple[int, int], str]) -> str:
    """Build escape sequence that only draws hints at specific positions."""
    parts: List[str] = ["\0337"]  # Save cursor position

    for (row, col), hint_char in hint_map.items():
        # Move to position and draw hint
        parts.append(f"\033[{row + 1};{col + 1}H")
        parts.append("\033[1;97;100m")  # Bold bright white on dark grey
        parts.append(hint_char)
        parts.append("\033[0m")

    parts.append("\0338")  # Restore cursor position
    return ''.join(parts)


def _build_restore_sequence(lines: List[iterm2.screen.LineContents],
                            positions: set) -> str:
    """Build escape sequence to restore cells at given positions."""
    parts: List[str] = ["\0337"]  # Save cursor position

    # Sort positions for consistent order
    sorted_positions = sorted(positions, key=lambda p: (p[0], p[1]))

    for (row, col) in sorted_positions:
        # Move to position
        parts.append(f"\033[{row + 1};{col + 1}H")

        try:
            cell_text = lines[row].string_at(col)
            style = None
            try:
                style = lines[row].style_at(col)
            except Exception:
                pass

            # Restore with original style
            parts.append(_style_to_sgr(style))
            parts.append(cell_text if cell_text else ' ')
        except (IndexError, Exception):
            # Clear with reset if can't get original
            parts.append("\033[0m ")

        # Reset after each char
        parts.append("\033[0m")

    parts.append("\0338")  # Restore cursor position
    return ''.join(parts)


def _build_screen_sequence(lines: List[iterm2.screen.LineContents],
                           hint_map: Optional[Dict[Tuple[int, int], str]] = None,
                           dim_non_hints: bool = False) -> str:
    """Build escape sequence to render screen with optional hints overlay."""
    parts: List[str] = ["\0337"]  # Save cursor position
    for row, line in enumerate(lines):
        cell_count = _line_cell_count(line)
        if cell_count == 0:
            continue

        for col in range(cell_count):
            key = (row, col)
            try:
                cell_text = line.string_at(col)
            except IndexError:
                break

            # Move cursor to exact position for each character
            parts.append(f"\033[{row + 1};{col + 1}H")

            if cell_text == "":
                parts.append(" ")  # Draw space for empty cells
                continue

            style = None
            try:
                style = line.style_at(col)
            except Exception:
                pass

            if hint_map and key in hint_map:
                # Highlighted hint: bold bright white on dark grey
                parts.append("\033[1;97;100m")
                parts.append(hint_map[key])
                parts.append("\033[0m")
            else:
                extras = ['2'] if dim_non_hints else None
                parts.append(_style_to_sgr(style, extras))
                parts.append(cell_text or ' ')

    parts.append("\033[0m\0338")  # Reset and restore cursor
    return ''.join(parts)


async def jump_to_position(connection,
                           session,
                           row: int,
                           col: int):
    """Jump to the specified position and enter Copy Mode."""
    try:
        lineInfo = await session.async_get_line_info()
        overflow = lineInfo.overflow
        first = lineInfo.first_visible_line_number

        start = iterm2.Point(col, first + overflow + row)
        end = iterm2.Point(col, first + overflow + row)

        coordRange = iterm2.CoordRange(start, end)
        windowedCoordRange = iterm2.WindowedCoordRange(coordRange)
        sub = iterm2.SubSelection(windowedCoordRange, iterm2.SelectionMode.CHARACTER,
                                  False)
        selection = iterm2.Selection([sub])

        await session.async_set_selection(selection)
        await asyncio.sleep(0.05)

        # Enter Copy Mode
        await iterm2.MainMenu.async_select_menu_item(connection, "Copy Mode")

    except Exception as e:
        print(f"Error jumping to position: {e}")


def _create_all_keys_pattern() -> iterm2.KeystrokePattern:
    """Create a pattern that matches all printable keys and Escape."""
    pattern = iterm2.KeystrokePattern()
    # Match all printable keys using characters instead of keycodes for broader coverage
    # This catches any single printable character
    pattern.characters = list(
        'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789`~!@#$%^&*()-_=+[]{}\\|;:\'",.<>/?'
    )
    pattern.keycodes = [iterm2.Keycode.ESCAPE, iterm2.Keycode.SPACE]
    return pattern


async def _ace_jump_with_alt_screen(connection, session, mon, lines, positions, hints):
    """Ace jump using alternate screen buffer (for non-tmux)."""
    typed_prefix = ""
    jump_target: Optional[Tuple[int, int]] = None

    # Switch to alternate screen buffer - main screen is preserved
    await session.async_inject(b'\033[?1049h')

    try:
        # Draw the original screen content on alternate buffer
        base_screen = _build_screen_sequence(lines)
        await session.async_inject(base_screen.encode('utf-8'))

        while True:
            # Get matching positions and check for exact match
            matches, exact_match = get_matching_positions(positions, hints, typed_prefix)

            # If exact match found, jump there
            if exact_match:
                jump_target = exact_match
                break

            if not matches:
                break

            # Check if only one candidate left - jump directly
            if len(matches) == 1:
                jump_target = matches[0][0]
                break

            # Display hints overlay on alternate screen
            hint_map = build_single_char_hint_map(matches)
            hint_only_sequence = _build_hint_only_sequence(lines, hint_map)
            await session.async_inject(hint_only_sequence.encode('utf-8'))

            # Capture hint character
            keystroke = await mon.async_get()

            # Handle Escape to cancel
            if keystroke.keycode == iterm2.Keycode.ESCAPE:
                break

            key = keystroke.characters
            if not key:
                continue

            typed_prefix += key.lower()[:1]

            # Redraw base screen before showing updated hints
            await session.async_inject(base_screen.encode('utf-8'))

    finally:
        # Switch back to main screen buffer - perfectly restores original
        await session.async_inject(b'\033[?1049l')

    return jump_target


async def _ace_jump_direct_overlay(connection, session, mon, lines, positions, hints):
    """Ace jump using direct hint overlay with restore (for tmux)."""
    typed_prefix = ""
    jump_target: Optional[Tuple[int, int]] = None

    # All target positions need restoration (each hint only modifies its target pos)
    all_positions: set = set(positions)

    try:
        while True:
            # Get matching positions and check for exact match
            matches, exact_match = get_matching_positions(positions, hints, typed_prefix)

            # If exact match found, jump there
            if exact_match:
                jump_target = exact_match
                break

            if not matches:
                break

            # Check if only one candidate left - jump directly
            if len(matches) == 1:
                jump_target = matches[0][0]
                break

            # Display hints overlay
            hint_map = build_single_char_hint_map(matches)
            hint_only_sequence = _build_hint_only_sequence(lines, hint_map)
            await session.async_inject(hint_only_sequence.encode('utf-8'))

            # Capture hint character
            keystroke = await mon.async_get()

            # Handle Escape to cancel
            if keystroke.keycode == iterm2.Keycode.ESCAPE:
                break

            key = keystroke.characters
            if not key:
                continue

            typed_prefix += key.lower()[:1]

    finally:
        # Restore all target positions
        if all_positions:
            restore_seq = _build_restore_sequence(lines, all_positions)
            await session.async_inject(restore_seq.encode('utf-8'))
            await asyncio.sleep(0.05)

    return jump_target


async def ace_jump_interactive(connection, session):
    """Interactive ace jump using KeystrokeMonitor and KeystrokeFilter."""
    session_id = session.session_id
    use_alt_screen = await should_use_alt_screen(session)

    # Create filter pattern to prevent keystrokes from reaching terminal
    filter_pattern = _create_all_keys_pattern()

    # Use KeystrokeFilter to intercept keys, KeystrokeMonitor to read them
    async with iterm2.KeystrokeFilter(connection, [filter_pattern],
                                      session_id) as _filter:
        async with iterm2.KeystrokeMonitor(connection, session_id) as mon:
            # Step 1: Capture target character
            keystroke = await mon.async_get()

            # Handle Escape to cancel
            if keystroke.keycode == iterm2.Keycode.ESCAPE:
                return

            target_char = keystroke.characters
            if not target_char or len(target_char) != 1:
                return

            # Step 2: Get screen content and find positions
            lines = await get_screen_content(session)
            positions = find_char_positions(lines, target_char)

            if not positions:
                return

            # Step 3: Generate hints
            hints = generate_hints(len(positions))

            # If only one match, jump directly
            if len(positions) == 1:
                row, col = positions[0]
                await jump_to_position(connection, session, row, col)
                return

            # Use different approach based on environment
            if use_alt_screen:
                # Normal: use alternate screen buffer (clean restoration)
                jump_target = await _ace_jump_with_alt_screen(
                    connection, session, mon, lines, positions, hints
                )
            else:
                # tmux/multiplexer: use direct overlay (no alternate screen)
                jump_target = await _ace_jump_direct_overlay(
                    connection, session, mon, lines, positions, hints
                )

            # Jump after restoring screen
            if jump_target:
                row, col = jump_target
                await jump_to_position(connection, session, row, col)


async def main(connection):
    """Main entry point for the ace jump script."""
    app = await iterm2.async_get_app(connection)

    # Run ace jump on current session
    window = app.current_terminal_window
    if window:
        session = window.current_tab.current_session
        if session:
            await ace_jump_interactive(connection, session)


iterm2.run_until_complete(main)
