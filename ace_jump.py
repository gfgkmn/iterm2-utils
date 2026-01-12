#!/usr/bin/env python3

import asyncio
from typing import Dict, Iterable, List, Optional, Tuple

import iterm2
import iterm2.screen

# QWERTY keyboard order for hints (home row first)
QWERTY_CHARS = 'asdfghjklqwertyuiopzxcvbnm'


def generate_hints(count: int) -> List[str]:
    """Generate hint labels in QWERTY order with equal length."""
    if count == 0:
        return []

    # Calculate the minimum length needed
    base = len(QWERTY_CHARS)
    length = 1
    while base ** length < count:
        length += 1

    final_hints: List[str] = []

    def generate_fixed_length(prefix: str, remaining_length: int) -> bool:
        if remaining_length == 0:
            final_hints.append(prefix)
            return len(final_hints) < count

        for char in QWERTY_CHARS:
            if not generate_fixed_length(prefix + char, remaining_length - 1):
                return False
        return True

    generate_fixed_length('', length)
    return final_hints[:count]


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


def _build_hint_map(
        active: Iterable[Tuple[Tuple[int, int], str]]) -> Dict[Tuple[int, int], str]:
    hint_map: Dict[Tuple[int, int], str] = {}
    for (row, col), hint in active:
        if not hint:
            continue
        for index, char in enumerate(hint.upper()):
            key = (row, col + index)
            if key not in hint_map:
                hint_map[key] = char
    return hint_map


def _build_screen_sequence(lines: List[iterm2.screen.LineContents],
                           hint_map: Optional[Dict[Tuple[int, int], str]] = None,
                           dim_non_hints: bool = False) -> str:
    """Build escape sequence to render screen with optional hints overlay."""
    parts: List[str] = ["\0337"]  # Save cursor position
    for row, line in enumerate(lines):
        cell_count = _line_cell_count(line)
        if cell_count == 0:
            continue

        parts.append(f"\033[{row + 1};1H")  # Move to row start

        for col in range(cell_count):
            key = (row, col)
            try:
                cell_text = line.string_at(col)
            except IndexError:
                break

            if cell_text == "":
                continue

            style = None
            try:
                style = line.style_at(col)
            except Exception:
                pass

            if hint_map and key in hint_map:
                # Highlighted hint: bold white on magenta
                parts.append("\033[1;97;45m")
                parts.append(hint_map[key])
                parts.append("\033[0m")
            else:
                extras = ['2'] if dim_non_hints else None
                parts.append(_style_to_sgr(style, extras))
                parts.append(cell_text or ' ')

    parts.append("\033[0m\0338")  # Reset and restore cursor
    return ''.join(parts)


async def jump_to_position(connection, session, row: int, col: int):
    """Jump to the specified position and enter Copy Mode."""
    try:
        lineInfo = await session.async_get_line_info()
        overflow = lineInfo.overflow
        first = lineInfo.first_visible_line_number

        start = iterm2.Point(col, first + overflow + row)
        end = iterm2.Point(col, first + overflow + row)

        coordRange = iterm2.CoordRange(start, end)
        windowedCoordRange = iterm2.WindowedCoordRange(coordRange)
        sub = iterm2.SubSelection(windowedCoordRange, iterm2.SelectionMode.CHARACTER, False)
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
    # Match all letter keys (for hints and target char)
    pattern.keycodes = [
        iterm2.Keycode.ANSI_A, iterm2.Keycode.ANSI_B, iterm2.Keycode.ANSI_C,
        iterm2.Keycode.ANSI_D, iterm2.Keycode.ANSI_E, iterm2.Keycode.ANSI_F,
        iterm2.Keycode.ANSI_G, iterm2.Keycode.ANSI_H, iterm2.Keycode.ANSI_I,
        iterm2.Keycode.ANSI_J, iterm2.Keycode.ANSI_K, iterm2.Keycode.ANSI_L,
        iterm2.Keycode.ANSI_M, iterm2.Keycode.ANSI_N, iterm2.Keycode.ANSI_O,
        iterm2.Keycode.ANSI_P, iterm2.Keycode.ANSI_Q, iterm2.Keycode.ANSI_R,
        iterm2.Keycode.ANSI_S, iterm2.Keycode.ANSI_T, iterm2.Keycode.ANSI_U,
        iterm2.Keycode.ANSI_V, iterm2.Keycode.ANSI_W, iterm2.Keycode.ANSI_X,
        iterm2.Keycode.ANSI_Y, iterm2.Keycode.ANSI_Z,
        iterm2.Keycode.ESCAPE,
    ]
    return pattern


async def ace_jump_interactive(connection, session):
    """Interactive ace jump using KeystrokeMonitor and KeystrokeFilter."""
    session_id = session.session_id

    # Create filter pattern to prevent keystrokes from reaching terminal
    filter_pattern = _create_all_keys_pattern()

    # Use KeystrokeFilter to intercept keys, KeystrokeMonitor to read them
    async with iterm2.KeystrokeFilter(connection, [filter_pattern], session_id) as _filter:
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

            # Build hint entries and original screen for restoration
            hint_entries = [((row, col), hint) for (row, col), hint in zip(positions, hints)]
            original_sequence = _build_screen_sequence(lines)
            typed_prefix = ""
            restored = False

            try:
                while True:
                    # Filter candidates based on typed prefix
                    lower_prefix = typed_prefix.lower()
                    candidates = [
                        entry for entry in hint_entries
                        if entry[1].lower().startswith(lower_prefix)
                    ]

                    if not candidates:
                        break

                    # Check for exact match
                    if lower_prefix:
                        exact_matches = [
                            entry for entry in candidates if entry[1].lower() == lower_prefix
                        ]
                        if len(exact_matches) == 1:
                            (row, col), _ = exact_matches[0]
                            await session.async_inject(original_sequence.encode('utf-8'))
                            restored = True
                            await jump_to_position(connection, session, row, col)
                            return

                    # Step 4: Display hints overlay
                    hint_map = _build_hint_map(candidates)
                    highlight_sequence = _build_screen_sequence(lines, hint_map=hint_map, dim_non_hints=True)
                    await session.async_inject(highlight_sequence.encode('utf-8'))

                    # Step 5: Capture hint character
                    keystroke = await mon.async_get()

                    # Handle Escape to cancel
                    if keystroke.keycode == iterm2.Keycode.ESCAPE:
                        break

                    key = keystroke.characters
                    if not key:
                        continue

                    typed_prefix += key.lower()[:1]

            finally:
                if not restored:
                    await session.async_inject(original_sequence.encode('utf-8'))


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
