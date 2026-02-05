#!/usr/bin/env python3
"""Line completion using screen content - entry point for line mode.

Assign this to a different hotkey for line completion.
"""

import iterm2
from screen_complete import CompletionMode, complete_interactive


async def main(connection):
    """Main entry point for line completion."""
    app = await iterm2.async_get_app(connection)

    window = app.current_terminal_window
    if not window:
        return

    session = window.current_tab.current_session
    if not session:
        return

    await complete_interactive(connection, session, CompletionMode.LINE)


iterm2.run_until_complete(main)
