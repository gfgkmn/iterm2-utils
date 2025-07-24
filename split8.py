#!/usr/bin/env python

import iterm2
import asyncio

async def main(connection):
    app=await iterm2.async_get_app(connection)
    window=app.current_terminal_window
    if window is not None:
        # tab=await window.async_create_tab()
        tab = window.current_tab
        for _ in range(3):
            session = tab.current_session
            await session.async_split_pane(vertical=True)
        sessions = tab.sessions
        for session in sessions:
            for _ in range(1):
                await session.async_split_pane(vertical=False)
    else:
        print("No Current Window")

iterm2.run_until_complete(main)
