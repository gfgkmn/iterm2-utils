#!/usr/bin/env python3.7

import iterm2


# This script was created with the "basic" environment which does not support adding dependencies
# with pip.
async def main(connection):
    # Your code goes here. Here's a bit of example code that adds a tab to the current window:
    app = await iterm2.async_get_app(connection)
    window = app.current_terminal_window
    panes = window.current_tab.sessions

    # 循环迭代每个面板，并输入增量数字
    for i, pane in enumerate(panes):
        input_string = str(i)  # 将数字转换为字符串
        await pane.async_send_text(input_string)


iterm2.run_until_complete(main)
