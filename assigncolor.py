# #!/usr/bin/env python3

# import iterm2

# # Define rainbow colors
# rainbow_colors = [
#     (255, 0, 0),   # Red
#     (255, 127, 0), # Orange
#     (255, 255, 0), # Yellow
#     (0, 255, 0),   # Green
#     (0, 0, 255),   # Blue
#     (75, 0, 130),  # Indigo
#     (143, 0, 255), # Violet
# ]

# # Define grey color
# grey_color = (128, 128, 128)

# async def set_tab_color(session, color):
#     change = iterm2.LocalWriteOnlyProfile()
#     tab_color = iterm2.Color(*color)
#     change.set_tab_color(tab_color)
#     change.set_use_tab_color(True)
#     await session.async_set_profile_properties(change)

# async def main(connection):
#     app = await iterm2.async_get_app(connection)
#     window = app.current_window
#     if window is not None:
#         # Get the current active tab
#         current_tab = window.current_tab

#         for i, tab in enumerate(window.tabs):
#             session = tab.current_session
#             # Check if the tab is the active tab
#             if tab == current_tab:
#                 # Set rainbow color for the active tab
#                 color = tab_colors[i % len(tab_colors)]
#             else:
#                 # Set grey color for inactive tabs
#                 color = grey_color
#             await set_tab_color(session, color)
#     else:
#         print("No current window")

# iterm2.run_until_complete(main)



#!/usr/bin/env python3

import iterm2

# Define tab colors — dark/saturated for visible iTerm2 tab tints
tab_colors = [
    (220, 40, 40),    # Red
    (0, 150, 136),    # Teal
    (30, 80, 220),    # Royal Blue
    (180, 0, 140),    # Magenta
    (30, 140, 30),    # Forest Green
    (200, 100, 0),    # Dark Orange
    (100, 40, 180),   # Purple
    (70, 130, 180),   # Steel Blue
    (200, 50, 80),    # Crimson Pink
    (120, 130, 0),    # Olive
]

async def set_tab_color(session, color):
    change = iterm2.LocalWriteOnlyProfile()
    tab_color = iterm2.Color(*color)
    change.set_tab_color(tab_color)
    change.set_use_tab_color(True)
    await session.async_set_profile_properties(change)

async def main(connection):
    app = await iterm2.async_get_app(connection)
    window = app.current_window
    if window is not None:
        tabs = window.tabs
        for i, tab in enumerate(tabs):
            # Cycle through the rainbow_colors for each tab
            color = tab_colors[i % len(tab_colors)]
            # Get the current session of the tab
            session = tab.current_session
            # Set the tab color
            await set_tab_color(session, color)
    else:
        print("No current window")

iterm2.run_until_complete(main)
