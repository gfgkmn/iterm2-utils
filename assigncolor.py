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

import os
import signal
import subprocess
import time

import iterm2


def _terminate_stale_instances(marker, timeout=3.0):
    """Kill older copies of this script before this one starts.

    iTerm2 does not stop autolaunch scripts when it quits: they are
    reparented to launchd and keep running, so every launch leaves
    another copy behind.  Several copies then race to
    colour the same new session.  Newest instance wins; our own process
    and our it2_api_wrapper parent are never touched.
    """
    me, parent = os.getpid(), os.getppid()
    try:
        out = subprocess.run(['/bin/ps', '-ax', '-o', 'pid=,command='],
                             capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return []
    victims = []
    for line in out.splitlines():
        pid_str, _, cmd = line.strip().partition(' ')
        tokens = cmd.split()
        # Match only a real invocation: an interpreter (or iTerm's wrapper)
        # whose ARGUMENT is this script.  Matching the bare string would also
        # hit any shell, editor or grep that merely mentions the filename.
        if not tokens or os.path.basename(tokens[0]) not in (
                'python', 'python3', 'bash', 'sh'):
            continue
        if not any(tok.endswith(marker) for tok in tokens[1:]):
            continue
        try:
            pid = int(pid_str)
        except ValueError:
            continue
        if pid not in (me, parent):
            victims.append(pid)
    for pid in victims:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    deadline = time.time() + timeout
    for pid in victims:
        while time.time() < deadline:          # wait for it to actually exit
            try:
                os.kill(pid, 0)
            except OSError:
                break
            time.sleep(0.1)
        else:                                   # still alive at the deadline
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    if victims:
        print("Terminated stale instance(s): %s" % victims)
    return victims


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

color_counter = 0

async def color_all_tabs(app):
    """Assign colors to all existing tabs in all windows."""
    global color_counter
    for window in app.windows:
        for tab in window.tabs:
            session = tab.current_session
            color = tab_colors[color_counter % len(tab_colors)]
            await set_tab_color(session, color)
            color_counter += 1

async def main(connection):
    global color_counter
    _terminate_stale_instances('assigncolor.py')
    app = await iterm2.async_get_app(connection)

    # Color all existing tabs on startup
    await color_all_tabs(app)

    # Monitor for new sessions and assign next color in sequence
    async with iterm2.NewSessionMonitor(connection) as mon:
        while True:
            session_id = await mon.async_get()
            app = await iterm2.async_get_app(connection)
            session = app.get_session_by_id(session_id)
            if session is None:
                continue
            color = tab_colors[color_counter % len(tab_colors)]
            await set_tab_color(session, color)
            color_counter += 1

iterm2.run_forever(main)
