#!/usr/bin/env python3

import asyncio
import os
import re
from functools import partial

import aiohttp.web
import iterm2
import paramiko
import pyperclip


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
            r'\[claude\]:',          # Claude Code status line prompt
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

    # Remove escaped brackets and split the command
    job_args = job_args.replace('\\[', '').replace('\\]', '')
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

        if session == active_session:
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

        username = await target_session.async_get_variable("username")
        hostname = await target_session.async_get_variable("hostname")
        job_name = await target_session.async_get_variable("jobName")
        job_args = await target_session.async_get_variable("commandLine")

        current_dir = ""
        if session_type == 'claude':
            for logical_line in reversed(logical_lines):
                claude_match = re.search(r'\[claude\]:\s*\{[^}]*\}(.*?)\s{2,}', logical_line)
                if claude_match:
                    current_dir = claude_match.group(1).strip()
                    break

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
        }

    return {}


async def send_code_to_iterm(code,
                             connection,
                             target_pane=None,
                             broadcast=False,
                             is_ascii=False,
                             multiline=False):
    """Enhanced function to send code to iTerm2 with broadcast and targeting support"""
    app = await iterm2.async_get_app(connection)
    window = app.current_terminal_window

    if not window:
        return False

    tab = window.current_tab
    if not tab:
        return False

    async def send_to_session(session, code, is_ascii, multiline):
        """Helper function to send code to a single session"""
        if is_ascii:
            # Send ASCII character
            ascii_code = ord(code) if len(code) == 1 else int(code)
            await session.async_send_text(chr(ascii_code))
        else:
            # Get session info to determine type
            session_info = await get_pane_info(connection, session)
            session_type = session_info.get('session_type', 'unknown')

            if session_type == 'claude':
                if multiline:
                    # Split into lines, send with Escape+Enter for newlines
                    lines = code.split('\n')
                    for i, line in enumerate(lines):
                        await session.async_send_text(line)
                        if i < len(lines) - 1:
                            # Escape then Enter = newline within Claude Code input
                            await session.async_send_text('\x1b')
                            await asyncio.sleep(0.05)
                            await session.async_send_text('\n')
                            await asyncio.sleep(0.05)
                    # Final Enter to submit
                    await session.async_send_text('\n')
                else:
                    # Single line — send directly + Enter to submit
                    await session.async_send_text(code)
                    await session.async_send_text('\n')
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

    try:
        if broadcast:
            # Broadcast to all sessions in current tab
            sessions = tab.sessions
            for session in sessions:
                await send_to_session(session, code, is_ascii, multiline)
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

            await send_to_session(session, code, is_ascii, multiline)

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
    """Create a new pane in the current tab"""
    app = await iterm2.async_get_app(connection)
    window = app.current_terminal_window

    if not window:
        return False

    tab = window.current_tab
    if not tab:
        return False

    try:
        # Split current session horizontally to create new pane
        current_session = tab.current_session
        if current_session:
            await current_session.async_split_pane(vertical=True)
            return True
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
        broadcast = data.get('broadcast', False)
        is_ascii = data.get('is_ascii', False)
        multiline = data.get('multiline', False)

        if not code:
            return aiohttp.web.Response(text='No code provided', status=400)

        success = await send_code_to_iterm(code, connection, target_pane, broadcast,
                                           is_ascii, multiline)

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
            success = await create_new_pane(connection)
            if success:
                return aiohttp.web.Response(text='New pane created successfully')
            else:
                return aiohttp.web.Response(text='Failed to create new pane', status=500)

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
    print("  POST /control    - Control operations (select_pane, new_pane)")
    print("  GET  /screen_content - Get current pane screen content")

    # Keep the server running
    while True:
        await asyncio.sleep(3600)


# Run the enhanced server
if __name__ == "__main__":
    iterm2.run_until_complete(main)
