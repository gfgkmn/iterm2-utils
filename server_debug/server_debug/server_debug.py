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
        'bash': [
            r'\$ $',
            r'# $',
            r'[^>]*@[^>]*:[^>]*\$ ',
            r'[^>]*:[^>]*\$ ',
            r'^yuhe\$',
            r'^gfgkmn\\>: ',
        ]
    }

    # Check last few lines
    recent_lines = [line.string.strip() for line in lines[-5:]]

    for session_type, patterns in session_indicators.items():
        for line in recent_lines:
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
    # Convert backslashes to forward slashes for consistency
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
    mappings = {
        "/root/.cache/huggingface/modules/transformers_modules":
            "/root/workspace/crhavk47v38s73fnfgbg/dcformer",
        "/home/yuhe/.cache/huggingface/modules/transformers_modules/unify_format":
            "/home/yuhe/dcformer"
    }
    for old_path, new_path in mappings.items():
        if old_path in path_str:
            path_str = path_str.replace(old_path, new_path)
    return path_str


async def get_all_panes_info(connection):
    """Get information from all panes in the current tab."""
    app = await iterm2.async_get_app(connection)
    window = app.current_terminal_window

    if not window:
        return []

    tab = window.current_tab
    if not tab:
        return []

    # Get all sessions (panes) in the current tab
    sessions = tab.sessions
    pane_count = len(sessions)

    results = []
    active_pane_result = None
    active_session = tab.current_session

    # For each pane, get its information
    for pane_ind, session in enumerate(sessions):
        # Get info for this specific session
        pane_info = await get_pane_info(connection, session)
        pane_info["pane_index"] = pane_ind

        # Check if this is the active pane
        if session == active_session:
            active_pane_result = pane_info
        else:
            results.append(pane_info)

    # Ensure active pane is first in the results
    if active_pane_result:
        results.insert(0, active_pane_result)

    return results


async def get_pane_info(connection, target_session):
    """Modified version of get_last_number that works with a specific session."""
    app = await iterm2.async_get_app(connection)
    window = app.current_terminal_window

    if window and target_session:
        line_info = await target_session.async_get_line_info()
        lines = await target_session.async_get_contents(
            first_line=line_info.first_visible_line_number,
            number_of_lines=line_info.mutable_area_height)

        session_type = detect_session_type(lines)

        line = -1
        current_file = ""
        arrow_line = -1
        for line_contents in reversed(lines):
            line_str = line_contents.string
            if determine_officel_package(line_str):
                continue
            file_match = re.search(r'(File|>)\s*"?([^"=(]+)(", line |\()(\d+)\)?',
                                   line_str)
            if file_match:
                current_file = file_match.group(2)
                line = int(file_match.group(4))
                break
            line_match = re.search(r'-> (\d*)', line_str)
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
        for line_contents in reversed(lines):
            line_str = line_contents.string
            folder_match = re.search(
                r"(?:^\([^\(]*\))?[^\s:]*:(\/.*?)\s*\d{4}-\d{2}-\d{2}\s*\d{2}:\d{2}:\d{2}",
                line_str)
            if folder_match:
                current_dir = folder_match.group(1)
                break
        if current_dir == "":
            # parse current_dir from current_file if current_dir is not found
            if current_file:
                current_dir = os.path.dirname(current_file)
            else:
                current_dir = await target_session.async_get_variable("path")

        if job_name == "ssh":
            configs = parse_ssh_config()
            if hostname in ["yuhes-mbp", "Yuhes-MacBook-Pro.local", "Yuhes-MBP.lan"]:
                real_host = job_args.split("@")[-1]
                hostname = real_host.split(":")[0]
            machine_str = job_args.split(' ')[-1]
            if configs.lookup(machine_str).get('hostname') != machine_str:
                hostname = machine_str

            if job_args.startswith("ssh -W"):
                # ssh -W %h:%p host
                # extract hostname, combine job_name and ssh config file content
                hostname = get_hostname_from_config_when_jump(job_args, configs)
        if (not username or username == "") and job_name == "ssh":
            try:
                username = job_args.split(" ")[1].split("@")[0]
            except (IndexError, AttributeError):
                username = ""  # or set a default username
        if (not hostname or hostname == "") and job_name == "ssh":
            try:
                hostname = job_args.split("@")[-1].split(":")[0]
            except (IndexError, AttributeError):
                hostname = ""

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


async def send_code_to_iterm(code, connection):
    app = await iterm2.async_get_app(connection)
    window = app.current_terminal_window
    if window:
        tab = window.current_tab
        if tab:
            session = tab.current_session
            if session:
                pyperclip.copy(code)
                await session.async_send_text(code)
                await session.async_send_text("\n")
                return True  # Return True when code is sent successfully
    return False  # Return False if any step fails


async def handle_runcode(request, connection):
    data = await request.json()
    code_snippet = data.get('code')
    if code_snippet:
        success = await send_code_to_iterm(code_snippet, connection)  # Check return value
        if success:
            return aiohttp.web.Response(text='Code sent successfully')
        else:
            return aiohttp.web.Response(text='Failed to send code', status=500)
    return aiohttp.web.Response(text='No code provided', status=400)


async def handle_breakpoint(request, connection):
    break_info = await get_all_panes_info(connection)
    return aiohttp.web.json_response(break_info)


async def main(connection):
    app = aiohttp.web.Application()
    breakpoint_handler = partial(handle_breakpoint, connection=connection)
    runcode_handler = partial(handle_runcode, connection=connection)
    app.router.add_get('/breakpoint', breakpoint_handler)
    app.router.add_post('/send_code', runcode_handler)
    runner = aiohttp.web.AppRunner(app)
    await runner.setup()
    site = aiohttp.web.TCPSite(runner, 'localhost', 17647)
    await site.start()
    print("Server running on http://localhost:17647")

    # Add an infinite loop to keep the server running:
    while True:
        await asyncio.sleep(3600)  # Sleep for 1 hour


# Run the server
iterm2.run_until_complete(main)
