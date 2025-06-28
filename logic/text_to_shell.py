# text_to_shell.py
import subprocess

def convert_text_to_shell(command_text):
    """
    Convert natural language text to a shell command using simple rule-based logic.
    """
    command_text = command_text.lower().strip()

    if "list files" in command_text or "show files" in command_text:
        return "ls -la"
    elif "current directory" in command_text or "where am i" in command_text:
        return "pwd"
    elif "disk usage" in command_text:
        return "df -h"
    elif "memory usage" in command_text:
        return "free -h"
    elif "check internet" in command_text:
        return "ping -c 4 google.com"
    elif "show processes" in command_text:
        return "ps aux"
    else:
        return None

def execute_shell_command(shell_command):
    try:
        result = subprocess.run(shell_command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout.strip() or result.stderr.strip()
    except Exception as e:
        return str(e)

def process_text_command(input_text):
    command = convert_text_to_shell(input_text)
    if command:
        return execute_shell_command(command)
    else:
        return "Unrecognized command. Please try again with a simpler phrasing."