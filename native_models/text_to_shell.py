"""
text_to_shell.py – Natural language to shell command conversion for Unimind native models.
Provides conversion from natural language to shell commands with safety validation.
"""

import re
import subprocess
import shlex
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

class ShellOperation(Enum):
    """Enumeration of shell operations."""
    LIST = "list"
    SHOW = "show"
    DISPLAY = "display"
    CREATE = "create"
    DELETE = "delete"
    MOVE = "move"
    COPY = "copy"
    SEARCH = "search"
    EXECUTE = "execute"
    STATUS = "status"
    HELP = "help"

@dataclass
class ShellResult:
    """Result of shell command conversion."""
    command: str
    operation: ShellOperation
    confidence: float
    arguments: List[str]
    flags: List[str]
    explanation: str
    safe_to_execute: bool

class TextToShell:
    """
    Converts natural language commands to shell commands.
    Provides safety validation and command execution capabilities.
    """
    
    def __init__(self):
        """Initialize the TextToShell converter."""
        self.command_patterns = {
            ShellOperation.LIST: [
                r"list|show|display|ls|dir|what files|what's in|contents of",
                r"list all|show all|display all|what's here"
            ],
            ShellOperation.SHOW: [
                r"show|display|cat|view|read|open|print",
                r"show me|display the|let me see"
            ],
            ShellOperation.CREATE: [
                r"create|make|new|touch|mkdir|add",
                r"create new|make new|add new"
            ],
            ShellOperation.DELETE: [
                r"delete|remove|rm|del|erase|clear",
                r"delete the|remove the|erase the"
            ],
            ShellOperation.MOVE: [
                r"move|mv|relocate|shift",
                r"move to|relocate to|shift to"
            ],
            ShellOperation.COPY: [
                r"copy|cp|duplicate|clone",
                r"copy to|duplicate to|clone to"
            ],
            ShellOperation.SEARCH: [
                r"search|find|grep|locate|where is",
                r"search for|find the|locate the"
            ],
            ShellOperation.STATUS: [
                r"status|ps|top|system|process",
                r"system status|process status|what's running"
            ],
            ShellOperation.HELP: [
                r"help|man|info|what can|how to",
                r"help me|show help|get help"
            ]
        }
        
        self.dangerous_commands = {
            "rm -rf", "rm -rf /", "rm -rf /*", "dd", "format", "mkfs",
            "fdisk", "parted", "chmod 777", "chmod -R 777", "sudo rm",
            "sudo dd", "sudo format", "sudo mkfs"
        }
        
        self.safe_commands = {
            "ls", "ls -la", "ls -l", "pwd", "whoami", "date", "uptime",
            "ps", "ps aux", "top", "df", "du", "cat", "head", "tail",
            "grep", "find", "locate", "which", "whereis", "file",
            "stat", "wc", "sort", "uniq", "echo", "touch", "mkdir"
        }
        
    def convert_to_shell(self, natural_command: str) -> ShellResult:
        """
        Convert natural language command to shell command.
        
        Args:
            natural_command: Natural language command string
            
        Returns:
            ShellResult containing the converted command and metadata
        """
        command_lower = natural_command.lower().strip()
        
        # Determine operation
        operation = self._detect_operation(command_lower)
        
        # Extract arguments and flags
        arguments = self._extract_arguments(command_lower)
        flags = self._extract_flags(command_lower)
        
        # Generate shell command
        command = self._generate_command(operation, arguments, flags)
        
        # Calculate confidence
        confidence = self._calculate_confidence(command_lower, operation, arguments)
        
        # Check safety
        safe_to_execute = self._check_safety(command)
        
        # Generate explanation
        explanation = self._generate_explanation(operation, arguments, flags)
        
        return ShellResult(
            command=command,
            operation=operation,
            confidence=confidence,
            arguments=arguments,
            flags=flags,
            explanation=explanation,
            safe_to_execute=safe_to_execute
        )
    
    def _detect_operation(self, command: str) -> ShellOperation:
        """Detect the shell operation from the command."""
        for operation, patterns in self.command_patterns.items():
            for pattern in patterns:
                if re.search(pattern, command):
                    return operation
        return ShellOperation.LIST  # Default to list
    
    def _extract_arguments(self, command: str) -> List[str]:
        """Extract command arguments from the natural language."""
        arguments = []
        
        # Look for file/directory names
        file_patterns = [
            r"file\s+(\w+)",
            r"directory\s+(\w+)",
            r"folder\s+(\w+)",
            r"(\w+\.\w+)",  # Files with extensions
            r"(\w+/\w+)",   # Paths
        ]
        
        for pattern in file_patterns:
            matches = re.findall(pattern, command)
            arguments.extend(matches)
        
        # Look for specific words that might be arguments
        words = command.split()
        for word in words:
            if word not in ["list", "show", "display", "create", "delete", "move", "copy", "search", "find", "the", "a", "an", "and", "or", "in", "to", "from", "with"]:
                if len(word) > 2:  # Avoid very short words
                    arguments.append(word)
        
        return list(set(arguments))  # Remove duplicates
    
    def _extract_flags(self, command: str) -> List[str]:
        """Extract command flags from the natural language."""
        flags = []
        
        flag_mappings = {
            "all": ["-a", "-la"],
            "long": ["-l"],
            "recursive": ["-r", "-R"],
            "force": ["-f"],
            "verbose": ["-v"],
            "quiet": ["-q"],
            "human": ["-h"],
            "details": ["-la", "-l"],
            "hidden": ["-a"],
            "subdirectories": ["-r", "-R"]
        }
        
        for flag_word, flag_options in flag_mappings.items():
            if flag_word in command:
                flags.extend(flag_options)
        
        return list(set(flags))  # Remove duplicates
    
    def _generate_command(self, operation: ShellOperation, arguments: List[str], flags: List[str]) -> str:
        """Generate shell command string."""
        flag_str = " " + " ".join(flags) if flags else ""
        arg_str = " " + " ".join(arguments) if arguments else ""
        
        if operation == ShellOperation.LIST:
            return f"ls{flag_str}{arg_str}"
        elif operation == ShellOperation.SHOW:
            return f"cat{flag_str}{arg_str}"
        elif operation == ShellOperation.CREATE:
            if any("dir" in arg.lower() or "folder" in arg.lower() for arg in arguments):
                return f"mkdir{flag_str}{arg_str}"
            else:
                return f"touch{flag_str}{arg_str}"
        elif operation == ShellOperation.DELETE:
            return f"rm{flag_str}{arg_str}"
        elif operation == ShellOperation.MOVE:
            return f"mv{flag_str}{arg_str}"
        elif operation == ShellOperation.COPY:
            return f"cp{flag_str}{arg_str}"
        elif operation == ShellOperation.SEARCH:
            return f"grep{flag_str}{arg_str}"
        elif operation == ShellOperation.STATUS:
            return f"ps{flag_str}{arg_str}"
        elif operation == ShellOperation.HELP:
            return f"man{flag_str}{arg_str}"
        else:
            return f"ls{flag_str}{arg_str}"
    
    def _calculate_confidence(self, command: str, operation: ShellOperation, arguments: List[str]) -> float:
        """Calculate confidence score for the conversion."""
        confidence = 0.5  # Base confidence
        
        # Boost confidence for clear operation detection
        if operation != ShellOperation.LIST:
            confidence += 0.2
        
        # Boost confidence for argument detection
        if arguments:
            confidence += 0.2
        
        # Boost confidence for flag detection
        if any(flag_word in command for flag_word in ["all", "long", "recursive", "force", "verbose"]):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _check_safety(self, command: str) -> bool:
        """Check if the command is safe to execute."""
        command_lower = command.lower()
        
        # Check for dangerous commands
        for dangerous in self.dangerous_commands:
            if dangerous in command_lower:
                return False
        
        # Check for sudo with dangerous operations
        if "sudo" in command_lower and any(op in command_lower for op in ["rm", "dd", "format", "mkfs"]):
            return False
        
        # Check for recursive operations on root
        if "-r" in command_lower and any(path in command_lower for path in ["/", "/root", "/home"]):
            return False
        
        return True
    
    def _generate_explanation(self, operation: ShellOperation, arguments: List[str], flags: List[str]) -> str:
        """Generate human-readable explanation of the command conversion."""
        operation_name = operation.value
        arg_list = ", ".join(arguments) if arguments else "current directory"
        flag_list = ", ".join(flags) if flags else "default options"
        
        explanation = f"Converted to {operation_name} operation"
        explanation += f" on: {arg_list}"
        explanation += f" with flags: {flag_list}"
        
        return explanation
    
    def execute_command(self, shell_result: ShellResult) -> Dict[str, Any]:
        """
        Execute the shell command safely.
        
        Args:
            shell_result: ShellResult containing the command to execute
            
        Returns:
            Dictionary containing execution results
        """
        if not shell_result.safe_to_execute:
            return {
                "success": False,
                "error": "Command marked as unsafe to execute",
                "command": shell_result.command,
                "output": None
            }
        
        try:
            # Parse command safely
            args = shlex.split(shell_result.command)
            
            # Execute command
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )
            
            return {
                "success": result.returncode == 0,
                "command": shell_result.command,
                "output": result.stdout,
                "error": result.stderr,
                "return_code": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Command execution timed out",
                "command": shell_result.command,
                "output": None
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Command execution failed: {str(e)}",
                "command": shell_result.command,
                "output": None
            }
    
    def suggest_alternatives(self, natural_command: str) -> List[str]:
        """
        Suggest alternative commands for the natural language input.
        
        Args:
            natural_command: Natural language command
            
        Returns:
            List of alternative shell commands
        """
        alternatives = []
        command_lower = natural_command.lower()
        
        if "list" in command_lower or "show" in command_lower:
            alternatives.extend(["ls", "ls -la", "ls -l", "ls -a"])
        
        if "find" in command_lower or "search" in command_lower:
            alternatives.extend(["find . -name", "grep -r", "locate"])
        
        if "status" in command_lower or "process" in command_lower:
            alternatives.extend(["ps aux", "top", "htop", "systemctl status"])
        
        return alternatives

# Module-level instance
text_to_shell = TextToShell()

def convert_to_shell(natural_command: str) -> ShellResult:
    """Convert natural language to shell command using the module-level instance."""
    return text_to_shell.convert_to_shell(natural_command)

def execute_command(shell_result: ShellResult) -> Dict[str, Any]:
    """Execute shell command using the module-level instance."""
    return text_to_shell.execute_command(shell_result)
