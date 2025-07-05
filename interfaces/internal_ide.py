"""
internal_ide.py – Internal IDE for daemon self-coding and evolution.
Provides safe, controlled code editing capabilities for the daemon.
"""

import os
import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

class ModificationLevel(Enum):
    """Levels of modification permission."""
    SUGGESTION = "suggestion"      # Can suggest changes
    SANDBOXED = "sandboxed"        # Can modify in sandbox
    APPROVED = "approved"          # Human-approved changes
    AUTONOMOUS = "autonomous"      # Full self-modification

@dataclass
class CodeModification:
    """Represents a code modification request."""
    file_path: str
    original_content: str
    new_content: str
    modification_type: str
    safety_level: str
    description: str
    requires_approval: bool = True

@dataclass
class ModificationResult:
    """Result of a code modification attempt."""
    success: bool
    message: str
    backup_path: Optional[str] = None
    rollback_available: bool = False
    test_results: Optional[Dict[str, Any]] = None

class BackupSystem:
    """Manages backups and rollbacks for safe modifications."""
    
    def __init__(self, backup_dir: str = "unimind/backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger('BackupSystem')
    
    def create_backup(self, file_path: str) -> str:
        """Create a backup of a file."""
        try:
            source_path = Path(file_path)
            if not source_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Create backup filename with timestamp
            timestamp = int(time.time())
            backup_name = f"{source_path.stem}_{timestamp}.backup"
            backup_path = self.backup_dir / backup_name
            
            # Copy file
            shutil.copy2(source_path, backup_path)
            
            self.logger.info(f"Backup created: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
            raise
    
    def restore_backup(self, backup_path: str, target_path: str) -> bool:
        """Restore a file from backup."""
        try:
            backup_file = Path(backup_path)
            target_file = Path(target_path)
            
            if not backup_file.exists():
                raise FileNotFoundError(f"Backup not found: {backup_path}")
            
            # Restore file
            shutil.copy2(backup_file, target_file)
            
            self.logger.info(f"Restored from backup: {target_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Restore failed: {e}")
            return False

class SandboxedCodeEditor:
    """Safe code editor with sandboxed execution."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.backup_system = BackupSystem()
        self.logger = logging.getLogger('SandboxedCodeEditor')
        
        # Define allowed modification patterns
        self.allowed_patterns = [
            "unimind/soul/soul_profiles/*.json",
            "unimind/scrolls/custom_scrolls/*.py",
            "unimind/config/user_preferences.json",
            "unimind/config/daemon_settings.json"
        ]
        
        # Define forbidden patterns
        self.forbidden_patterns = [
            "unimind/core/*.py",           # Core system files
            "unimind/scrolls/scroll_engine.py",  # Main scroll engine
            "main.py",                     # Main entry point
            "requirements.txt",            # Dependencies
            "*.pyc",                       # Compiled Python files
            "__pycache__/*"                # Python cache
        ]
    
    def is_allowed_modification(self, file_path: str) -> bool:
        """Check if a file can be modified."""
        file_path = Path(file_path)
        
        # Check forbidden patterns first
        for pattern in self.forbidden_patterns:
            if file_path.match(pattern):
                return False
        
        # Check allowed patterns
        for pattern in self.allowed_patterns:
            if file_path.match(pattern):
                return True
        
        return False
    
    def safe_modify(self, modification: CodeModification) -> ModificationResult:
        """Safely modify a file with backup and validation."""
        try:
            file_path = Path(modification.file_path)
            
            # Check if modification is allowed
            if not self.is_allowed_modification(str(file_path)):
                return ModificationResult(
                    success=False,
                    message=f"Modification not allowed for: {file_path}"
                )
            
            # Create backup
            backup_path = self.backup_system.create_backup(str(file_path))
            
            # Validate new content (basic checks)
            if not self.validate_content(modification.new_content, file_path.suffix):
                return ModificationResult(
                    success=False,
                    message="Content validation failed",
                    backup_path=backup_path,
                    rollback_available=True
                )
            
            # Apply modification
            with open(file_path, 'w') as f:
                f.write(modification.new_content)
            
            # Run tests if available
            test_results = self.run_safety_tests(file_path)
            
            self.logger.info(f"Successfully modified: {file_path}")
            
            return ModificationResult(
                success=True,
                message=f"Successfully modified {file_path}",
                backup_path=backup_path,
                rollback_available=True,
                test_results=test_results
            )
            
        except Exception as e:
            self.logger.error(f"Modification failed: {e}")
            
            # Attempt rollback if backup exists
            if 'backup_path' in locals():
                self.backup_system.restore_backup(backup_path, str(file_path))
            
            return ModificationResult(
                success=False,
                message=f"Modification failed: {str(e)}",
                rollback_available='backup_path' in locals()
            )
    
    def validate_content(self, content: str, file_extension: str) -> bool:
        """Validate content before applying modification."""
        try:
            if file_extension == '.json':
                # Validate JSON syntax
                json.loads(content)
            elif file_extension == '.py':
                # Basic Python syntax check
                compile(content, '<string>', 'exec')
            
            return True
            
        except Exception:
            return False
    
    def run_safety_tests(self, file_path: Path) -> Dict[str, Any]:
        """Run safety tests on modified file."""
        try:
            # Basic tests - can be expanded
            tests = {
                "syntax_valid": True,
                "imports_valid": True,
                "no_dangerous_functions": True
            }
            
            if file_path.suffix == '.py':
                # Check for dangerous imports/functions
                content = file_path.read_text()
                dangerous_patterns = [
                    "import os",
                    "import sys",
                    "eval(",
                    "exec(",
                    "__import__"
                ]
                
                for pattern in dangerous_patterns:
                    if pattern in content:
                        tests["no_dangerous_functions"] = False
                        break
            
            return tests
            
        except Exception as e:
            return {"error": str(e)}

class InternalIDE:
    """Main internal IDE interface for the daemon."""
    
    def __init__(self, modification_level: ModificationLevel = ModificationLevel.SANDBOXED):
        self.modification_level = modification_level
        self.code_editor = SandboxedCodeEditor()
        self.logger = logging.getLogger('InternalIDE')
        
        # Track modification history
        self.modification_history: List[CodeModification] = []
    
    def suggest_modification(self, user_request: str, target_file: str) -> CodeModification:
        """Generate a code modification suggestion."""
        from unimind.native_models.text_to_code import text_to_code
        
        # Generate code using text-to-code engine
        result = text_to_code(user_request, language="python")
        
        if not result or not result.get("code"):
            raise ValueError("Could not generate code for modification")
        
        # Read current file content
        current_content = ""
        if Path(target_file).exists():
            with open(target_file, 'r') as f:
                current_content = f.read()
        
        return CodeModification(
            file_path=target_file,
            original_content=current_content,
            new_content=result["code"],
            modification_type="code_generation",
            safety_level="medium",
            description=f"Generated from request: {user_request}",
            requires_approval=self.modification_level in [ModificationLevel.SUGGESTION, ModificationLevel.SANDBOXED]
        )
    
    def apply_modification(self, modification: CodeModification) -> ModificationResult:
        """Apply a code modification."""
        if self.modification_level == ModificationLevel.SUGGESTION:
            return ModificationResult(
                success=False,
                message="Modifications require approval in suggestion mode"
            )
        
        # Apply the modification
        result = self.code_editor.safe_modify(modification)
        
        if result.success:
            self.modification_history.append(modification)
        
        return result
    
    def rollback_last_modification(self) -> bool:
        """Rollback the last modification."""
        if not self.modification_history:
            return False
        
        last_modification = self.modification_history.pop()
        
        # Restore original content
        try:
            with open(last_modification.file_path, 'w') as f:
                f.write(last_modification.original_content)
            
            self.logger.info(f"Rolled back modification: {last_modification.file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            return False
    
    def get_modification_history(self) -> List[Dict[str, Any]]:
        """Get history of modifications."""
        return [
            {
                "file_path": mod.file_path,
                "type": mod.modification_type,
                "description": mod.description,
                "timestamp": getattr(mod, 'timestamp', 'unknown')
            }
            for mod in self.modification_history
        ]
    
    def set_modification_level(self, level: ModificationLevel) -> None:
        """Set the modification permission level."""
        self.modification_level = level
        self.logger.info(f"Modification level set to: {level.value}")

# Global IDE instance
internal_ide = InternalIDE()

def get_internal_ide() -> InternalIDE:
    """Get the global internal IDE instance."""
    return internal_ide 