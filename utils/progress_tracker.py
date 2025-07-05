#!/usr/bin/env python3
"""
Progress tracking utility for Unimind learning and execution activities.
Provides real-time progress bars, time estimation, and execution monitoring.
"""

import time
import threading
import sys
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass

@dataclass
class ProgressStep:
    """Represents a single step in a progress sequence."""
    name: str
    description: str
    status: str = "pending"  # pending, running, completed, failed
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Optional[str] = None
    error: Optional[str] = None

class ProgressTracker:
    """Tracks progress of multi-step operations with real-time updates."""
    
    def __init__(self, title: str, total_steps: int, show_progress_bar: bool = True):
        self.title = title
        self.total_steps = total_steps
        self.current_step = 0
        self.steps: List[ProgressStep] = []
        self.start_time = time.time()
        self.show_progress_bar = show_progress_bar
        self.is_running = False
        
    def add_step(self, name: str, description: str) -> int:
        """Add a step to the progress tracker."""
        step = ProgressStep(name=name, description=description)
        self.steps.append(step)
        return len(self.steps) - 1
    
    def start_step(self, step_index: int) -> None:
        """Start executing a step."""
        if 0 <= step_index < len(self.steps):
            self.steps[step_index].status = "running"
            self.steps[step_index].start_time = time.time()
            self.current_step = step_index + 1
            
            if self.show_progress_bar:
                self._display_progress()
    
    def complete_step(self, step_index: int, result: str = None) -> None:
        """Mark a step as completed."""
        if 0 <= step_index < len(self.steps):
            self.steps[step_index].status = "completed"
            self.steps[step_index].end_time = time.time()
            self.steps[step_index].result = result
            
            if self.show_progress_bar:
                self._display_progress()
    
    def fail_step(self, step_index: int, error: str) -> None:
        """Mark a step as failed."""
        if 0 <= step_index < len(self.steps):
            self.steps[step_index].status = "failed"
            self.steps[step_index].end_time = time.time()
            self.steps[step_index].error = error
            
            if self.show_progress_bar:
                self._display_progress()
    
    def _display_progress(self) -> None:
        """Display the current progress."""
        progress = (self.current_step / self.total_steps) * 100
        elapsed_time = time.time() - self.start_time
        
        # Clear line and display progress
        sys.stdout.write('\r')
        sys.stdout.write(f"🔄 {self.title}: {progress:.1f}% | ⏱️ {elapsed_time:.1f}s | Step {self.current_step}/{self.total_steps}")
        sys.stdout.flush()
        
        # If complete, add newline
        if self.current_step >= self.total_steps:
            sys.stdout.write('\n')
            sys.stdout.flush()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the progress execution."""
        total_time = time.time() - self.start_time
        completed_steps = sum(1 for step in self.steps if step.status == "completed")
        failed_steps = sum(1 for step in self.steps if step.status == "failed")
        
        return {
            "title": self.title,
            "total_steps": self.total_steps,
            "completed_steps": completed_steps,
            "failed_steps": failed_steps,
            "total_time": total_time,
            "steps": [
                {
                    "name": step.name,
                    "description": step.description,
                    "status": step.status,
                    "execution_time": step.end_time - step.start_time if step.start_time and step.end_time else 0,
                    "result": step.result,
                    "error": step.error
                }
                for step in self.steps
            ]
        }

class LearningProgressTracker(ProgressTracker):
    """Specialized progress tracker for learning activities."""
    
    def __init__(self, topic: str):
        super().__init__(f"Learning: {topic}", 5, show_progress_bar=True)
        self.topic = topic
        
        # Add standard learning steps
        self.add_step("research", f"Research current best practices in {topic}")
        self.add_step("analyze", "Identify specific skills and knowledge gaps")
        self.add_step("plan", "Create a structured learning approach")
        self.add_step("practice", "Practice and apply new knowledge")
        self.add_step("evaluate", "Evaluate progress and adjust learning strategy")
    
    def execute_with_progress(self, step_executor: Callable[[str, int], str]) -> Dict[str, Any]:
        """Execute the learning plan with real-time progress tracking."""
        
        results = []
        
        for i in range(self.total_steps):
            # Start step
            self.start_step(i)
            
            # Execute step
            try:
                step_name = self.steps[i].name
                result = step_executor(step_name, i + 1)
                self.complete_step(i, result)
                results.append({
                    "step_number": i + 1,
                    "step_description": self.steps[i].description,
                    "status": "completed",
                    "result": result,
                    "execution_time": self.steps[i].end_time - self.steps[i].start_time if self.steps[i].start_time and self.steps[i].end_time else 0,
                    "progress_percentage": ((i + 1) / self.total_steps) * 100
                })
            except Exception as e:
                self.fail_step(i, str(e))
                results.append({
                    "step_number": i + 1,
                    "step_description": self.steps[i].description,
                    "status": "failed",
                    "result": None,
                    "error": str(e),
                    "execution_time": self.steps[i].end_time - self.steps[i].start_time if self.steps[i].start_time and self.steps[i].end_time else 0,
                    "progress_percentage": ((i + 1) / self.total_steps) * 100
                })
            
            # Small delay to show progress
            time.sleep(0.5)
        
        # Get summary
        summary = self.get_summary()
        
        return {
            "status": "learning_executed",
            "topic": self.topic,
            "understanding": f"I understand you want me to learn about {self.topic}. This is an excellent area for self-improvement!",
            "learning_plan": [step.description for step in self.steps],
            "implementation": f"I can apply what I learn about {self.topic} to enhance my capabilities and provide better assistance.",
            "next_steps": [
                "I'll start researching this topic immediately",
                "I'll update my knowledge base with new information",
                "I'll practice applying this knowledge in our interactions",
                "I'll track my progress and share insights with you"
            ],
            "execution_results": results,
            "execution_time": summary["total_time"],
            "progress_complete": True,
            "timestamp": time.time()
        }

def create_progress_bar(progress: float, width: int = 50) -> str:
    """Create a visual progress bar."""
    filled = int(width * progress / 100)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {progress:.1f}%"

def display_execution_summary(summary: Dict[str, Any]) -> None:
    """Display a formatted execution summary."""
    print(f"\n📊 Execution Summary:")
    print(f"   Title: {summary['title']}")
    print(f"   Total Steps: {summary['total_steps']}")
    print(f"   Completed: {summary['completed_steps']}")
    print(f"   Failed: {summary['failed_steps']}")
    print(f"   Total Time: {summary['total_time']:.2f} seconds")
    print(f"   Average Time per Step: {summary['total_time'] / summary['total_steps']:.2f} seconds")
    
    print(f"\n📋 Step Details:")
    for step in summary['steps']:
        status_emoji = "✅" if step['status'] == 'completed' else "❌" if step['status'] == 'failed' else "⏳"
        print(f"   {status_emoji} {step['name']}: {step['description']}")
        print(f"      Time: {step['execution_time']:.2f}s | Status: {step['status']}")
        if step['result']:
            print(f"      Result: {step['result']}")
        if step['error']:
            print(f"      Error: {step['error']}")
        print() 