"""
Optimized workflow manager for the Story Writer Agent.
Provides better stage management, error recovery, and progress tracking.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

from data_models import StoryProject
from config_manager import AppConfig

logger = logging.getLogger(__name__)


class StageStatus(Enum):
    """Status of workflow stages"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class WorkflowStatus(Enum):
    """Overall workflow status"""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class StageResult:
    """Result of executing a workflow stage"""
    stage_name: str
    status: StageStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    error: Optional[Exception] = None
    output: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[float]:
        """Get stage duration in seconds"""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    @property
    def success(self) -> bool:
        """Check if stage completed successfully"""
        return self.status == StageStatus.COMPLETED


@dataclass
class WorkflowState:
    """Current state of the workflow"""
    status: WorkflowStatus = WorkflowStatus.NOT_STARTED
    current_stage: Optional[str] = None
    completed_stages: List[str] = field(default_factory=list)
    failed_stages: List[str] = field(default_factory=list)
    stage_results: Dict[str, StageResult] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_stages: int = 0
    project: Optional[StoryProject] = None

    @property
    def progress_percentage(self) -> float:
        """Get completion percentage"""
        if self.total_stages == 0:
            return 0.0
        return (len(self.completed_stages) / self.total_stages) * 100

    @property
    def duration(self) -> Optional[float]:
        """Get total workflow duration in seconds"""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


class WorkflowError(Exception):
    """Base exception for workflow errors"""
    pass


class StageError(WorkflowError):
    """Exception raised when a stage fails"""
    def __init__(self, stage_name: str, message: str, original_error: Exception = None):
        super().__init__(f"Stage '{stage_name}' failed: {message}")
        self.stage_name = stage_name
        self.original_error = original_error


class WorkflowManager:
    """Optimized workflow manager with stage orchestration and error recovery"""

    def __init__(self, config: AppConfig, story_agent: Any = None):
        """Initialize the workflow manager"""
        self.config = config
        self.story_agent = story_agent
        self.state = WorkflowState()

        # Stage handlers mapping
        self._stage_handlers: Dict[str, Callable] = {}
        self._stage_validators: Dict[str, Callable] = {}
        self._stage_recovery_handlers: Dict[str, Callable] = {}

        # Event callbacks
        self._stage_start_callbacks: List[Callable] = []
        self._stage_complete_callbacks: List[Callable] = []
        self._workflow_complete_callbacks: List[Callable] = []
        self._error_callbacks: List[Callable] = []

        # Recovery options
        self._auto_recovery_enabled = config.workflow.allow_stage_skipping
        self._max_retries_per_stage = 3
        self._recovery_strategies: Dict[str, str] = {}

        # Progress tracking
        self._progress_callback: Optional[Callable] = None
        self._checkpoint_callback: Optional[Callable] = None

        self._initialize_default_stages()

    def _initialize_default_stages(self):
        """Initialize default stage configuration"""
        default_stages = [
            "concept_generation",
            "structure_planning",
            "character_development",
            "setting_creation",
            "draft_generation",
            "revision_enhancement",
            "final_polish"
        ]

        self.state.total_stages = len(default_stages)

        # Set up default recovery strategies
        self._recovery_strategies = {
            "concept_generation": "retry",
            "structure_planning": "skip_with_defaults",
            "character_development": "retry",
            "setting_creation": "skip_with_defaults",
            "draft_generation": "retry_required",
            "revision_enhancement": "skip_optional",
            "final_polish": "skip_optional"
        }

    def register_stage_handler(self, stage_name: str, handler: Callable):
        """Register a handler for a specific stage"""
        self._stage_handlers[stage_name] = handler
        logger.debug(f"Registered handler for stage: {stage_name}")

    def register_stage_validator(self, stage_name: str, validator: Callable):
        """Register a validator for a specific stage"""
        self._stage_validators[stage_name] = validator
        logger.debug(f"Registered validator for stage: {stage_name}")

    def register_recovery_handler(self, stage_name: str, handler: Callable):
        """Register a recovery handler for a specific stage"""
        self._stage_recovery_handlers[stage_name] = handler
        logger.debug(f"Registered recovery handler for stage: {stage_name}")

    def set_progress_callback(self, callback: Callable[[WorkflowState], None]):
        """Set callback for progress updates"""
        self._progress_callback = callback

    def set_checkpoint_callback(self, callback: Callable[[WorkflowState], None]):
        """Set callback for checkpoint saves"""
        self._checkpoint_callback = callback

    def add_stage_start_callback(self, callback: Callable[[str], None]):
        """Add callback for stage start events"""
        self._stage_start_callbacks.append(callback)

    def add_stage_complete_callback(self, callback: Callable[[StageResult], None]):
        """Add callback for stage completion events"""
        self._stage_complete_callbacks.append(callback)

    def add_workflow_complete_callback(self, callback: Callable[[WorkflowState], None]):
        """Add callback for workflow completion"""
        self._workflow_complete_callbacks.append(callback)

    def add_error_callback(self, callback: Callable[[Exception, str], None]):
        """Add callback for error handling"""
        self._error_callbacks.append(callback)

    async def run_workflow(self, project: Optional[StoryProject] = None) -> WorkflowState:
        """Run the complete workflow with error handling and recovery"""
        try:
            # Initialize workflow state
            self.state.project = project or StoryProject()
            self.state.status = WorkflowStatus.RUNNING
            self.state.start_time = datetime.now()

            logger.info("Starting story writing workflow")
            await self._notify_progress()

            # Execute stages in sequence
            for stage_name in self.config.workflow.stages:
                if self.state.status == WorkflowStatus.CANCELLED:
                    break

                try:
                    await self._execute_stage(stage_name)
                except StageError as e:
                    logger.error(f"Stage {stage_name} failed: {e}")

                    # Attempt recovery if enabled
                    if not await self._handle_stage_failure(stage_name, e):
                        # Critical failure - stop workflow
                        self.state.status = WorkflowStatus.FAILED
                        break

                # Save checkpoint after each stage
                await self._save_checkpoint()

            # Finalize workflow
            if self.state.status == WorkflowStatus.RUNNING:
                self.state.status = WorkflowStatus.COMPLETED
                logger.info("Workflow completed successfully")

            self.state.end_time = datetime.now()

            # Notify completion
            for callback in self._workflow_complete_callbacks:
                try:
                    await self._call_async_or_sync(callback, self.state)
                except Exception as e:
                    logger.warning(f"Workflow completion callback failed: {e}")

            return self.state

        except Exception as e:
            logger.error(f"Workflow failed with unexpected error: {e}")
            self.state.status = WorkflowStatus.FAILED
            self.state.end_time = datetime.now()

            # Notify error callbacks
            for callback in self._error_callbacks:
                try:
                    await self._call_async_or_sync(callback, e, "workflow")
                except Exception as callback_error:
                    logger.warning(f"Error callback failed: {callback_error}")

            raise WorkflowError(f"Workflow execution failed: {e}") from e

    async def _execute_stage(self, stage_name: str):
        """Execute a single workflow stage with retries"""
        logger.info(f"Starting stage: {stage_name}")

        # Create stage result
        result = StageResult(
            stage_name=stage_name,
            status=StageStatus.RUNNING,
            start_time=datetime.now()
        )

        self.state.current_stage = stage_name
        self.state.stage_results[stage_name] = result

        # Notify stage start
        for callback in self._stage_start_callbacks:
            try:
                await self._call_async_or_sync(callback, stage_name)
            except Exception as e:
                logger.warning(f"Stage start callback failed: {e}")

        # Attempt execution with retries
        last_error = None
        for attempt in range(self._max_retries_per_stage):
            try:
                # Validate prerequisites
                if stage_name in self._stage_validators:
                    validator = self._stage_validators[stage_name]
                    if not await self._call_async_or_sync(validator, self.state.project):
                        raise StageError(stage_name, "Stage validation failed")

                # Execute stage handler
                if stage_name in self._stage_handlers:
                    handler = self._stage_handlers[stage_name]
                    result.output = await self._call_async_or_sync(handler)
                elif self.story_agent and hasattr(self.story_agent, f"stage_{stage_name}"):
                    # Fall back to story agent methods
                    handler = getattr(self.story_agent, f"stage_{stage_name}")
                    result.output = await self._call_async_or_sync(handler)
                else:
                    raise StageError(stage_name, f"No handler found for stage: {stage_name}")

                # Stage completed successfully
                result.status = StageStatus.COMPLETED
                result.end_time = datetime.now()

                self.state.completed_stages.append(stage_name)
                logger.info(f"Stage {stage_name} completed in {result.duration:.2f}s")

                # Notify stage completion
                for callback in self._stage_complete_callbacks:
                    try:
                        await self._call_async_or_sync(callback, result)
                    except Exception as e:
                        logger.warning(f"Stage completion callback failed: {e}")

                await self._notify_progress()
                return

            except Exception as e:
                last_error = e
                if attempt < self._max_retries_per_stage - 1:
                    logger.warning(f"Stage {stage_name} attempt {attempt + 1} failed: {e}, retrying...")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Stage {stage_name} failed after {self._max_retries_per_stage} attempts")

        # All attempts failed
        result.status = StageStatus.FAILED
        result.end_time = datetime.now()
        result.error = last_error

        self.state.failed_stages.append(stage_name)
        raise StageError(stage_name, f"Stage failed after {self._max_retries_per_stage} attempts", last_error)

    async def _handle_stage_failure(self, stage_name: str, error: StageError) -> bool:
        """Handle stage failure with recovery strategies"""
        recovery_strategy = self._recovery_strategies.get(stage_name, "stop")

        logger.info(f"Handling failure for stage {stage_name} with strategy: {recovery_strategy}")

        if recovery_strategy == "retry":
            # Already handled by _execute_stage retries
            return False

        elif recovery_strategy == "skip_optional":
            # Skip this stage and continue
            logger.info(f"Skipping optional stage: {stage_name}")
            result = self.state.stage_results[stage_name]
            result.status = StageStatus.SKIPPED
            result.end_time = datetime.now()
            return True

        elif recovery_strategy == "skip_with_defaults":
            # Try to provide default values and skip
            if stage_name in self._stage_recovery_handlers:
                try:
                    handler = self._stage_recovery_handlers[stage_name]
                    await self._call_async_or_sync(handler, self.state.project)

                    result = self.state.stage_results[stage_name]
                    result.status = StageStatus.SKIPPED
                    result.end_time = datetime.now()
                    result.metadata["recovery"] = "defaults_applied"

                    logger.info(f"Applied defaults for stage: {stage_name}")
                    return True
                except Exception as recovery_error:
                    logger.error(f"Recovery handler failed for {stage_name}: {recovery_error}")

            # Fall back to simple skip
            result = self.state.stage_results[stage_name]
            result.status = StageStatus.SKIPPED
            result.end_time = datetime.now()
            return True

        elif recovery_strategy == "retry_required":
            # This is a critical stage - stop workflow
            logger.error(f"Critical stage {stage_name} failed - stopping workflow")
            return False

        else:  # "stop" or unknown strategy
            return False

    async def _call_async_or_sync(self, func: Callable, *args, **kwargs):
        """Call function whether it's async or sync"""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    async def _notify_progress(self):
        """Notify progress callback of current state"""
        if self._progress_callback:
            try:
                await self._call_async_or_sync(self._progress_callback, self.state)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

    async def _save_checkpoint(self):
        """Save workflow checkpoint"""
        if self._checkpoint_callback:
            try:
                await self._call_async_or_sync(self._checkpoint_callback, self.state)
            except Exception as e:
                logger.warning(f"Checkpoint save failed: {e}")

    def pause_workflow(self):
        """Pause the workflow execution"""
        if self.state.status == WorkflowStatus.RUNNING:
            self.state.status = WorkflowStatus.PAUSED
            logger.info("Workflow paused")

    def resume_workflow(self):
        """Resume paused workflow execution"""
        if self.state.status == WorkflowStatus.PAUSED:
            self.state.status = WorkflowStatus.RUNNING
            logger.info("Workflow resumed")

    def cancel_workflow(self):
        """Cancel workflow execution"""
        self.state.status = WorkflowStatus.CANCELLED
        self.state.end_time = datetime.now()
        logger.info("Workflow cancelled")

    def get_stage_result(self, stage_name: str) -> Optional[StageResult]:
        """Get result for a specific stage"""
        return self.state.stage_results.get(stage_name)

    def is_stage_completed(self, stage_name: str) -> bool:
        """Check if a stage is completed"""
        result = self.get_stage_result(stage_name)
        return result is not None and result.success

    def get_workflow_summary(self) -> Dict[str, Any]:
        """Get a summary of the workflow execution"""
        completed_count = len(self.state.completed_stages)
        failed_count = len(self.state.failed_stages)
        skipped_count = len([r for r in self.state.stage_results.values()
                           if r.status == StageStatus.SKIPPED])

        total_duration = 0
        stage_durations = {}
        for stage_name, result in self.state.stage_results.items():
            if result.duration:
                stage_durations[stage_name] = result.duration
                total_duration += result.duration

        return {
            "status": self.state.status.value,
            "progress_percentage": self.state.progress_percentage,
            "total_duration": self.state.duration or total_duration,
            "stages": {
                "total": self.state.total_stages,
                "completed": completed_count,
                "failed": failed_count,
                "skipped": skipped_count,
                "pending": self.state.total_stages - completed_count - failed_count - skipped_count
            },
            "stage_durations": stage_durations,
            "completed_stages": self.state.completed_stages,
            "failed_stages": self.state.failed_stages,
            "current_stage": self.state.current_stage,
            "start_time": self.state.start_time.isoformat() if self.state.start_time else None,
            "end_time": self.state.end_time.isoformat() if self.state.end_time else None
        }

    def export_state(self) -> Dict[str, Any]:
        """Export workflow state for persistence"""
        return {
            "status": self.state.status.value,
            "current_stage": self.state.current_stage,
            "completed_stages": self.state.completed_stages,
            "failed_stages": self.state.failed_stages,
            "total_stages": self.state.total_stages,
            "start_time": self.state.start_time.isoformat() if self.state.start_time else None,
            "end_time": self.state.end_time.isoformat() if self.state.end_time else None,
            "stage_results": {
                name: {
                    "status": result.status.value,
                    "start_time": result.start_time.isoformat(),
                    "end_time": result.end_time.isoformat() if result.end_time else None,
                    "duration": result.duration,
                    "error": str(result.error) if result.error else None,
                    "metadata": result.metadata
                }
                for name, result in self.state.stage_results.items()
            }
        }

    def import_state(self, state_data: Dict[str, Any]):
        """Import workflow state from persistence"""
        try:
            self.state.status = WorkflowStatus(state_data["status"])
            self.state.current_stage = state_data.get("current_stage")
            self.state.completed_stages = state_data.get("completed_stages", [])
            self.state.failed_stages = state_data.get("failed_stages", [])
            self.state.total_stages = state_data.get("total_stages", 0)

            if state_data.get("start_time"):
                self.state.start_time = datetime.fromisoformat(state_data["start_time"])
            if state_data.get("end_time"):
                self.state.end_time = datetime.fromisoformat(state_data["end_time"])

            # Import stage results
            for stage_name, result_data in state_data.get("stage_results", {}).items():
                result = StageResult(
                    stage_name=stage_name,
                    status=StageStatus(result_data["status"]),
                    start_time=datetime.fromisoformat(result_data["start_time"]),
                    metadata=result_data.get("metadata", {})
                )

                if result_data.get("end_time"):
                    result.end_time = datetime.fromisoformat(result_data["end_time"])
                if result_data.get("error"):
                    result.error = Exception(result_data["error"])

                self.state.stage_results[stage_name] = result

            logger.info("Workflow state imported successfully")

        except Exception as e:
            logger.error(f"Failed to import workflow state: {e}")
            raise WorkflowError(f"State import failed: {e}") from e


class DefaultStageHandlers:
    """Default stage handlers for common workflow operations"""

    @staticmethod
    def create_concept_recovery_handler(default_concepts: List[Dict[str, str]]):
        """Create a recovery handler for concept generation"""
        async def recovery_handler(project: StoryProject):
            if not project.concept and default_concepts:
                # Use first default concept
                concept_data = default_concepts[0]
                from data_models import StoryConcept
                project.concept = StoryConcept(**concept_data)
                logger.info("Applied default concept for recovery")
        return recovery_handler

    @staticmethod
    def create_structure_recovery_handler():
        """Create a recovery handler for structure planning"""
        async def recovery_handler(project: StoryProject):
            if not project.structure:
                from data_models import StoryStructure
                project.structure = StoryStructure(
                    act1_setup={"description": "Basic setup structure"},
                    act2_confrontation={"description": "Basic confrontation structure"},
                    act3_resolution={"description": "Basic resolution structure"},
                    character_arc={"description": "Basic character development"}
                )
                logger.info("Applied default structure for recovery")
        return recovery_handler
