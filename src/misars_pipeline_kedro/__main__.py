"""model-deployment-test file for ensuring the package is executable
as `model-deployment-test` and `python -m misars_pipeline_kedro`
"""
import sys
from pathlib import Path
from typing import Any

from kedro.framework.cli.utils import find_run_command
from kedro.framework.project import configure_project
from .schedulers.apscheduler_adapter import APSchedulerAdapter
from .managers.task_manager import TaskManager
# from .tasks.your_task_function import your_task_function


def main(*args, **kwargs) -> Any:
    scheduler = APSchedulerAdapter()
    task_manager = TaskManager(scheduler)
    package_name = Path(__file__).parent.name
    configure_project(package_name)

    interactive = hasattr(sys, 'ps1')
    kwargs["standalone_mode"] = not interactive

    run = find_run_command(package_name)
    return run(*args, **kwargs)


if __name__ == "__main__":
    main()
