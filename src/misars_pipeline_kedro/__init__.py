"""model-deployment-test
"""
from .pipelines.sit_manager import create_sit_manager_pipeline
def register_pipelines():
    return {
        "sit_manager": create_sit_manager_pipeline(),
    }
__version__ = "0.1"
