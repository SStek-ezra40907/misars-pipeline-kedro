from apscheduler.schedulers.background import BackgroundScheduler
from .scheduler_interface import SchedulerInterface

class APSchedulerAdapter(SchedulerInterface):
    def __init__(self):
        self.scheduler = BackgroundScheduler()

    def schedule_task(self, task, *args, **kwargs):
        self.scheduler.add_job(task, *args, **kwargs)

    def start(self):
        self.scheduler.start()

    def stop(self):
        self.scheduler.shutdown()