from ..schedulers.scheduler_interface import SchedulerInterface
from ..events.event_handler import handle_event

# class TaskManager:
#     def __init__(self, scheduler: SchedulerInterface):
#         self.scheduler = scheduler
#
#     def schedule_task(self, task, event=None):
#         if event:
#             handle_event(event)
#         self.scheduler.schedule_task(task, trigger='interval', seconds=10)
#
#     def start(self):
#         self.scheduler.start()
#
#     def stop(self):
#         self.scheduler.stop()