from abc import ABC, abstractmethod

class SchedulerInterface(ABC):
    @abstractmethod
    def schedule_task(self, task, *args, **kwargs):
        pass

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass