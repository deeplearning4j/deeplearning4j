from .java_classes import Nd4j

memory_manager = Nd4j.getMemoryManager()


def disable_gc():
    memory_manager.togglePeriodicGc(False)


def enable_gc():
    memory_manager.togglePeriodicGc(True)


def set_gc_interval(interval=5000):
    memory_manager.setAutoGcWindow(interval)
