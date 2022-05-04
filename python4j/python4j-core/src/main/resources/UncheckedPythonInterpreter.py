
import sys

# Based on work from: https://github.com/invesdwin/invesdwin-context-python/blob/master/invesdwin-context-python-parent/invesdwin-context-python-runtime-python4j/src/main/java/de/invesdwin/context/python/runtime/python4j/internal/UncheckedPythonEngineWrapper.py

# http://stackoverflow.com/questions/3543833/how-do-i-clear-all-variables-in-the-middle-of-a-python-script
# Permission for use granted here: https://github.com/eclipse/deeplearning4j/issues/9595
__saved_context__ = {}

def saveContext():
    __saved_context__.update(sys.modules[__name__].__dict__)

def restoreContext():
    names = list(sys.modules[__name__].__dict__.keys())
    for n in names:
        if n not in __saved_context__:
            del sys.modules[__name__].__dict__[n]

saveContext()