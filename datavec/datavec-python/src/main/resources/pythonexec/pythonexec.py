import sys
import traceback
import json
import inspect


try:

    pass
    sys.stdout.flush()
    sys.stderr.flush()
except Exception as ex:
    try:
        exc_info = sys.exc_info()
    finally:
        print(ex)
        traceback.print_exception(*exc_info)
        sys.stdout.flush()
        sys.stderr.flush()

