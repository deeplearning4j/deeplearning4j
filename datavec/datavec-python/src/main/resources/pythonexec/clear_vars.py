#See: https://stackoverflow.com/questions/3543833/how-do-i-clear-all-variables-in-the-middle-of-a-python-script
import sys
this = sys.modules[__name__]
for n in dir():
      if n[0]!='_': delattr(this, n)