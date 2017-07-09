# Jumpy
--------------------------

Jumpy is the python interface for (nd4j)[https://nd4j.org]

Jumpy allows for python to use nd4j via pointers
(no network communication required like a lot of python libraries)

Jumpy is a thin wrapper around numpy and [jnius](https://github.com/kivy/pyjnius).


To setup jumpy, you need to set a jumpy classpath via the enviroment variable:

JUMPY_CLASSPATH

The JUMPY_CLASSPATH can be set to a list of jar files that contain
the necessary jar files for running an [nd4j backend](http://nd4j.org/backend.html)




