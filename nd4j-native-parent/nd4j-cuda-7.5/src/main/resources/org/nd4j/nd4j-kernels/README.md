# nd4j-kernels


Compiling the kernels:

     make


Project setup (Development of code):

Nested hierarchy. Each kind of kernel (reduce/etc) are in their own folder. These are meant to be compiled as self contained modules for each operation.
Part of why the project may look strange is due to the nature of [jcuda](http://jcuda.org) and the need to load kernels by name. Due to limited access to templating
and the like,there aren't a lot of ways to pass the neccessary information to the compiler. Would love feedback if there's a better way to do this. 




NSight:

To get this up and running in nsight:
You can create a custom Makefile and make sure it's in your project root. Then you need to disable the auto generated makefiles. You can do that by going here:

Project Properties (right click on project and select properties) -> C/C++ Build -> in that window uncheck "Generate Makefiles Automatically." 

To use your own targets you can open the View called "Make Target":

Window -> Show View -> Make Target

In that view you can create a new target that will use the corresponding target in your custom Makefile.

Credit: http://stackoverflow.com/questions/1867990/using-custom-makefile-with-eclipse-cdt


