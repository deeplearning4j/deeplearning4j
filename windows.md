#Building on Windows

For compiling on windows, we use [Msys2](https://msys2.github.io/). 

We leverage the msys2 port of pacman for the package manager to get everything up and running.

To compile libnd4j for WITHIN the msys2 port shell, [setup msys2 first](https://msys2.github.io/)

Start the msys2 shell and setup the dev environment with:
     
      pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-extra-cmake-modules make pkg-config grep sed gzip tar mingw64/mingw-w64-x86_64-openblas

This will install the needed dependencies for use in the msys2 shell.


For cpu, we recommend openblas. We will be adding instructions for mkl and other cpu implementations later.

Send us a pull request or [file an issue](https://github.com/deeplearning4j/libnd4j/issues) if you have

something in particular you are looking for.



Finally, in that directory to build the dll:

./buildnativeoperations.sh blas cpu

For cuda:
./buildnativeoperations.sh blas cuda