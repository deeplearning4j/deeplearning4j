Due to the macros and sheer number of type combinations we often run in to undefined symbols when linking. 

We typically use this command on linux to find issues:
nm -uC libnd4j/blasbuild/cpu/blas/libnd4jcpu.so

or cuda:
nm -uC libnd4j/blasbuild/cuda/blas/libnd4jcuda.so


You can also do the same for *just* undefined:
nm -C --undefined-only libnd4jcpu.so
nm -C --undefined-only libnd4jcuda.so