# ND4J

A Numpy and Matlab like environment for cross-platform scientific computing. 

* Supports GPUs via CUDA and Native via Jblas.
* All of this is wrapped in a unifying interface.
* The API is a mix of Numpy and Jblas.

An example creation:

           INDArray arr = Nd4j.create(new float[]{1,2,3,4},new int[]{2,2});

This will create a 2 x 2 NDarray.
###Setup

Please see:
http://nd4j.org/getstarted.html



### Basics

In-place operations:

             INDArray arr = Nd4j.create(new float[]{1,2,3,4},new int[]{2,2});
             //scalar operation
             arr.addi(1);

             //element wise operations
             INDArray arr2 = ND4j.create(new float[]{5,6,7,8},new int[]{2,2});
             arr.addi(arr2);
         
Duplication operations:
                
             //clone then add
             arr.add(1);
             //clone then add
             arr.add(arr2);
                 
Dimensionwise operations (column and row order depending on the implementation chosen)
         
             arr.sum(0);
