# ND4J

A Numpy and Matlab like environment for cross-platform scientific computing. 

* Supports GPUS via CUDA and Native via Jblas.
* All of this is wrapped in a unifying interface.
* The API is a mix of Numpy and Jblas.

An example creation:

          INDArray arr = Nd4j.create(new float[]{1,2,3,4},new int[]{2,2});

This will create a 2 x 2 NDarray.

The project works as follows:

Include the following in your pom.xml:

       <dependency>
        <artifactId>nd4j</artifactId>
        <groupId>org.nd4j</groupId>
        <artifactId>nd4j-api</artifactId>
        <version>0.0.1-SNAPSHOT</version>
       </dependency>

From there, you need to pick a suitable implementation. This can be either Jblas for native or Cuda for GPUs.

Jblas:

             <dependency>
                <artifactId>nd4j</artifactId>
                <groupId>org.nd4j</groupId>
                <artifactId>nd4j-jblas</artifactId>
                <version>0.0.1-SNAPSHOT</version>
               </dependency>

Jcuda:

                    <dependency>
                       <artifactId>nd4j</artifactId>
                       <groupId>org.nd4j</groupId>
                       <artifactId>nd4j-jcublas</artifactId>
                       <version>0.0.1-SNAPSHOT</version>
                      </dependency>

For Jcuda, we are still in the process of streamlining the release for this one. For now, please do the following:

                  git clone https://github.com/SkymindIO/mavenized-jcuda
                  cd mavenized-jcuda
                  mvn clean install

This will install the Jcuda jar files.

You need to specify a version of Jcuda to use as well. The version will depend on your GPU. Amazon supports 0.5.5.

We will be streamlining this process soon. 

Basics:

In place operations:

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
