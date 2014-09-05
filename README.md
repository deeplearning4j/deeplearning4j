ND4j
=========================================


A Numpy and matlab like environment for cross platform scientific computing. 

Supports GPUS via CUDA and Native via jblas.

All of this is wrapped in a unifying interface.

The api is a mix of numpy and jblas.




An example creation:

          INDArray arr = Nd4j.create(new float[]{1,2,3,4},new int[]{2,2});


This will create a 2 x 2 ndarray.


The way the project works as follows:


Include the following in your pom.xml:


       <dependency>
        <artifactId>nd4j</artifactId>
        <groupId>org.nd4j</groupId>
        <artifactId>nd4j-api</artifactId>
        <version>0.0.1-SNAPSHOT</version>
       </dependency>



From here, you need to pick an implementation suitable for your needs. This can be either jblas for native or cuda for GPUs.


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

For jcuda, we are still in the process of streamlining the release for this one. For now, please do the following:


                  git clone https://github.com/SkymindIO/mavenized-jcuda
                  cd mavenized-jcuda
                  mvn clean install


This will install the jcuda jar files.

You need to specify a version of jcuda to use as well. The version will depend on your GPU. Amazon supports 0.5.5.


We will be streamllining this process soon as well. 




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
                 
        
         Dimension wise operations (column and row order depending on the implementation chosen)
         
         arr.sum(0);
         
       







