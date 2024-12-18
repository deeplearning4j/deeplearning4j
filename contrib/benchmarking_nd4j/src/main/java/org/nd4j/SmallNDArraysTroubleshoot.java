package org.nd4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class SmallNDArraysTroubleshoot {
   // /home/agibsonccc/Documents/GitHub/deeplearning4j/contrib/benchmarking_nd4j
   // -cp target/benchmarks.jar  org.nd4j.SmallNDArraysTroubleshoot
    public static void main(String...args) {
        INDArray arr = Nd4j.ones(200);
        INDArray arr2 = Nd4j.ones(200);
        for(int i = 0; i < 1000000; i++) {
            System.out.println("Iteration " + i);
            arr.addi(arr2);
            System.out.println(arr);
        }
    }

}
