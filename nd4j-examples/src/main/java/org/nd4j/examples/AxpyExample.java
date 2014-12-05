package org.nd4j.examples;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by agibsonccc on 11/25/14.
 */
public class AxpyExample {


    public static void main(String[] args) {
        INDArray arr = Nd4j.create(300);
        double numTimes = 10000000;
        double total = 0;
        for(int i = 0; i < numTimes; i++) {
            long start = System.nanoTime();
            Nd4j.getBlasWrapper().axpy(1.0,arr,arr);
            long after = System.nanoTime();
            long add = Math.abs(after - start);
            System.out.println("Took " + add);
            total += Math.abs(after - start);


        }

        System.out.println("Avg time " + (total / numTimes));
    }


}
