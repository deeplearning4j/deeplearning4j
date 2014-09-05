package org.nd4j.linalg.jblas.benchmark;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.benchmark.TimeOperations;
import org.nd4j.linalg.factory.Nd4j;
import org.jblas.DoubleMatrix;

/**
 * Created by agibsonccc on 8/20/14.
 */
public class JblasBenchMark {



    public static void main(String[] args) {
        INDArray n = Nd4j.linspace(1, 100000, 100000).reshape(50000, 2);
        TimeOperations ops = new TimeOperations(n,1000);
        ops.run();

        DoubleMatrix linspace = DoubleMatrix.linspace(1, 100000, 100000).reshape(50000, 2);
        DoubleMatrix linspace2 = DoubleMatrix.linspace(1, 100000, 100000).reshape(50000, 2);

        long timeDiff = 0;

        for(int i = 0; i < 1000; i++) {
            long before = System.currentTimeMillis();
            linspace.mul(linspace2);
            long after = System.currentTimeMillis();
            timeDiff += Math.abs(after - before);
        }

        System.out.println("Took on avg " + (timeDiff / 1000) + " milliseconds");


    }


}
