package org.deeplearning4j.linalg.jblas.benchmark;

import org.apache.commons.lang3.time.StopWatch;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.benchmark.TimeOperations;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.linalg.jblas.NDArray;

/**
 * Created by agibsonccc on 8/20/14.
 */
public class JblasBenchMark {



    public static void main(String[] args) {
        INDArray n = NDArrays.linspace(1, 100000, 100000).reshape(50000, 2);
        TimeOperations ops = new TimeOperations(n,100);
        ops.run();
        long source = 0;

        StopWatch watch = new StopWatch();

        for(int i = 0; i < 100000; i++) {
            watch.start();
            NDArray n2 = new NDArray(new float[]{10000},new int[]{2,5000});
            watch.stop();
            source += watch.getTime();
            System.out.println("Source " + watch.getTime());
            watch.reset();
        }

        source /= 1000;

        System.out.println("Avg time creation " + source);

    }


}
