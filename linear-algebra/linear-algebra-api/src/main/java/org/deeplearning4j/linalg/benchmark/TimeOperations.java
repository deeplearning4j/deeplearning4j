package org.deeplearning4j.linalg.benchmark;

import org.apache.commons.lang3.time.StopWatch;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Times matrix ops
 *
 * @author Adam Gibson
 */
public class TimeOperations {

    private static Logger log = LoggerFactory.getLogger(TimeOperations.class);
    private INDArray testing;
    private int numTimesRun = 1;
    private StopWatch watch = new StopWatch();

    public TimeOperations(INDArray n) {
        this(n,1);
    }

    public TimeOperations(INDArray n,int numTimesRun) {
        this.testing = n;
        this.numTimesRun = numTimesRun;
    }


    public void benchmarkCreation() {
        System.out.println("Benchmarking creation...");
        System.out.println(runNTimes(new Runnable() {
            @Override
            public void run() {
                NDArrays.create(new double[]{10000},new int[]{2,5000});
            }
        }).getMean() + " milliseconds");
    }

    public void benchmarkRavel() {
        System.out.println("Benchmarking ravel...");
        System.out.println(runNTimes(new Runnable() {
            @Override
            public void run() {
                testing.ravel();
            }
        }).getMean() + " milliseconds");

    }

    public SummaryStatistics runNTimes(Runnable run) {
        SummaryStatistics sum = new SummaryStatistics();
        for(int i = 0; i < numTimesRun; i++) {
            sum.addValue(timeOp(run));
        }

        return sum;

    }



    public long timeOp(Runnable run) {
        watch.start();
        run.run();
        watch.stop();
        long time = watch.getTime();
        watch.reset();
        return time;
    }






    public void run() {
        benchmarkRavel();
        benchmarkCreation();

    }




}
