package org.nd4j.linalg.benchmark.api;


import org.apache.commons.lang3.time.StopWatch;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.concurrent.TimeUnit;

/**
 * @author Adam Gibson
 */
public abstract class BaseBenchmarkPerformer implements BenchMarkPerformer {
    protected int nTimes;
    protected long averageTime;
    protected StopWatch stopWatch;
    protected OpRunner runner;


    public BaseBenchmarkPerformer(int nTimes) {
        this.nTimes = nTimes;
        stopWatch = new StopWatch();
    }

    public BaseBenchmarkPerformer(OpRunner opRunner,int nTimes) {
        stopWatch = new StopWatch();
        this.nTimes = nTimes;
        this.runner = opRunner;
    }


    @Override
    public int nTimes() {
        return nTimes;
    }

    @Override
    public long averageTime() {
        return averageTime;
    }


    @Override
    public long run(Nd4jBackend backend) {
        Nd4j nd4j = new Nd4j();
        nd4j.initWithBackend(backend);
        averageTime = 0;

        for(int i = 0; i < nTimes; i++) {
            stopWatch.start();
            runner.runOp();
            stopWatch.stop();
            averageTime += stopWatch.getNanoTime();
            System.out.println("Time for trial " + i + " took " + stopWatch.getNanoTime() + "(ns) and " + TimeUnit.NANOSECONDS.toMillis(stopWatch.getNanoTime()) + " (ms)");
            stopWatch.reset();
        }

        averageTime /= nTimes;
        return averageTime;
    }

}
