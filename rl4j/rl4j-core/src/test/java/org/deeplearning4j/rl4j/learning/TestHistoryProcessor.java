package org.deeplearning4j.rl4j.learning;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class TestHistoryProcessor implements IHistoryProcessor {
    public int recordCallCount;
    public int addCallCount;
    public int getHistoryCallCount;
    public int getScaleCallCount;

    private final Configuration conf;

    public TestHistoryProcessor(int skipFrame) {
        conf = new Configuration.ConfigurationBuilder()
                .skipFrame(skipFrame)
                .historyLength(10)
                .build();
    }

    public TestHistoryProcessor() {
        this(3);
    }

    @Override
    public Configuration getConf() {
        return conf;
    }

    @Override
    public INDArray[] getHistory() {
        ++getHistoryCallCount;

        INDArray result = Nd4j.create(new int[] { 1, 1 });
        result.putScalar(new int[] { 0, 0 }, getHistoryCallCount);

        return new INDArray[] { result };
    }

    @Override
    public void record(INDArray image) {
        ++recordCallCount;
    }

    @Override
    public void add(INDArray image) {
        ++addCallCount;
    }

    @Override
    public void startMonitor(String filename, int[] shape) {

    }

    @Override
    public void stopMonitor() {

    }

    @Override
    public boolean isMonitoring() {
        return false;
    }

    @Override
    public double getScale() {
        ++getScaleCallCount;
        return 1.0;
    }
}
