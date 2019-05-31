package org.deeplearning4j.rl4j.learning;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class TestHistoryProcessor implements IHistoryProcessor {
    public int recordCallCount;
    public int addCallCount;
    public int getHistoryCallCount;
    public int getScaleCallCount;

    private Configuration conf = new Configuration.ConfigurationBuilder()
            .skipFrame(3)
            .historyLength(10)
            .build();


    @Override
    public Configuration getConf() {
        return conf;
    }

    @Override
    public INDArray[] getHistory() {
        ++getHistoryCallCount;
        return new INDArray[] { Nd4j.create(new int[] { 1, 1 })};
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
        return 0;
    }
}
