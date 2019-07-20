package org.deeplearning4j.rl4j.support;

import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.nd4j.linalg.api.ndarray.INDArray;

public class MockHistoryProcessor implements IHistoryProcessor {

    private final Configuration config;

    public MockHistoryProcessor(Configuration config) {

        this.config = config;
    }

    @Override
    public Configuration getConf() {
        return config;
    }

    @Override
    public INDArray[] getHistory() {
        return new INDArray[0];
    }

    @Override
    public void record(INDArray image) {

    }

    @Override
    public void add(INDArray image) {

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
        return 0;
    }
}
