package org.deeplearning4j.rl4j.support;

import lombok.Getter;
import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class TestHistoryProcessor implements IHistoryProcessor {

    @Getter
    private int recordCount = 0;

    @Getter
    private int addCount = 0;

    private int historyLength = 4;
    private int skipFrame = 3;

    private final Configuration config;

    public TestHistoryProcessor(int historyLength, int skipFrame)
    {
        config = new Configuration(historyLength, 0, 0, 0, 0, 0, 0, skipFrame);
    }

    @Override
    public Configuration getConf() {
        return config;
    }

    @Override
    public INDArray[] getHistory() {
        return new INDArray[] {
            Nd4j.zeros(new int[] { 2, 2 }),
            Nd4j.zeros(new int[] { 2, 2 }),
        };
    }

    @Override
    public void record(INDArray image) {
        ++recordCount;
    }

    @Override
    public void add(INDArray image) {
        ++addCount;
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
