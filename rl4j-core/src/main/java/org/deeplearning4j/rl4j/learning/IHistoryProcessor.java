package org.deeplearning4j.rl4j.learning;

import lombok.AllArgsConstructor;
import lombok.Value;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/6/16.
 *
 * An IHistoryProcessor come directly from the atari DQN paper.
 * It applies pre-processing the pixels of one state (gray-scaling + resizing)
 * then stacks it in different channels to be fed to a conv net
 */
public interface IHistoryProcessor {

    Configuration getConf();

    INDArray[] getHistory();

    void record(INDArray image);

    void add(INDArray image);

    void startMonitor(String filename);

    void stopMonitor();

    boolean isMonitoring();


    @AllArgsConstructor
    @Value
    public static class Configuration {
        int historyLength;
        int rescaledWidth;
        int rescaledHeight;
        int croppingWidth;
        int croppingHeight;
        int offsetX;
        int offsetY;
        int skipFrame;

        public Configuration() {
            historyLength = 4;
            rescaledWidth = 84;
            rescaledHeight = 84;
            croppingWidth = 84;
            croppingHeight = 84;
            offsetX = 0;
            offsetY = 0;
            skipFrame = 4;
        }

        public int[] getShape() {
            return new int[] {getHistoryLength(), getCroppingWidth(), getCroppingHeight()};
        }
    }
}
