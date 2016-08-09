package org.deeplearning4j.optimize.listeners;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.util.ArrayUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Simple IterationListener that tracks time spend on training per iteration.
 *
 * @author raver119@gmail.com
 */
public class PerformanceListener implements IterationListener {
    private final int frequency;
    private final boolean reportScore;
    private static final Logger logger = LoggerFactory.getLogger(PerformanceListener.class);
    private float samplesPerSec = 0.0f;
    private float batchesPerSec = 0.0f;
    private long lastTime;


    public PerformanceListener(int frequency) {
        this(frequency, false);
    }

    public PerformanceListener(int frequency, boolean reportScore) {
        this.frequency = frequency;
        this.reportScore = reportScore;
        this.lastTime = System.currentTimeMillis();
    }

    @Override
    public boolean invoked() {
        return false;
    }

    @Override
    public void invoke() {

    }

    @Override
    public void iterationDone(Model model, int iteration) {
        // we update statistics on every iteration
        // but print it out only with specific freq
        long currentTime = System.currentTimeMillis();
        long timeSpent = lastTime - currentTime;
        float timeSec = timeSpent / 1000;

        INDArray input = model.input();
        long tadLength = Shape.getTADLength(input.shape(), ArrayUtil.range(1,input.rank()));

        long numSamples = input.lengthLong() / tadLength;

        samplesPerSec = numSamples / timeSec;
        batchesPerSec = 1 / timeSec;


        if (iteration % frequency == 0) {
            StringBuilder builder = new StringBuilder();

            builder.append("Iteration ").append(iteration).append("; ")
                    .append("samples/sec: ").append(samplesPerSec).append("; ")
                    .append("batches/sec: ").append(batchesPerSec).append("; ");

            if (reportScore)
                builder.append("score: ").append(model.score()).append(";");


            logger.info(builder.toString());
        }

        lastTime = System.currentTimeMillis();
    }
}
