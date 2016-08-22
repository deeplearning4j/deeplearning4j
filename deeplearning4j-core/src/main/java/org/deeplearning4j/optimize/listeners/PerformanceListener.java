package org.deeplearning4j.optimize.listeners;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.util.ArrayUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.atomic.AtomicLong;

/**
 * Simple IterationListener that tracks time spend on training per iteration.
 *
 * @author raver119@gmail.com
 */
public class PerformanceListener implements IterationListener {
    private final int frequency;
    private static final Logger logger = LoggerFactory.getLogger(PerformanceListener.class);
    private double samplesPerSec = 0.0f;
    private double batchesPerSec = 0.0f;
    private long lastTime;
    private AtomicLong iterationCount = new AtomicLong(0);

    private boolean reportScore;
    private boolean reportSample = true;
    private boolean reportBatch = true;
    private boolean reportIteration = true;
    private boolean reportTime = true;



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
        // we update lastTime on every iteration
        // just to simplify things


        if (iterationCount.getAndIncrement() % frequency == 0) {
            long currentTime = System.currentTimeMillis();

            long timeSpent = currentTime - lastTime ;
            float timeSec = timeSpent / 1000f;

            INDArray input = model.input();
            long tadLength = Shape.getTADLength(input.shape(), ArrayUtil.range(1,input.rank()));

            long numSamples = input.lengthLong() / tadLength;

            samplesPerSec = numSamples / timeSec;
            batchesPerSec = 1 / timeSec;

            StringBuilder builder = new StringBuilder();

            if (reportIteration)
                builder.append("iteration ").append(iterationCount.get()).append("; ");

            if (reportTime)
                builder.append("iteration time: ").append(timeSpent).append(" ms; ");

            if (reportSample)
                builder.append("samples/sec: ").append(String.format("%.3f", samplesPerSec)).append("; ");

            if (reportBatch)
                builder.append("batches/sec: ").append(String.format("%.3f", batchesPerSec)).append("; ");

            if (reportScore)
                builder.append("score: ").append(model.score()).append(";");


            logger.info(builder.toString());
        }

        lastTime = System.currentTimeMillis();
    }

    public static class Builder {
        private int frequency = 1;

        private boolean reportScore;
        private boolean reportSample = true;
        private boolean reportBatch = true;
        private boolean reportIteration = true;
        private boolean reportTime = true;

        public Builder() {

        }

        /**
         * This method defines, if iteration number should be reported together with other data
         *
         * @param reallyReport
         * @return
         */
        public Builder reportIteration(boolean reallyReport) {
            this.reportIteration = reallyReport;
            return this;
        }

        /**
         * This method defines, if time per iteration should be reported together with other data
         *
         * @param reallyReport
         * @return
         */
        public Builder reportTime(boolean reallyReport) {
            this.reportTime = reallyReport;
            return this;
        }

        /**
         * This method defines, if samples/sec should be reported together with other data
         *
         * @param reallyReport
         * @return
         */
        public Builder reportSample(boolean reallyReport) {
            this.reportSample = reallyReport;
            return this;
        }


        /**
         * This method defines, if batches/sec should be reported together with other data
         *
         * @param reallyReport
         * @return
         */
        public Builder reportBatch(boolean reallyReport) {
            this.reportBatch = reallyReport;
            return this;
        }

        /**
         * This method defines, if score should be reported together with other data
         *
         * @param reallyReport
         * @return
         */
        public Builder reportScore(boolean reallyReport) {
            this.reportScore = reallyReport;
            return this;
        }

        /**
         * Desired IterationListener activation frequency
         *
         * @param frequency
         * @return
         */
        public Builder setFrequency(int frequency) {
            this.frequency = frequency;
            return this;
        }

        /**
         * This method returns configured PerformanceListener instance
         *
         * @return
         */
        public PerformanceListener build() {
            PerformanceListener listener = new PerformanceListener(frequency, reportScore);
            listener.reportIteration = this.reportIteration;
            listener.reportTime = this.reportTime;
            listener.reportBatch = this.reportBatch;
            listener.reportSample = this.reportSample;

            return listener;
        }
    }
}
