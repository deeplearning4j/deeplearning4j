/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.optimize.listeners;

import com.google.common.base.Preconditions;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.Serializable;

/**
 * Simple IterationListener that tracks time spend on training per iteration.
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class PerformanceListener extends BaseTrainingListener implements Serializable {
    private final int frequency;
    private transient ThreadLocal<Double> samplesPerSec = new ThreadLocal<>();
    private transient ThreadLocal<Double> batchesPerSec = new ThreadLocal<>();
    private transient ThreadLocal<Long> lastTime = new ThreadLocal<>();

    private boolean reportScore;
    private boolean reportSample = true;
    private boolean reportBatch = true;
    private boolean reportIteration = true;
    private boolean reportEtl = true;
    private boolean reportTime = true;



    public PerformanceListener(int frequency) {
        this(frequency, false);
    }

    public PerformanceListener(int frequency, boolean reportScore) {
        Preconditions.checkArgument(frequency > 0, "Invalid frequency, must be > 0: Got " + frequency);
        this.frequency = frequency;
        this.reportScore = reportScore;

        lastTime.set(System.currentTimeMillis());
    }

    @Override
    public void iterationDone(Model model, int iteration, int epoch) {
        // we update lastTime on every iteration
        // just to simplify things
        if (lastTime.get() == null)
            lastTime.set(System.currentTimeMillis());

        if (samplesPerSec.get() == null)
            samplesPerSec.set(0.0);

        if (batchesPerSec.get() == null)
            batchesPerSec.set(0.0);

        if (iteration % frequency == 0) {
            long currentTime = System.currentTimeMillis();

            long timeSpent = currentTime - lastTime.get();
            float timeSec = timeSpent / 1000f;

            INDArray input;
            if (model instanceof ComputationGraph) {
                // for comp graph (with multidataset
                ComputationGraph cg = (ComputationGraph) model;
                INDArray[] inputs = cg.getInputs();

                if (inputs != null && inputs.length > 0)
                    input = inputs[0];
                else
                    input = model.input();
            } else {
                input = model.input();
            }

            //            long tadLength = Shape.getTADLength(input.shape(), ArrayUtil.range(1, input.rank()));

            long numSamples = input.size(0);

            samplesPerSec.set((double) (numSamples / timeSec));
            batchesPerSec.set((double) (1 / timeSec));


            StringBuilder builder = new StringBuilder();

            if (Nd4j.getAffinityManager().getNumberOfDevices() > 1)
                builder.append("Device: [").append(Nd4j.getAffinityManager().getDeviceForCurrentThread()).append("]; ");

            if (reportEtl) {
                long time = (model instanceof MultiLayerNetwork) ? ((MultiLayerNetwork) model).getLastEtlTime()
                                : ((ComputationGraph) model).getLastEtlTime();
                builder.append("ETL: ").append(time).append(" ms; ");
            }

            if (reportIteration)
                builder.append("iteration ").append(iteration).append("; ");

            if (reportTime)
                builder.append("iteration time: ").append(timeSpent).append(" ms; ");

            if (reportSample)
                builder.append("samples/sec: ").append(String.format("%.3f", samplesPerSec.get())).append("; ");

            if (reportBatch)
                builder.append("batches/sec: ").append(String.format("%.3f", batchesPerSec.get())).append("; ");

            if (reportScore)
                builder.append("score: ").append(model.score()).append(";");


            log.info(builder.toString());
        }

        lastTime.set(System.currentTimeMillis());
    }

    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
        //Custom deserializer, as transient ThreadLocal fields won't be initialized...
        in.defaultReadObject();
        samplesPerSec = new ThreadLocal<>();
        batchesPerSec = new ThreadLocal<>();
        lastTime = new ThreadLocal<>();
    }

    public static class Builder {
        private int frequency = 1;

        private boolean reportScore;
        private boolean reportSample = true;
        private boolean reportBatch = true;
        private boolean reportIteration = true;
        private boolean reportTime = true;
        private boolean reportEtl = true;

        public Builder() {

        }

        /**
         * This method defines, if iteration number should be reported together with other data
         *
         * @param reportIteration
         * @return
         */
        public Builder reportIteration(boolean reportIteration) {
            this.reportIteration = reportIteration;
            return this;
        }

        /**
         * This method defines, if time per iteration should be reported together with other data
         *
         * @param reportTime
         * @return
         */
        public Builder reportTime(boolean reportTime) {
            this.reportTime = reportTime;
            return this;
        }

        /**
         * This method defines, if ETL time per iteration should be reported together with other data
         *
         * @param reportEtl
         * @return
         */
        public Builder reportETL(boolean reportEtl) {
            this.reportEtl = reportEtl;
            return this;
        }

        /**
         * This method defines, if samples/sec should be reported together with other data
         *
         * @param reportSample
         * @return
         */
        public Builder reportSample(boolean reportSample) {
            this.reportSample = reportSample;
            return this;
        }


        /**
         * This method defines, if batches/sec should be reported together with other data
         *
         * @param reportBatch
         * @return
         */
        public Builder reportBatch(boolean reportBatch) {
            this.reportBatch = reportBatch;
            return this;
        }

        /**
         * This method defines, if score should be reported together with other data
         *
         * @param reportScore
         * @return
         */
        public Builder reportScore(boolean reportScore) {
            this.reportScore = reportScore;
            return this;
        }

        /**
         * Desired TrainingListener activation frequency
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
            listener.reportEtl = this.reportEtl;

            return listener;
        }
    }
}
