package org.deeplearning4j.parallelism;

import lombok.NonNull;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * This class is simple wrapper for ParallelInference using batched input
 *
 * @author raver119@gmail.com
 */
public class ParallelInference {
    private Model model;
    private List<String> labels;
    private long nanos;
    private int workers;
    private int batchLimit;
    private InferenceMode inferenceMode;


    private AtomicLong sequenceId = new AtomicLong(0);


    public enum InferenceMode {
        SEQUENTIAL, // input will be passed into the model as is
        BATCHED, // input will be included into the batch
    }

    protected ParallelInference() {
        //
    }


    public INDArray output(double[] input) {
        return output(Nd4j.create(input));
    }

    public INDArray output(float[] input) {
        return output(Nd4j.create(input));
    }

    public INDArray output(INDArray input) {


        // basically, depending on model type we either throw stuff to specific model, or wait for batch

        return null;
    }


    public static class Builder {
        private Model model;
        private List<String> labels;
        private long nanos;
        private int workers = Nd4j.getAffinityManager().getNumberOfDevices();
        private int batchLimit = 32;
        private InferenceMode inferenceMode = InferenceMode.SEQUENTIAL;

        public Builder(@NonNull ComputationGraph model) {
            this.model = model;
        }

        public Builder(@NonNull MultiLayerNetwork model) {
            this.model = model;
        }

        /**
         * This method allows you to define mode that'll be used during inference. Options are:
         *
         * SEQUENTIAL: Input will be sent to last-used worker unmodified.
         * BATCHED: Multiple inputs will be packed into single batch, and sent to last-used device.
         *
         * @param inferenceMode
         * @return
         */
        public Builder inferenceModel(@NonNull InferenceMode inferenceMode){
            this.inferenceMode = inferenceMode;
            return this;
        }


        /**
         * This method allows you to specify String labels that'll be used as output to your input
         *
         * PLEASE NOTE: This method is optional, and applies to classification models only
         *
         * @param labels
         * @return
         */
        public Builder labels(@NonNull List<String> labels) {
            this.labels = labels;
            return this;
        }

        /**
         * This method defines, how long model will wait to fulfill the batch before sending it out to model
         *
         * PLEASE NOTE: This value has no effect in SEQUENTIAL inference mode
         *
         * @param nanos
         * @return
         */
        public Builder timeoutNanos(long nanos) {
            if (nanos < 1)
                throw new IllegalStateException("Timeout should be positive value");

            this.nanos = nanos;
            return this;
        }

        /**
         * This method defines, how many model copies will be used for inference.
         *
         * PLEASE NOTE: This method primarily suited for multi-GPU systems
         *
         * @param workers
         * @return
         */
        public Builder workers(int workers) {
            if (workers < 1)
                throw new IllegalStateException("Workers should be positive value");

            this.workers = workers;
            return this;
        }

        /**
         * This method defines, how many input samples can be batched within given time frame.
         *
         * PLEASE NOTE: This value has no effect in SEQUENTIAL inference mode
         *
         * @param limit
         * @return
         */
        public Builder batchLimit(int limit) {
            if (limit < 1)
                throw new IllegalStateException("Batch limit should be positive value");

            this.batchLimit = limit;
            return this;
        }

        /**
         * This method builds new ParallelInference instance
         *
         * @return
         */
        public ParallelInference build() {
            ParallelInference inference = new ParallelInference();

            return inference;
        }
    }


    /**
     * This class actually does inference with respect to device affinity
     *
     */
    private class InferenceWorker extends Thread implements Runnable {
        private BlockingQueue<INDArray> inputQueue;
        private AtomicBoolean shouldWork = new AtomicBoolean(true);
        private AtomicBoolean isStopped = new AtomicBoolean(false);
        private Model protoModel;
        private ComputationGraph replicatedModel;

        private InferenceWorker (int id) {

            this.setDaemon(true);
            this.setName("InferenceThread-"+id);
        }

        @Override
        public void run() {
            try {
                // model should be replicated & initialized here

                while (shouldWork.get()) {
                    INDArray array = inputQueue.poll(100, TimeUnit.NANOSECONDS);

                    if (array != null) {
                        INDArray[] output = replicatedModel.output(false, array);
                    } else {
                        // just do nothing, i guess and hope for next round?
                    }
                }
            } catch (InterruptedException e) {
                // do nothing
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
            isStopped.set(true);
        }

        protected void shutdown() {
            shouldWork.set(false);
            while (!isStopped.get()){
                // block until main loop is finished
            }
        }
    }

    /**
     * This class holds reference
     */
    private static class InferenceQuery {
        private INDArray[] input;
        private long id;

        private InferenceQuery(INDArray... inputs) {
            this.input = inputs;
        }
    }


    private static class InferenceFuture implements Future<INDArray> {

        @Override
        public boolean cancel(boolean mayInterruptIfRunning) {
            return false;
        }

        @Override
        public boolean isCancelled() {
            return false;
        }

        @Override
        public boolean isDone() {
            return false;
        }

        @Override
        public INDArray get() throws InterruptedException, ExecutionException {
            return null;
        }

        @Override
        public INDArray get(long timeout, @NotNull TimeUnit unit) throws InterruptedException, ExecutionException, TimeoutException {
            return null;
        }
    }
}
