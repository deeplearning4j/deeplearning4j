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

package org.deeplearning4j.parallelism;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.parallelism.inference.InferenceMode;
import org.deeplearning4j.parallelism.inference.InferenceObservable;
import org.deeplearning4j.parallelism.inference.LoadBalanceMode;
import org.deeplearning4j.parallelism.inference.observers.BasicInferenceObservable;
import org.deeplearning4j.parallelism.inference.observers.BasicInferenceObserver;
import org.deeplearning4j.parallelism.inference.observers.BatchedInferenceObservable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.util.ArrayList;
import java.util.List;
import java.util.Observer;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * This class is simple wrapper for
 * ParallelInference using batched input
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class ParallelInference {
    protected Model model;
    protected long nanos;
    protected int workers;
    protected int batchLimit;
    protected InferenceMode inferenceMode;
    protected int queueLimit;
    protected LoadBalanceMode loadBalanceMode;

    // this queue holds data for inference
    private BlockingQueue<InferenceObservable> observables;

    private final Object locker = new Object();

    private InferenceWorker[] zoo;
    private ObservablesProvider provider;



    public final static int DEFAULT_NUM_WORKERS = Nd4j.getAffinityManager().getNumberOfDevices();
    public final static int DEFAULT_BATCH_LIMIT = 32;
    public final static InferenceMode DEFAULT_INFERENCE_MODE = InferenceMode.BATCHED;
    public final static int DEFAULT_QUEUE_LIMIT = 64;



    protected ParallelInference() {
        //
    }

    /**
     * This method allows to update Model used for inference in runtime, without queue reset
     *
     * @param model
     */
    public void updateModel(@NonNull Model model) {
        if (zoo != null) {
            for (val w: zoo)
                w.updateModel(model);
        } else {
            // if zoo wasn't initalized yet - just replace model
            this.model = model;
        }
    }

    /**
     * This method returns Models used in workers at this moment
     * PLEASE NOTE: This method is NOT thread safe, and should NOT be used anywhere but tests
     *
     * @return
     */
    protected Model[] getCurrentModelsFromWorkers() {
        if (zoo == null)
            return new Model[0];

        val models = new Model[zoo.length];
        int cnt = 0;
        for (val w:zoo) {
            models[cnt++] = w.replicatedModel;
        }

        return models;
    }

    protected void init() {
        observables = new LinkedBlockingQueue<>(queueLimit);

        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
        int currentDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        AtomicBoolean assignedRoot = new AtomicBoolean(false);

        zoo = new InferenceWorker[workers];
        for (int i = 0; i < workers; i++) {
            int cDevice = i % numDevices;
            boolean cRoot = !assignedRoot.get() && cDevice == currentDevice;
            assignedRoot.compareAndSet(false, cRoot);

            zoo[i] = new InferenceWorker(i, model, observables, cRoot);

            Nd4j.getAffinityManager().attachThreadToDevice(zoo[i], cDevice);
            zoo[i].setDaemon(true);
            zoo[i].start();
        }


        if (inferenceMode == InferenceMode.BATCHED) {
            log.info("Initializing ObservablesProvider...");
            provider = new ObservablesProvider(nanos, batchLimit, observables);
        }
    }

    protected long getWorkerCounter(int workerIdx) {
        return zoo[workerIdx].getCounterValue();
    }

    /**
     * This method gracefully shuts down ParallelInference instance
     */
    public synchronized void shutdown() {
        if (zoo == null)
            return;

        for (int e = 0; e < zoo.length; e++) {
            if (zoo[e] == null)
                continue;

            zoo[e].interrupt();
            zoo[e].shutdown();
            zoo[e] = null;
        }
        zoo = null;

        System.gc();
    }

    /**
     *
     * @param input
     * @return
     */
    public INDArray output(double[] input) {
        return output(Nd4j.create(input));
    }

    /**
     *
     * @param input
     * @return
     */
    public INDArray output(float[] input) {
        return output(Nd4j.create(input));
    }

    public INDArray output(INDArray input) {
        return output(input, null);
    }

    public INDArray output(INDArray input, INDArray inputMask){
        INDArray[] out = output(new INDArray[]{input}, (inputMask == null ? null : new INDArray[]{inputMask}));
        // basically, depending on model type we either
        // throw stuff to specific model, or wait for batch
        if(out.length != 1){
            throw new IllegalArgumentException("Network has multiple (" + out.length + ") output arrays, but only a" +
                    " single output can be returned using this method. Use for output(INDArray[] input, INDArray[] " +
                    "inputMasks) for multi-output nets");
        }
        return out[0];
    }

    /**
     *
     * @param dataSet
     * @return
     */
    public INDArray output(DataSet dataSet) {
        return output(dataSet.getFeatures(), dataSet.getFeaturesMaskArray());
    }

    /**
     * Generate predictions/output from the netwonk
     *
     * @param input Input to the network
     * @return Output from the network
     */
    public INDArray[] output(INDArray... input) {
        return output(input, null);
    }

    /**
     * Generate predictions/outputs from the network, optionally using input masks for predictions
     *
     * @param input      Input to the network
     * @param inputMasks Input masks for the network. May be null.
     * @return Output from the network
     */
    public INDArray[] output(INDArray[] input, INDArray[] inputMasks){
        // basically, depending on model type we either throw stuff to specific model, or wait for batch

        BasicInferenceObserver observer = new BasicInferenceObserver();
        InferenceObservable observable;

        if (inferenceMode == InferenceMode.SEQUENTIAL) {
            observable = new BasicInferenceObservable(input, inputMasks);
            observable.addObserver(observer);
            try {
                observables.put(observable);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException(e);
            }
        } else {
            observable = provider.setInput(observer, input, inputMasks);
        }

        try {
            // submit query to processing
            // and block until Observable returns
            //observer.wait();

            observer.waitTillDone();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        return observable.getOutput();
    }


    public static class Builder {
        private Model model;
        private int workers = DEFAULT_NUM_WORKERS;
        private int batchLimit = DEFAULT_BATCH_LIMIT;
        private InferenceMode inferenceMode = DEFAULT_INFERENCE_MODE;
        private int queueLimit = DEFAULT_QUEUE_LIMIT;
        protected LoadBalanceMode loadBalanceMode;

        public Builder(@NonNull Model model) {
            this.model = model;
        }


        /**
         * This method allows you to define mode that'll be used during inference. Options are:
         *
         * SEQUENTIAL: Input will be sent to last-used worker unmodified.
         * BATCHED: Multiple inputs will be packed into single batch, and
         * sent to last-used device.
         *
         * @param inferenceMode
         * @return
         */
        public Builder inferenceMode(@NonNull InferenceMode inferenceMode) {
            this.inferenceMode = inferenceMode;
            return this;
        }


        /**
         * This method allows you to specify load balance mode
         *
         * @param loadBalanceMode
         * @return
         */
        public Builder loadBalanceMode(@NonNull LoadBalanceMode loadBalanceMode) {
            this.loadBalanceMode = loadBalanceMode;
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
         * This method defines, how many input samples can
         * be batched within given time frame.
         *
         * PLEASE NOTE: This value has no effect in
         * SEQUENTIAL inference mode
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
         * This method defines buffer queue size.
         *
         * Default value: 64
         *
         * @param limit
         * @return
         */
        public Builder queueLimit(int limit) {
            if (limit < 1)
                throw new IllegalStateException("Queue limit should be positive value");

            this.queueLimit = limit;
            return this;
        }

        /**
         * This method builds new ParallelInference instance
         *
         * @return
         */
        public ParallelInference build() {
            if (this.inferenceMode == InferenceMode.INPLACE) {
                val inf = new InplaceParallelInference();
                inf.inferenceMode = this.inferenceMode;
                inf.model = this.model;
                inf.workers = this.workers;
                inf.loadBalanceMode = this.loadBalanceMode;

                inf.init();

                return inf;
            } else {
                ParallelInference inference = new ParallelInference();
                inference.batchLimit = this.batchLimit;
                inference.queueLimit = this.queueLimit;
                inference.inferenceMode = this.inferenceMode;
                inference.model = this.model;
                inference.workers = this.workers;
                inference.loadBalanceMode = this.loadBalanceMode;

                inference.init();

                return inference;
            }
        }
    }


    /**
     * This class actually does inference with respect to device affinity
     *
     */
    private class InferenceWorker extends Thread implements Runnable {
        private BlockingQueue<InferenceObservable> inputQueue;
        private AtomicBoolean shouldWork = new AtomicBoolean(true);
        private AtomicBoolean isStopped = new AtomicBoolean(false);
        private Model protoModel;
        private Model replicatedModel;
        private AtomicLong counter = new AtomicLong(0);
        private boolean rootDevice;

        private ReentrantReadWriteLock modelLock = new ReentrantReadWriteLock();

        private InferenceWorker(int id, @NonNull Model model, @NonNull BlockingQueue inputQueue, boolean rootDevice) {
            this.inputQueue = inputQueue;
            this.protoModel = model;
            this.rootDevice = rootDevice;

            this.setDaemon(true);
            this.setName("InferenceThread-" + id);

        }

        protected long getCounterValue() {
            return counter.get();
        }

        protected void updateModel(@NonNull Model model) {
            try {
                modelLock.writeLock().lock();
                this.protoModel = model;

                // now re-init model
                initializeReplicaModel();
            } finally {
                modelLock.writeLock().unlock();
            }
        }

        /**
         * This method duplicates model for future use during inference
         */
        protected void initializeReplicaModel() {
            if (protoModel instanceof ComputationGraph) {
                if (!rootDevice) {
                    this.replicatedModel = new ComputationGraph(ComputationGraphConfiguration
                            .fromJson(((ComputationGraph) protoModel).getConfiguration().toJson()));
                    this.replicatedModel.init();

                    synchronized (locker) {
                        this.replicatedModel.setParams(protoModel.params().unsafeDuplication(true));

                        Nd4j.getExecutioner().commit();
                    }
                } else {
                    this.replicatedModel = protoModel;
                }
            } else if (protoModel instanceof MultiLayerNetwork) {
                if (!rootDevice) {
                    this.replicatedModel = new MultiLayerNetwork(MultiLayerConfiguration.fromJson(
                            ((MultiLayerNetwork) protoModel).getLayerWiseConfigurations().toJson()));
                    this.replicatedModel.init();

                    synchronized (locker) {
                        this.replicatedModel.setParams(protoModel.params().unsafeDuplication(true));

                        Nd4j.getExecutioner().commit();
                    }
                } else {
                    this.replicatedModel = protoModel;
                }
            }
        }

        @Override
        public void run() {
            try {
                // model should be replicated & initialized here
                initializeReplicaModel();

                boolean isCG = replicatedModel instanceof  ComputationGraph;
                boolean isMLN = replicatedModel instanceof  MultiLayerNetwork;

                while (shouldWork.get()) {
                    InferenceObservable request = inputQueue.take();

                    if (request != null) {
                        counter.incrementAndGet();

                        // FIXME: get rid of instanceof here, model won't change during runtime anyway
                        if (isCG) {
                            List<Pair<INDArray[],INDArray[]>> batches = request.getInputBatches();
                            List<INDArray[]> out = new ArrayList<>(batches.size());
                            try {
                                for (Pair<INDArray[],INDArray[]> inBatch : batches) {
                                    try {
                                        modelLock.readLock().lock();

                                        INDArray[] output = ((ComputationGraph) replicatedModel).output(false, inBatch.getFirst(), inBatch.getSecond());
                                        out.add(output);
                                    } finally {
                                        modelLock.readLock().unlock();
                                    }

                                }
                                request.setOutputBatches(out);
                            } catch (Exception e){
                                request.setOutputException(e);
                            }
                        } else if (isMLN) {
                            List<Pair<INDArray[],INDArray[]>> batches = request.getInputBatches();
                            List<INDArray[]> out = new ArrayList<>(batches.size());
                            try {
                                for (Pair<INDArray[],INDArray[]> inBatch : batches) {
                                    INDArray f = inBatch.getFirst()[0];
                                    INDArray fm = (inBatch.getSecond() == null ? null : inBatch.getSecond()[0]);
                                    try {
                                        modelLock.readLock().lock();

                                        INDArray output = ((MultiLayerNetwork) replicatedModel).output(f, false, fm, null);
                                        out.add(new INDArray[]{output});
                                    } finally {
                                        modelLock.readLock().unlock();
                                    }
                                }
                                request.setOutputBatches(out);
                            } catch (Exception e){
                                request.setOutputException(e);
                            }
                        }


                    } else {
                        // just do nothing, i guess and hope for next round?
                    }
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                // do nothing
            } catch (Exception e) {
                throw new RuntimeException(e);
            } finally {
                isStopped.set(true);
            }
        }

        protected void shutdown() {
            shouldWork.set(false);
            while (!isStopped.get()) {
                // block until main loop is finished
            }
        }
    }


    protected static class ObservablesProvider {
        private BlockingQueue<InferenceObservable> targetQueue;
        private long nanos;
        private int batchLimit;

        private volatile BatchedInferenceObservable currentObservable;
        private final Object locker = new Object();

        protected ObservablesProvider(long nanos, int batchLimit, @NonNull BlockingQueue<InferenceObservable> queue) {
            this.targetQueue = queue;
            this.nanos = nanos;
            this.batchLimit = batchLimit;
        }

        protected InferenceObservable setInput(@NonNull Observer observer, INDArray input){
            return setInput(observer, new INDArray[]{input}, null);
        }

        protected InferenceObservable setInput(@NonNull Observer observer, INDArray... input){
            return setInput(observer, input, null);
        }

        protected InferenceObservable setInput(@NonNull Observer observer, INDArray[] input, INDArray[] inputMask) {
            synchronized (locker) {
                boolean isNew = false;
                if (currentObservable == null || currentObservable.getCounter() >= batchLimit
                                || currentObservable.isLocked()) {
                    isNew = true;
                    currentObservable = new BatchedInferenceObservable();
                }

                currentObservable.addInput(input, inputMask);
                currentObservable.addObserver(observer);

                try {
                    if (isNew)
                        targetQueue.put(currentObservable);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new RuntimeException(e);
                }

                return currentObservable;
            }
        }
    }
}
