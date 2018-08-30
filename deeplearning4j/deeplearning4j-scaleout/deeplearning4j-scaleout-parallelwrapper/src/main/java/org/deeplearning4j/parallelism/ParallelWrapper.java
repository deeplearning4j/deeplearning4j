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

import lombok.Data;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.api.storage.StatsStorageRouter;
import org.deeplearning4j.api.storage.listener.RoutingIterationListener;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.datasets.iterator.AsyncMultiDataSetIterator;
import org.deeplearning4j.datasets.iterator.callbacks.InterleavedDataSetCallback;
import org.deeplearning4j.exception.DL4JInvalidConfigException;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.updater.graph.ComputationGraphUpdater;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.optimize.listeners.SharedGradient;
import org.deeplearning4j.optimize.solvers.accumulation.EncodedGradientsAccumulator;
import org.deeplearning4j.optimize.solvers.accumulation.GradientsAccumulator;
import org.deeplearning4j.optimize.solvers.accumulation.Registerable;
import org.deeplearning4j.parallelism.factory.DefaultTrainerContext;
import org.deeplearning4j.parallelism.factory.SymmetricTrainerContext;
import org.deeplearning4j.parallelism.factory.TrainerContext;
import org.deeplearning4j.parallelism.trainer.Trainer;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * This is simple data-parallel wrapper
 * suitable for multi-cpu/multi-gpu environments.
 *
 * PLEASE NOTE: This implementation is NOT NUMA-aware.
 *
 * @author raver119@gmail.com
 */
// TODO: We want this thing to be NUMA-aware in foreseeable future
@Slf4j
@Data
public class ParallelWrapper implements AutoCloseable {
    public enum TrainingMode {
        /**
         * Averaging every X epochs will be applied
         */
        AVERAGING,

        /**
         * Models within ParallelWrapper instance will share gradients updates
         */
        SHARED_GRADIENTS,

        /**
         * This option assumes use of GradientsAccumulator with any MessageHandler
         */
        CUSTOM,
    }

    protected AtomicBoolean exceptionEncountered;
    protected Throwable exception;
    protected final String uuid = java.util.UUID.randomUUID().toString();
    protected Model model;
    protected int workers = 2;
    protected int prefetchSize = 2;
    protected int averagingFrequency = 1;
    protected Trainer[] zoo;
    protected TrainerContext trainerContext;
    protected AtomicLong iterationsCounter = new AtomicLong(0);
    protected boolean reportScore = false;
    protected boolean averageUpdaters = true;
    protected boolean legacyAveraging = false;
    protected boolean wasAveraged = false;
    protected AtomicBoolean stopFit = new AtomicBoolean(false);
    protected List<TrainingListener> listeners = new ArrayList<>();
    protected StatsStorageRouter storageRouter;
    protected boolean isMQ;
    protected WorkspaceMode workspaceMode;
    protected Object[] trainerContextArgs;
    protected boolean debug = false;

    protected ThreadPoolExecutor executorService;

    protected final AtomicInteger workerCounter = new AtomicInteger(0);
    @Getter
    @Setter
    protected GradientsAccumulator gradientsAccumulator;

    // log uncaught exceptions
    Thread.UncaughtExceptionHandler handler = new Thread.UncaughtExceptionHandler() {
        public void uncaughtException(Thread th, Throwable ex) {
            log.error("Uncaught exception: " + ex);
            ex.printStackTrace();
            if(exceptionEncountered != null){
                exceptionEncountered.set(true);
                exception = ex;
            }
        }
    };

    protected ParallelWrapper(Model model, int workers, int prefetchSize) {
        this.model = model;
        this.workers = workers;
        this.prefetchSize = prefetchSize;

        if (this.model instanceof MultiLayerNetwork) {
            ((MultiLayerNetwork) this.model).getUpdater();
        } else if (this.model instanceof ComputationGraph) {
            ((ComputationGraph) this.model).getUpdater();
        }
    }

    protected void init() {
        workerCounter.set(0);
        this.executorService = (ThreadPoolExecutor) Executors.newFixedThreadPool(workers, new ThreadFactory() {
            @Override
            public Thread newThread(@NotNull Runnable r) {
                Thread t = Executors.defaultThreadFactory().newThread(r);

                int cThread = workerCounter.getAndIncrement();

                t.setName("ParallelWrapper training thread " + cThread);
                t.setDaemon(true);
                t.setUncaughtExceptionHandler(handler);

                Nd4j.getAffinityManager().attachThreadToDevice(t,
                                cThread % Nd4j.getAffinityManager().getNumberOfDevices());

                return t;
            }
        });
    }

    @Override
    public void close() throws Exception {
        if (zoo != null) {
            for (int i = 0; i < zoo.length; i++) {
                if (zoo[i] != null)
                    zoo[i].shutdown();
            }
            zoo = null;
        }

        if (executorService != null) {
            executorService.shutdown();
            executorService = null;
        }

        if (gradientsAccumulator != null)
            gradientsAccumulator.reset();
    }

    /**
     * This method causes all threads used for parallel training to stop
     */
    public synchronized void shutdown() {
        try {
            close();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Will stop a fit operation from continuing to iterate.
     */
    public void stopFit() {
        stopFit.set(true);
    }

    /**
     *
     * @param source
     */
    public synchronized void fit(@NonNull MultiDataSetIterator source) {
        stopFit.set(false);
        createZooIfNeccessary(true);

        if (source.resetSupported())
            source.reset();

        MultiDataSetIterator iterator = source;
        if (prefetchSize > 0 && source.asyncSupported()) {
            if (isMQ) {
                if (workers % Nd4j.getAffinityManager().getNumberOfDevices() != 0)
                    log.warn("Number of workers [{}] isn't optimal for available devices [{}]", workers,
                                    Nd4j.getAffinityManager().getNumberOfDevices());

                iterator = new AsyncMultiDataSetIterator(source, prefetchSize,
                                new LinkedBlockingQueue<>(prefetchSize * workers), true,
                                new InterleavedDataSetCallback(prefetchSize * 2));
            } else
                iterator = new AsyncMultiDataSetIterator(source, prefetchSize);
        }

        AtomicInteger locker = new AtomicInteger(0);

        long time1 = System.currentTimeMillis();
        while (iterator.hasNext() && !stopFit.get()) {
            MultiDataSet dataSet = iterator.next();
            long time2 = System.currentTimeMillis();

            if (dataSet == null)
                throw new ND4JIllegalStateException("You can't have NULL as MultiDataSet");

            /*
             now dataSet should be dispatched to next free workers, until all workers are busy. And then we should block till all finished.
            */
            int pos = locker.getAndIncrement();
            zoo[pos].feedMultiDataSet(dataSet, time2 - time1);

            /*
                if all workers are dispatched now, join till all are finished
            */
            if (pos + 1 == workers) {
                iterationsCounter.incrementAndGet();

                /*
                    if we're using registerable accumulator (i.e. we're on spark or cuda with gradients sharing),
                    update it & notify about number of threads in this training round then
                  */
                if (gradientsAccumulator != null && gradientsAccumulator instanceof Registerable) {
                    ((Registerable) gradientsAccumulator).registerConsumers(workers);
                }

                if (zoo[0].averagingRequired()) {
                    for (int cnt = 0; cnt < workers && cnt < locker.get(); cnt++) {
                        try {
                            zoo[cnt].waitTillRunning();
                        } catch (Exception e) {
                            throw new RuntimeException(e);
                        }
                    }

                    Nd4j.getMemoryManager().invokeGcOccasionally();

                    /*
                        average model, and propagate it to whole
                    */
                    if (iterationsCounter.get() % averagingFrequency == 0 && pos + 1 == workers
                                    && zoo[0].averagingRequired()) {
                        // averaging model
                        double score = getScore(locker);

                        // averaging updaters state
                        averageUpdatersState(locker, score);
                    }
                }

                locker.set(0);
            }

            time1 = System.currentTimeMillis();
        }

        // launch last update
        if (locker.get() != 0 && gradientsAccumulator != null && gradientsAccumulator instanceof Registerable) {
            ((Registerable) gradientsAccumulator).registerConsumers(locker.get());
        }


        if (debug)
            log.info("Stopping everyone...");

        // ensure all threads stopped processing
        for (int cnt = 0; cnt < workers; cnt++) {
            try {
                zoo[cnt].waitTillRunning();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        if (debug)
            log.info("Shutting down iterator...");

        if (prefetchSize > 0 && source.asyncSupported())
            ((AsyncMultiDataSetIterator) iterator).shutdown();

        /*
        // TODO: get rid of this code, 0 model is not replicated anyway
        // now we transfer models back from workers
        List<Model> models = new ArrayList<>();
        for (int i = 0; i < zoo.length; i++) {
            models.add(zoo[0].getModel());
        }
        
        // actual transfer code depends on trainer
        trainerContext.finalizeTraining(model, models.toArray(new Model[0]));
        */
        try {
            close();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        // sanity checks, or the dataset may never average
        if (!wasAveraged)
            log.warn("Parameters were never averaged on current fit(). Ratios of batch size, num workers, and averaging frequency may be responsible.");
        //            throw new IllegalStateException("Parameters were never averaged. Please check batch size ratios, number of workers, and your averaging frequency.");

        log.debug("Iterations passed: {}", iterationsCounter.get());
        //        iterationsCounter.set(0);
    }

    private double getScore(AtomicInteger locker) {
        wasAveraged = true;
        double score = 0.0;

        List<INDArray> params = new ArrayList<>();
        for (int cnt = 0; cnt < workers && cnt < locker.get(); cnt++) {
            params.add(zoo[cnt].getModel().params());
            score += zoo[cnt].getModel().score();
        }

        Nd4j.averageAndPropagate(null, params);


        score /= Math.min(workers, locker.get());

        // TODO: improve this
        if (reportScore)
            log.info("Averaged score: " + score);

        return score;
    }

    private void averageUpdatersState(AtomicInteger locker, double score) {
        // averaging updaters state
        if (model instanceof MultiLayerNetwork) {
            if (averageUpdaters) {
                Updater updater = ((MultiLayerNetwork) model).getUpdater();
                int batchSize = 0;

                if (updater != null && updater.getStateViewArray() != null) {
                    List<INDArray> updaters = new ArrayList<>();
                    for (int cnt = 0; cnt < workers && cnt < locker.get(); cnt++) {
                        MultiLayerNetwork workerModel = (MultiLayerNetwork) zoo[cnt].getModel();
                        updaters.add(workerModel.getUpdater().getStateViewArray());
                        batchSize += workerModel.batchSize();
                    }

                    Nd4j.averageAndPropagate(null, updaters);
                }
            }

            ((MultiLayerNetwork) model).setScore(score);
        } else if (model instanceof ComputationGraph) {
            if (averageUpdaters) {
                ComputationGraphUpdater updater = ((ComputationGraph) model).getUpdater();
                int batchSize = 0;

                if (updater != null && updater.getStateViewArray() != null) {
                    List<INDArray> updaters = new ArrayList<>();
                    for (int cnt = 0; cnt < workers && cnt < locker.get(); cnt++) {
                        ComputationGraph workerModel = (ComputationGraph) zoo[cnt].getModel();
                        updaters.add(workerModel.getUpdater().getStateViewArray());
                        batchSize += workerModel.batchSize();
                    }
                    Nd4j.averageAndPropagate(null, updaters);
                }
            }

            ((ComputationGraph) model).setScore(score);
        }
    }


    /**
     * This method allows you to specify trainingListeners for this model.
     * Note that for listeners like StatsListener (that have state that will be sent somewhere), consider instead
     * using {@link #setListeners(StatsStorageRouter, Collection)}
     *
     * @param listeners    Listeners to set
     */
    public void setListeners(@NonNull Collection<TrainingListener> listeners) {
        setListeners(null, listeners);
    }

    /**
     * This method allows you to specify trainingListeners for this model.
     * Note that for listeners like StatsListener (that have state that will be sent somewhere), consider instead
     * using {@link #setListeners(StatsStorageRouter, Collection)}
     *
     * @param listeners    Listeners to set
     */
    public void setListeners(@NonNull TrainingListener... listeners) {
        setListeners(Arrays.asList(listeners));
    }

    /**
     * Set the listeners, along with a StatsStorageRouter that the results will be shuffled to (in the case of any listeners
     * that implement the {@link RoutingIterationListener} interface)
     *
     * @param statsStorage Stats storage router to place the results into
     * @param listeners    Listeners to set
     */
    public void setListeners(StatsStorageRouter statsStorage, TrainingListener... listeners) {
        setListeners(statsStorage, Arrays.asList(listeners));
    }

    /**
     * Set the listeners, along with a StatsStorageRouter that the results will be shuffled to (in the case of any listeners
     * that implement the {@link RoutingIterationListener} interface)
     *
     * @param statsStorage Stats storage router to place the results into
     * @param listeners    Listeners to set
     */
    public void setListeners(StatsStorageRouter statsStorage, Collection<? extends TrainingListener> listeners) {
        //Check if we have any RoutingIterationListener instances that need a StatsStorage implementation...
        if (listeners != null) {
            for (TrainingListener l : listeners) {
                if (l instanceof RoutingIterationListener) {
                    RoutingIterationListener rl = (RoutingIterationListener) l;
                    if (statsStorage == null && rl.getStorageRouter() == null) {
                        log.warn("RoutingIterationListener provided without providing any StatsStorage instance. Iterator may not function without one. Listener: {}",
                                        l);
                    }
                }
            }

            this.listeners.addAll(listeners);
        } else {
            this.listeners.clear();
        }

        this.storageRouter = statsStorage;
    }

    /**
     * This method will propagate gradients across all workers
     *
     * @param gradients
     */
    public void broadcastGradients(SharedGradient gradients) {
        // TODO: add implementation
        /*
            Basically all we want here is:
            1) Ensure length matches parameters length
            2) Ensure data is acessible from all devices somehow (i.e. it's in HOST-only mode
         */
        /*
        if (zoo[0] instanceof CommunicativeTrainer) {
            for (int i = 0; i < zoo.length; i++) {
                ((CommunicativeTrainer) zoo[i]).enqueueGradient(gradients);
            }
        }
        */
    }


    /**
     * This method takes DataSetIterator, and starts training over it by scheduling DataSets to different executors
     *
     * @param source
     */
    public synchronized void fit(@NonNull DataSetIterator source) {
        log.info("Using workspaceMode {} for training", workspaceMode.name());
        stopFit.set(false);
        createZooIfNeccessary(false);



        if (source.resetSupported())
            source.reset();

        DataSetIterator iterator = source;

        if (prefetchSize > 0 && source.asyncSupported()) {
            log.info("Creating asynchronous prefetcher...");
            if (isMQ) {
                if (workers % Nd4j.getAffinityManager().getNumberOfDevices() != 0)
                    log.warn("Number of workers [{}] isn't optimal for available devices [{}]", workers,
                                    Nd4j.getAffinityManager().getNumberOfDevices());

                iterator = new AsyncDataSetIterator(source, prefetchSize,
                                new LinkedBlockingQueue<>(prefetchSize * workers), true,
                                new InterleavedDataSetCallback(prefetchSize * 2));

            } else
                iterator = new AsyncDataSetIterator(source, prefetchSize);
        }


        List<Long> nanos = new ArrayList<>();
        AtomicInteger locker = new AtomicInteger(0);
        long time1 = System.currentTimeMillis();
        log.info("Starting ParallelWrapper training round...");
        long intcnt = 0;
        while (iterator.hasNext() && !stopFit.get()) {
            //while (intcnt < 1000) {
            intcnt++;
            DataSet dataSet = iterator.next();
            long time2 = System.currentTimeMillis();
            long lastEtlTime = time2 - time1;
            //nanos.add((time2 - time1));

            if (dataSet == null)
                throw new ND4JIllegalStateException("You can't have NULL as DataSet");

            /*
             now dataSet should be dispatched to next free workers, until all workers are busy. And then we should block till all finished.
            */
            int pos = locker.getAndIncrement();

            if (debug)
                log.info("Feeding dataset {} to worker {}", intcnt, pos);

            if (zoo == null)
                throw new IllegalStateException(
                                "ParallelWrapper.shutdown() has been called too early and will fail from this point forward.");

            zoo[pos].feedDataSet(dataSet, lastEtlTime);

            /*
                if all workers are dispatched now, join till all are finished
            */
            if (pos + 1 == workers) {
                iterationsCounter.incrementAndGet();

                /*
                    if we're using registerable accumulator (i.e. we're on spark or cuda with gradients sharing),
                    update it & notify about number of threads in this training round then
                  */
                if (gradientsAccumulator != null && gradientsAccumulator instanceof Registerable) {
                    ((Registerable) gradientsAccumulator).registerConsumers(workers);
                }

                if (zoo[0].averagingRequired()) {
                    for (int cnt = 0; cnt < workers && cnt < locker.get(); cnt++) {
                        try {
                            zoo[cnt].waitTillRunning();
                        } catch (Exception e) {
                            throw new RuntimeException(e);
                        }
                    }

                    Nd4j.getMemoryManager().invokeGcOccasionally();

                    /*
                        average model, and propagate it to whole
                    */
                    if (iterationsCounter.get() % averagingFrequency == 0 && pos + 1 == workers
                                    && zoo[0].averagingRequired()) {
                        long timeA1 = System.currentTimeMillis();

                        // model averaging happens within
                        double score = getScore(locker);

                        // updaters averging happens within (if any)
                        averageUpdatersState(locker, score);

                        long timeA2 = System.currentTimeMillis();
                        if (reportScore)
                            log.info("Averaging time: {} ms", timeA2 - timeA1);
                    }

                }
                locker.set(0);
            }

            time1 = System.currentTimeMillis();
        }

        // launch last update
        if (locker.get() != 0 && gradientsAccumulator != null && gradientsAccumulator instanceof Registerable) {
            //log.info("Finalizing process: {}", locker.get());
            ((Registerable) gradientsAccumulator).registerConsumers(locker.get());
        }

        if (debug)
            log.info("Stopping everyone...");

        // ensure all threads stopped processing
        for (int cnt = 0; cnt < workers; cnt++) {
            try {
                zoo[cnt].waitTillRunning();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        if (debug)
            log.info("Shutting down iterator...");

        if (prefetchSize > 0 && source.asyncSupported())
            ((AsyncDataSetIterator) iterator).shutdown();

        // TODO: get rid of this code, 0 model is not replicated anyway
        // now we transfer models back from workers
        /*
        List<Model> models = new ArrayList<>();
        for (int i = 0; i < zoo.length; i++) {
            models.add(zoo[0].getModel());
        }
        
        // actual transfer code depends on trainer
        trainerContext.finalizeTraining(model, models.toArray(new Model[0]));
        */

        try {
            close();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        if (debug)
            log.info("Iterations passed: {}", iterationsCounter.get());
    }


    private void createZooIfNeccessary(boolean useMDS) {
        if (zoo == null) {
            trainerContext.init(model, trainerContextArgs);

            zoo = new Trainer[workers];
            int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
            for (int cnt = 0; cnt < workers; cnt++) {
                // we pass true here, to tell Trainer to use MultiDataSet queue for training
                zoo[cnt] = trainerContext.create(this.uuid, cnt, model, Nd4j.getAffinityManager().getDeviceForCurrentThread(),
                                useMDS, this, workspaceMode, averagingFrequency);

                /*
                zoo[cnt].setUncaughtExceptionHandler(handler);
                
                if (zoo[cnt] instanceof Thread) {
                    Nd4j.getAffinityManager().attachThreadToDevice((Thread) zoo[cnt], cnt % numDevices);
                }
                zoo[cnt].start();
                */

                if (executorService == null)
                    init();

                executorService.execute(zoo[cnt]);
            }
        }
    }

    public static class Builder<T extends Model> {
        protected TrainingMode trainingMode = TrainingMode.AVERAGING;
        protected T model;
        protected int workers = Nd4j.getAffinityManager().getNumberOfDevices();
        protected int prefetchSize = 16;
        protected int averagingFrequency = 1;
        protected boolean reportScore = false;
        protected boolean averageUpdaters = true;
        protected boolean legacyAveraging = true;
        protected boolean isMQ = Nd4j.getAffinityManager().getNumberOfDevices() > 1;
        protected TrainerContext trainerContext = new DefaultTrainerContext();
        protected Object[] trainerContextArgs;
        protected WorkspaceMode workspaceMode = WorkspaceMode.ENABLED;

        protected GradientsAccumulator accumulator;

        /**
         * Transer context args are for calling a
         * {@link TrainerContext} init method
         * when {@link ParallelWrapper} starts training
         * @param trainerContextArgs the args to use (maybe null)
         * @return
         */
        public Builder trainerContextArgs(Object... trainerContextArgs) {
            this.trainerContextArgs = trainerContextArgs;
            return this;
        }

        /**
         * Specify a {@link TrainerContext}
         * for the given {@link ParallelWrapper}
         * instance.
         * Defaults to {@link DefaultTrainerContext}
         * otherwise
         * @param trainerContext the trainer factory to use
         * @return builder pattern
         */
        public Builder trainerFactory(@NonNull TrainerContext trainerContext) {
            this.trainerContext = trainerContext;
            return this;
        }

        public Builder workspaceMode(@NonNull WorkspaceMode mode) {
            this.workspaceMode = mode;
            return this;
        }

        /**
         * Build ParallelWrapper for MultiLayerNetwork
         *
         * @param model
         */
        public Builder(@NonNull T model) {
            this.model = model;
        }

        /**
         * This method allows to configure number of workers that'll be used for parallel training
         *
         * @param num
         * @return
         */
        public Builder workers(int num) {
            if (num < 2)
                throw new RuntimeException("Number of workers can't be lower then 2!");

            this.workers = num;
            return this;
        }

        /**
         * Model averaging frequency.
         *
         * @param freq number of iterations between averaging
         * @return
         */
        public Builder averagingFrequency(int freq) {
            if (freq < 0)
                freq = 0;

            this.averagingFrequency = freq;
            return this;
        }

        /**
         * This method enables/disables updaters averaging.
         *
         * Default value: TRUE
         *
         * PLEASE NOTE: This method is suitable for debugging purposes mostly. So don't change default value, unless you're sure why you need it.
         * PLEASE NOTE: This method is suitable for parameters averaging training only. For gradients sharing mechanism it'll be ignored
         *
         * @param reallyAverage
         * @return
         */
        public Builder averageUpdaters(boolean reallyAverage) {
            this.averageUpdaters = reallyAverage;
            return this;
        }


        /**
         * Size of prefetch buffer that will be used for background data prefetching.
         * Usually it's better to keep this value equal to the number of workers.
         *
         * Default value: 2
         *
         * @param size 0 to disable prefetching, any positive number
         * @return
         */
        public Builder prefetchBuffer(int size) {
            if (size < 0)
                size = 0;

            this.prefetchSize = size;

            return this;
        }

        /**
         *  This method allows you to specify training mode for this instance of PW.<br>
         *  1) AVERAGING - stands for parameters averaging. Each X epochs weights and updaters state will be averaged across all models<br>
         *  2) SHARED_GRADIENTS - stands for gradients sharing - more details available here: <a href="https://deeplearning4j.org/distributed">https://deeplearning4j.org/distributed</a><br>
         *  3) CUSTOM - this method allows you to specify custom gradients accumulator, this giving you better control of configuration params for training.<br>
         *
         * @param mode
         * @return
         */
        public Builder trainingMode(@NonNull TrainingMode mode) {
            this.trainingMode = mode;
            return this;
        }

        /**
         * This method allows you to specify GradientsAccumulator instance to be used in this ParallelWrapper instance
         *
         * PLEASE NOTE: This method is applicable only to gradients sharing mechanics. If parameters averaging is used, accumulator will be ignored
         *
         * @param accumulator
         * @return
         */
        public Builder gradientsAccumulator(@NonNull GradientsAccumulator accumulator) {
            this.accumulator = accumulator;
            return this;
        }


        /**
         * This method enables/disables averaged model score reporting
         *
         * @param reallyReport
         * @return
         */
        public Builder reportScoreAfterAveraging(boolean reallyReport) {
            this.reportScore = reallyReport;
            return this;
        }

        /**
         * This method returns ParallelWrapper instance
         *
         * @return
         */
        public ParallelWrapper build() {
            ParallelWrapper wrapper = new ParallelWrapper(model, workers, prefetchSize);
            wrapper.averagingFrequency = this.averagingFrequency;
            wrapper.reportScore = this.reportScore;
            wrapper.averageUpdaters = this.averageUpdaters;
            wrapper.legacyAveraging = this.legacyAveraging;
            wrapper.isMQ = this.isMQ;
            wrapper.workspaceMode = this.workspaceMode;


            switch (trainingMode) {
                case AVERAGING: {
                    this.trainerContext = new DefaultTrainerContext();
                    this.accumulator = null;
                    log.info("Creating new AveragingTraining instance");
                }
                    break;
                case SHARED_GRADIENTS: {
                    this.trainerContext = new SymmetricTrainerContext();
                    if (this.accumulator == null) {
                        log.info("Creating new GradientsAccumulator instance with threshold of [5e-4");
                        this.accumulator = new EncodedGradientsAccumulator(workers, 5e-4);
                    }
                }
                    break;
                case CUSTOM: {
                    this.trainerContext = new SymmetricTrainerContext();
                    if (this.accumulator == null)
                        throw new DL4JInvalidConfigException(
                                        "Please specify GradientsAccumulator fo encoded gradients mode");
                }
                    break;
                default:
                    throw new UnsupportedOperationException("Unknown trainingMode: [" + trainingMode + "]");
            }

            wrapper.trainerContext = this.trainerContext;
            wrapper.gradientsAccumulator = this.accumulator;

            wrapper.init();

            List<TrainingListener> modelListeners = null;
            if (model instanceof MultiLayerNetwork) {
                modelListeners = new ArrayList<>(((MultiLayerNetwork) model).getListeners());
                model.setListeners(Collections.emptyList());
            } else if (model instanceof ComputationGraph) {
                modelListeners = new ArrayList<>(((ComputationGraph) model).getListeners());
                model.setListeners(Collections.emptyList());
            }

            if (modelListeners != null && !modelListeners.isEmpty()) {
                wrapper.setListeners(modelListeners);
            }

            return wrapper;
        }
    }

    private static TrainingListener cloneListener(TrainingListener original) {
        if (original instanceof RoutingIterationListener) {
            return ((RoutingIterationListener) original).clone();
        }
        return original;
    }

    private void configureListeners(String workerUUID, Collection<TrainingListener> oldListeners,
                    Collection<TrainingListener> replicatedListeners) {
        for (TrainingListener listener : oldListeners) {
            TrainingListener l = cloneListener(listener);

            if (l instanceof RoutingIterationListener) {
                RoutingIterationListener rl = (RoutingIterationListener) l;
                //We're assuming session ID is set by the original RoutingIterationListener constructor, which means
                // it will be synced across all cloned instances
                rl.setSessionID(((RoutingIterationListener) listener).getSessionID());
                rl.setWorkerID(workerUUID);

                StatsStorageRouter currentRouter = ((RoutingIterationListener) listener).getStorageRouter();
                if (currentRouter != null) {
                    //User has set router on the listener/model, instead of via the
                    // setListeners(StatsStorageRouter, ...) method
                    rl.setStorageRouter(currentRouter);
                } else {
                    rl.setStorageRouter(ParallelWrapper.this.storageRouter);
                }

            }
            replicatedListeners.add(l);
        }
    }
}

