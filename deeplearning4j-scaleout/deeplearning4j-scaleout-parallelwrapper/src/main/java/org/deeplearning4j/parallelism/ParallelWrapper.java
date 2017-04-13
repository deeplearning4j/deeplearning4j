package org.deeplearning4j.parallelism;

import com.google.common.base.Preconditions;
import lombok.Data;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.api.storage.StatsStorageRouter;
import org.deeplearning4j.api.storage.listener.RoutingIterationListener;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.datasets.iterator.AsyncMultiDataSetIterator;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.updater.graph.ComputationGraphUpdater;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.parallelism.factory.DefaultTrainerContext;
import org.deeplearning4j.parallelism.factory.TrainerContext;
import org.deeplearning4j.parallelism.trainer.Trainer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
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
// TODO: We want this thing to be NUMA-aware in foreseable future
@Slf4j
@Data
public class ParallelWrapper implements AutoCloseable {
    protected Model model;
    protected int workers = 2;
    protected int prefetchSize = 2;
    protected int averagingFrequency = 1;
    protected Trainer[] zoo;
    private TrainerContext trainerContext = new DefaultTrainerContext();
    protected AtomicLong iterationsCounter = new AtomicLong(0);
    protected boolean reportScore = false;
    protected boolean averageUpdaters = true;
    protected boolean legacyAveraging = false;
    protected boolean wasAveraged = false;
    protected AtomicBoolean stopFit = new AtomicBoolean(false);
    protected List<IterationListener> listeners = new ArrayList<>();
    protected StatsStorageRouter storageRouter;
    protected boolean isMQ;
    protected WorkspaceMode workspaceMode;
    private Object[] trainerContextArgs;

    private MagicQueue mq;

    // log uncaught exceptions
    Thread.UncaughtExceptionHandler handler = new Thread.UncaughtExceptionHandler() {
        public void uncaughtException(Thread th, Throwable ex) {
            log.error("Uncaught exception: " + ex);
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

    @Override
    public void close() throws Exception {
        if (zoo != null) {
            for (int i = 0; i < zoo.length; i++) {
                if (zoo[i] != null)
                    zoo[i].shutdown();
            }
            zoo = null;
        }
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
        source.reset();

        MultiDataSetIterator iterator;
        if (prefetchSize > 0 && source.asyncSupported()) {
            iterator = new AsyncMultiDataSetIterator(source, prefetchSize);
        } else
            iterator = source;

        AtomicInteger locker = new AtomicInteger(0);

        while (iterator.hasNext() && !stopFit.get()) {
            MultiDataSet dataSet = iterator.next();

            if (dataSet == null)
                throw new ND4JIllegalStateException("You can't have NULL as MultiDataSet");

            /*
             now dataSet should be dispatched to next free workers, until all workers are busy. And then we should block till all finished.
            */
            int pos = locker.getAndIncrement();
            zoo[pos].feedMultiDataSet(dataSet);

            /*
                if all workers are dispatched now, join till all are finished
            */
            if (pos + 1 == workers || !iterator.hasNext()) {
                iterationsCounter.incrementAndGet();

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
                if (iterationsCounter.get() % averagingFrequency == 0 && pos + 1 == workers) {
                    double score = getScore(locker);

                    // averaging updaters state
                    if (model instanceof ComputationGraph) {
                        averageUpdatersState(locker, score);
                    } else
                        throw new RuntimeException("MultiDataSet must only be used with ComputationGraph model");

                    if (legacyAveraging && Nd4j.getAffinityManager().getNumberOfDevices() > 1) {
                        for (int cnt = 0; cnt < workers; cnt++) {
                            zoo[cnt].updateModel(model);
                        }
                    }
                }
                locker.set(0);
            }
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
        if (!legacyAveraging || Nd4j.getAffinityManager().getNumberOfDevices() == 1) {
            List<INDArray> params = new ArrayList<>();
            for (int cnt = 0; cnt < workers && cnt < locker.get(); cnt++) {
                params.add(zoo[cnt].getModel().params());
                score += zoo[cnt].getModel().score();
            }

            Nd4j.averageAndPropagate(model.params(), params);
        }
        else {
            INDArray params = Nd4j.zeros(model.params().shape());
            int cnt = 0;
            for (; cnt < workers && cnt < locker.get(); cnt++) {
                params.addi(zoo[cnt].getModel().params());
                score += zoo[cnt].getModel().score();
            }

            params.divi(cnt);
            model.setParams(params);
        }

        score /= Math.min(workers, locker.get());

        // TODO: improve this
        if (reportScore)
            log.info("Averaged score: " + score);
        return score;
    }

    private void averageUpdatersState(AtomicInteger locker, double score) {
        if (averageUpdaters) {
            ComputationGraphUpdater updater = ((ComputationGraph) model).getUpdater();
            int batchSize = 0;

            if (updater != null && updater.getStateViewArray() != null) {
                if (!legacyAveraging || Nd4j.getAffinityManager().getNumberOfDevices() == 1) {
                    List<INDArray> updaters = new ArrayList<>();
                    for (int cnt = 0; cnt < workers && cnt < locker.get(); cnt++) {
                        ComputationGraph workerModel = (ComputationGraph) zoo[cnt].getModel();
                        updaters.add(workerModel.getUpdater().getStateViewArray());
                        batchSize += workerModel.batchSize();
                    }
                    Nd4j.averageAndPropagate(updater.getStateViewArray(), updaters);
                } else {
                    INDArray state = Nd4j.zeros(updater.getStateViewArray().shape());
                    int cnt = 0;
                    for (; cnt < workers && cnt < locker.get(); cnt++) {
                        ComputationGraph workerModel = (ComputationGraph) zoo[cnt].getModel();
                        state.addi(workerModel.getUpdater().getStateViewArray());
                        batchSize += workerModel.batchSize();
                    }
                    state.divi(cnt);
                    updater.setStateViewArray(state);
                }
            }
        }

        ((ComputationGraph) model).setScore(score);
    }


    /**
     * This method allows you to specify IterationListeners for this model.
     * Note that for listeners like StatsListener (that have state that will be sent somewhere), consider instead
     * using {@link #setListeners(StatsStorageRouter, Collection)}
     *
     * @param listeners    Listeners to set
     */
    public void setListeners(@NonNull Collection<IterationListener> listeners) {
        setListeners(null, listeners);
    }

    /**
     * This method allows you to specify IterationListeners for this model.
     * Note that for listeners like StatsListener (that have state that will be sent somewhere), consider instead
     * using {@link #setListeners(StatsStorageRouter, Collection)}
     *
     * @param listeners    Listeners to set
     */
    public void setListeners(@NonNull IterationListener... listeners) {
        setListeners(Arrays.asList(listeners));
    }

    /**
     * Set the listeners, along with a StatsStorageRouter that the results will be shuffled to (in the case of any listeners
     * that implement the {@link RoutingIterationListener} interface)
     *
     * @param statsStorage Stats storage router to place the results into
     * @param listeners    Listeners to set
     */
    public void setListeners(StatsStorageRouter statsStorage, IterationListener... listeners) {
        setListeners(statsStorage, Arrays.asList(listeners));
    }

    /**
     * Set the listeners, along with a StatsStorageRouter that the results will be shuffled to (in the case of any listeners
     * that implement the {@link RoutingIterationListener} interface)
     *
     * @param statsStorage Stats storage router to place the results into
     * @param listeners    Listeners to set
     */
    public void setListeners(StatsStorageRouter statsStorage, Collection<? extends IterationListener> listeners) {
        //Check if we have any RoutingIterationListener instances that need a StatsStorage implementation...
        if (listeners != null) {
            for (IterationListener l : listeners) {
                if (l instanceof RoutingIterationListener) {
                    RoutingIterationListener rl = (RoutingIterationListener) l;
                    if (statsStorage == null && rl.getStorageRouter() == null) {
                        log.warn("RoutingIterationListener provided without providing any StatsStorage instance. Iterator may not function without one. Listener: {}",
                                l);
                    } else if (rl.getStorageRouter() != null && !(rl.getStorageRouter() instanceof Serializable)) {
                        //Spark would throw a (probably cryptic) serialization exception later anyway...
                        throw new IllegalStateException(
                                "RoutingIterationListener provided with non-serializable storage router "
                                        + "\nRoutingIterationListener class: " + rl.getClass().getName()
                                        + "\nStatsStorageRouter class: "
                                        + rl.getStorageRouter().getClass().getName());
                    }
                }
            }

            this.listeners.addAll(listeners);
        }
        else {
            this.listeners.clear();
        }

        this.storageRouter = statsStorage;
    }


    /**
     * This method takes DataSetIterator, and starts training over it by scheduling DataSets to different executors
     *
     * @param source
     */
    public synchronized void fit(@NonNull DataSetIterator source) {
        stopFit.set(false);
        createZooIfNeccessary(false);
        source.reset();

        DataSetIterator iterator;
        if (prefetchSize > 0 && source.asyncSupported()) {
            if (isMQ) {
                if (workers % Nd4j.getAffinityManager().getNumberOfDevices() != 0)
                    log.warn("Number of workers [{}] isn't optimal for available devices [{}]", workers,
                            Nd4j.getAffinityManager().getNumberOfDevices());

                if (mq == null)
                    mq = new MagicQueue.Builder().setCapacityPerFlow(prefetchSize).setMode(MagicQueue.Mode.SEQUENTIAL)
                        .setNumberOfBuckets(Nd4j.getAffinityManager().getNumberOfDevices()).build();

                iterator = new AsyncDataSetIterator(source, prefetchSize * workers, mq);

            } else
                iterator = new AsyncDataSetIterator(source, prefetchSize * workers);
        } else
            iterator = source;

        AtomicInteger locker = new AtomicInteger(0);
        while (iterator.hasNext() && !stopFit.get()) {
            DataSet dataSet = iterator.next();

            if (dataSet == null)
                throw new ND4JIllegalStateException("You can't have NULL as DataSet");

            /*
             now dataSet should be dispatched to next free workers, until all workers are busy. And then we should block till all finished.
            */
            int pos = locker.getAndIncrement();
            if (zoo == null)
                throw new IllegalStateException(
                        "ParallelWrapper.shutdown() has been called too early and will fail from this point forward.");
            zoo[pos].feedDataSet(dataSet);

            /*
                if all workers are dispatched now, join till all are finished
            */
            if (pos + 1 == workers || !iterator.hasNext()) {
                iterationsCounter.incrementAndGet();

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
                if (iterationsCounter.get() % averagingFrequency == 0 && pos + 1 == workers) {
                    double score = getScore(locker);

                    // averaging updaters state
                    if (model instanceof MultiLayerNetwork) {
                        if (averageUpdaters) {
                            Updater updater = ((MultiLayerNetwork) model).getUpdater();
                            int batchSize = 0;

                            if (updater != null && updater.getStateViewArray() != null) {
                                if (!legacyAveraging || Nd4j.getAffinityManager().getNumberOfDevices() == 1) {
                                    List<INDArray> updaters = new ArrayList<>();
                                    for (int cnt = 0; cnt < workers && cnt < locker.get(); cnt++) {
                                        MultiLayerNetwork workerModel = (MultiLayerNetwork) zoo[cnt].getModel();
                                        updaters.add(workerModel.getUpdater().getStateViewArray());
                                        batchSize += workerModel.batchSize();
                                    }

                                    Nd4j.averageAndPropagate(updater.getStateViewArray(), updaters);
                                }
                                else {
                                    INDArray state = Nd4j.zeros(updater.getStateViewArray().shape());
                                    int cnt = 0;
                                    for (; cnt < workers && cnt < locker.get(); cnt++) {
                                        MultiLayerNetwork workerModel = (MultiLayerNetwork) zoo[cnt].getModel();
                                        state.addi(workerModel.getUpdater().getStateViewArray().dup());
                                        batchSize += workerModel.batchSize();
                                    }
                                    state.divi(cnt);
                                    updater.setStateViewArray((MultiLayerNetwork) model, state, false);
                                }
                            }
                        }

                        ((MultiLayerNetwork) model).setScore(score);
                    } else if (model instanceof ComputationGraph) {
                        averageUpdatersState(locker, score);
                    }

                    if (legacyAveraging && Nd4j.getAffinityManager().getNumberOfDevices() > 1) {
                        for (int cnt = 0; cnt < workers; cnt++) {
                            zoo[cnt].updateModel(model);
                        }
                    }
                }
                locker.set(0);
            }
        }

        // sanity checks, or the dataset may never average
        if (!wasAveraged)
            log.warn("Parameters were never averaged on current fit(). Ratios of batch size, num workers, and averaging frequency may be responsible.");
        //            throw new IllegalStateException("Parameters were never averaged. Please check batch size ratios, number of workers, and your averaging frequency.");

        log.debug("Iterations passed: {}", iterationsCounter.get());
    }


    private void createZooIfNeccessary(boolean useMDS) {
        if (zoo == null) {
            trainerContext.init(model,trainerContextArgs);
            zoo = new Trainer[workers];
            for (int cnt = 0; cnt < workers; cnt++) {
                // we pass true here, to tell Trainer to use MultiDataSet queue for training
                zoo[cnt] = trainerContext.create(cnt,
                        model,
                        Nd4j.getAffinityManager().getDeviceForCurrentThread(),
                        useMDS,
                        this, workspaceMode);
                zoo[cnt].setUncaughtExceptionHandler(handler);
                zoo[cnt].start();
            }
        }
    }

    public static class Builder<T extends Model> {
        protected T model;
        protected int workers = Nd4j.getAffinityManager().getNumberOfDevices();
        protected int prefetchSize = 16;
        protected int averagingFrequency = 1;
        protected boolean reportScore = false;
        protected boolean averageUpdaters = true;
        protected boolean legacyAveraging = true;
        protected boolean isMQ = false; // Nd4j.getAffinityManager().getNumberOfDevices() > 1;
        protected TrainerContext trainerContext = new DefaultTrainerContext();
        protected Object[] trainerContextArgs;
        protected WorkspaceMode workspaceMode = WorkspaceMode.SEPARATE;

        /**
         * Transer context args are for calling a
         * {@link TrainerContext} init method
         * when {@link ParallelWrapper} starts training
         * @param trainerContextArgs the args to use (maybe null)
         * @return
         */
        public Builder trainerContextArgs(Object...trainerContextArgs) {
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
        public Builder trainerFactory(TrainerContext trainerContext) {
            Preconditions.checkNotNull(trainerContext);
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
            this.averagingFrequency = freq;
            return this;
        }

        /**
         * This method enables/disables updaters averaging.
         *
         * Default value: TRUE
         *
         * PLEASE NOTE: This method is suitable for debugging purposes mostly. So don't change default value, unless you're sure why you need it.
         *
         * @param reallyAverage
         * @return
         */
        public Builder averageUpdaters(boolean reallyAverage) {
            this.averageUpdaters = reallyAverage;
            return this;
        }

        /**
         * This method enables/disable MagicQueue use
         * If set to true, all datasets will be spread among all available devices at prefetch phase using AsyncDataSetIterator
         *
         * PLEASE NOTE: This is experimental feature.
         *
         * Default: false
         * @param reallyUse
         * @return
         */
        public Builder useMQ(boolean reallyUse) {
            this.isMQ = reallyUse;
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
         * If set to true, legacy averaging method is used. This might be used as fallback on multi-gpu systems without P2P access available.
         *
         * Default value: false
         *
         * @param reallyUse
         * @return
         */
        public Builder useLegacyAveraging(boolean reallyUse) {
            this.legacyAveraging = reallyUse;
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

            return wrapper;
        }
    }

    private static IterationListener cloneListener(IterationListener original){
        if(original instanceof RoutingIterationListener){
            return ((RoutingIterationListener) original).clone();
        }
        return original;
    }

    private void configureListeners(String workerUUID, Collection<IterationListener> oldListeners,
                                    Collection<IterationListener> replicatedListeners){
        for (IterationListener listener : oldListeners) {
            IterationListener l = cloneListener(listener);

            if (l instanceof RoutingIterationListener) {
                RoutingIterationListener rl = (RoutingIterationListener)l;
                //We're assuming session ID is set by the original RoutingIterationListener constructor, which means
                // it will be synced across all cloned instances
                rl.setSessionID(((RoutingIterationListener) listener).getSessionID());
                rl.setWorkerID(workerUUID);

                StatsStorageRouter currentRouter = ((RoutingIterationListener)listener).getStorageRouter();
                if(currentRouter != null){
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

