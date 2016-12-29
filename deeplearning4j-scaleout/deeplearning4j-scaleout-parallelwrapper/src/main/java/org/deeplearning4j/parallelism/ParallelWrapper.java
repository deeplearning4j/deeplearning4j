package org.deeplearning4j.parallelism;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.api.storage.Persistable;
import org.deeplearning4j.api.storage.StatsStorageRouter;
import org.deeplearning4j.api.storage.StatsStorageRouterProvider;
import org.deeplearning4j.api.storage.StorageMetaData;
import org.deeplearning4j.api.storage.listener.RoutingIterationListener;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.datasets.iterator.AsyncMultiDataSetIterator;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.updater.graph.ComputationGraphUpdater;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;


import java.io.Serializable;
import java.util.*;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * This is simple data-parallel wrapper
 * suitable for multi-cpu/multi-gpu environments.
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class ParallelWrapper implements AutoCloseable {
    private Model model;
    private int workers = 2;
    private int prefetchSize = 2;
    private int averagingFrequency = 1;
    private Trainer zoo[];
    private AtomicLong iterationsCounter = new AtomicLong(0);
    private boolean reportScore = false;
    private boolean averageUpdaters = true;
    private boolean legacyAveraging = false;
    private boolean wasAveraged = false;
    private AtomicBoolean stopFit = new AtomicBoolean(false);
    private List<IterationListener> listeners = new ArrayList<>();
    private StatsStorageRouter storageRouter;
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

        /*
        zoo = new Trainer[workers];
        for (int cnt = 0; cnt < workers; cnt++) {
            zoo[cnt] = new Trainer(cnt, model);
            zoo[cnt].setUncaughtExceptionHandler(handler);
            zoo[cnt].start();
        }
        */
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
    public void stopFit() { stopFit.set(true); }

    /**
     *
     * @param source
     */
    public synchronized void fit(@NonNull MultiDataSetIterator source) {
        stopFit.set(false);
        if (zoo == null) {
            zoo = new Trainer[workers];
            for (int cnt = 0; cnt < workers; cnt++) {
                // we pass true here, to tell Trainer to use MultiDataSet queue for training
                zoo[cnt] = new Trainer(cnt, model, true);
                zoo[cnt].setUncaughtExceptionHandler(handler);
                zoo[cnt].start();
            }
        } else {
            for (int cnt = 0; cnt < workers; cnt++) {
                zoo[cnt].useMDS = true;
            }
        }
        source.reset();

        MultiDataSetIterator iterator;
        if (prefetchSize > 0 && source.asyncSupported()) {
            iterator = new AsyncMultiDataSetIterator(source, prefetchSize);
        } else iterator = source;

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

                for (int cnt = 0; cnt < workers && cnt < locker.get(); cnt ++) {
                    try {
                        zoo[cnt].waitTillRunning();
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                }

                /*
                    average model, and propagate it to whole
                */
                if (iterationsCounter.get() % averagingFrequency == 0 && pos + 1 == workers) {
                    wasAveraged = true;
                    double score = 0.0;
                    if (!legacyAveraging || Nd4j.getAffinityManager().getNumberOfDevices() == 1) {
                        List<INDArray> params = new ArrayList<>();
                        for (int cnt = 0; cnt < workers && cnt < locker.get(); cnt++) {
                            params.add(zoo[cnt].getModel().params());
                            score += zoo[cnt].getModel().score();
                        }
                        Nd4j.averageAndPropagate(model.params(), params);
                    } else {
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

                    // averaging updaters state
                    if (model instanceof ComputationGraph) {
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
                    } else throw new RuntimeException("MultiDataSet must only be used with ComputationGraph model");

                    if (legacyAveraging &&  Nd4j.getAffinityManager().getNumberOfDevices() > 1) {
                        for (int cnt = 0; cnt < workers; cnt++) {
                            zoo[cnt].updateModel(model);
                        }
                    }
                }
                locker.set(0);
            }
        }

        // sanity checks, or the dataset may never average
        if(!wasAveraged)
            log.warn("Parameters were never averaged on current fit(). Ratios of batch size, num workers, and averaging frequency may be responsible.");
//            throw new IllegalStateException("Parameters were never averaged. Please check batch size ratios, number of workers, and your averaging frequency.");

        log.debug("Iterations passed: {}", iterationsCounter.get());
//        iterationsCounter.set(0);
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
    public void setListeners(@NonNull IterationListener... listeners){
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
        StatsStorageRouterProvider routerProvider = null;
        if(listeners != null) {
            for(IterationListener l : listeners) {
                if(l instanceof RoutingIterationListener) {
                    RoutingIterationListener rl = (RoutingIterationListener) l;
                    if(rl.getStorageRouter() == null) {
                        log.warn("RoutingIterationListener provided without providing any StatsStorage instance. Iterator may not function without one. Listener: {}", l);
                    } else if(!(rl.getStorageRouter() instanceof Serializable)) {
                        //Spark would throw a (probably cryptic) serialization exception later anyway...
                        throw new IllegalStateException("RoutingIterationListener provided with non-serializable storage router");
                    }

                }
            }
        }


        this.storageRouter = statsStorage;
        this.listeners.addAll(listeners);

    }


    /**
     * This method takes DataSetIterator, and starts training over it by scheduling DataSets to different executors
     *
     * @param source
     */
    public synchronized void fit(@NonNull DataSetIterator source) {
        stopFit.set(false);
        if (zoo == null) {
            zoo = new Trainer[workers];
            for (int cnt = 0; cnt < workers; cnt++) {
                zoo[cnt] = new Trainer(cnt, model);
                zoo[cnt].setUncaughtExceptionHandler(handler);
                zoo[cnt].start();
            }
        }
        source.reset();

        DataSetIterator iterator;
        if (prefetchSize > 0 && source.asyncSupported()) {
            iterator = new AsyncDataSetIterator(source, prefetchSize);
        } else iterator = source;

        AtomicInteger locker = new AtomicInteger(0);
        int whiles = 0;
        while (iterator.hasNext() && !stopFit.get()) {
            whiles++;
            DataSet dataSet = iterator.next();

            if (dataSet == null)
                throw new ND4JIllegalStateException("You can't have NULL as DataSet");

            /*
             now dataSet should be dispatched to next free workers, until all workers are busy. And then we should block till all finished.
            */
            int pos = locker.getAndIncrement();
            if(zoo == null) throw new IllegalStateException("ParallelWrapper.shutdown() has been called too early and will fail from this point forward.");
            zoo[pos].feedDataSet(dataSet);

            /*
                if all workers are dispatched now, join till all are finished
            */
            if (pos + 1 == workers || !iterator.hasNext()) {
                iterationsCounter.incrementAndGet();

                for (int cnt = 0; cnt < workers && cnt < locker.get(); cnt ++) {
                    try {
                        zoo[cnt].waitTillRunning();
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                }

                /*
                    average model, and propagate it to whole
                */
                if (iterationsCounter.get() % averagingFrequency == 0 && pos + 1 == workers) {
                    wasAveraged = true;
                    double score = 0.0;
                    if (!legacyAveraging || Nd4j.getAffinityManager().getNumberOfDevices() == 1) {
                        List<INDArray> params = new ArrayList<>();
                        for (int cnt = 0; cnt < workers && cnt < locker.get(); cnt++) {
                            params.add(zoo[cnt].getModel().params());
                            score += zoo[cnt].getModel().score();
                        }
                        Nd4j.averageAndPropagate(model.params(), params);
                    } else {
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
                                } else {
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

                    if (legacyAveraging &&  Nd4j.getAffinityManager().getNumberOfDevices() > 1) {
                        for (int cnt = 0; cnt < workers; cnt++) {
                            zoo[cnt].updateModel(model);
                        }
                    }
                }
                locker.set(0);
            }
        }

        // sanity checks, or the dataset may never average
        if(!wasAveraged)
            log.warn("Parameters were never averaged on current fit(). Ratios of batch size, num workers, and averaging frequency may be responsible.");
//            throw new IllegalStateException("Parameters were never averaged. Please check batch size ratios, number of workers, and your averaging frequency.");

        log.debug("Iterations passed: {}", iterationsCounter.get());
//        iterationsCounter.set(0);
    }

    public static class Builder<T extends Model> {
        private T model;
        private int workers = 2;
        private int prefetchSize = 16;
        private int averagingFrequency = 1;
        private boolean reportScore = false;
        private boolean averageUpdaters = true;
        private boolean legacyAveraging = true;

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

            return wrapper;
        }
    }

    private  class Trainer extends Thread implements Runnable {
        private Model originalModel;
        private Model replicatedModel;
        private LinkedBlockingQueue<DataSet> queue = new LinkedBlockingQueue<>();
        private LinkedBlockingQueue<MultiDataSet> queueMDS = new LinkedBlockingQueue<>();
        private AtomicInteger running = new AtomicInteger(0);
        private int threadId;
        private AtomicBoolean shouldUpdate = new AtomicBoolean(false);
        private AtomicBoolean shouldStop = new AtomicBoolean(false);
        private Exception thrownException;
        private volatile boolean useMDS = false;
        private final String uuid = UUID.randomUUID().toString();


        public Trainer(int threadId, Model model, boolean useMDS) {
            this(threadId, model);
            this.useMDS = useMDS;
        }

        public Trainer(int threadId, Model model) {
            this.threadId = threadId;
            this.setDaemon(true);
            this.setName("ParallelWrapper trainer " + threadId);

            this.originalModel = model;
            if (model instanceof MultiLayerNetwork) {
                this.replicatedModel = ((MultiLayerNetwork) model).clone();

            } else if (model instanceof ComputationGraph) {
                this.replicatedModel = ((ComputationGraph) model).clone();
            }
        }

        public void feedMultiDataSet(@NonNull MultiDataSet dataSet) {
            running.incrementAndGet();
            queueMDS.add(dataSet);
        }

        public void feedDataSet(@NonNull DataSet dataSet) {
            running.incrementAndGet();
            queue.add(dataSet);
        }

        public Model getModel() {
            return replicatedModel;
        }

        public void updateModel(@NonNull Model model) {

            this.shouldUpdate.set(true);

            if (replicatedModel instanceof MultiLayerNetwork) {
                replicatedModel.setParams(model.params().dup());

                Updater updater = ((MultiLayerNetwork) model).getUpdater();
                INDArray view = updater.getStateViewArray();

                if (view != null) {
                    updater = ((MultiLayerNetwork) replicatedModel).getUpdater();
                    INDArray viewD = view.dup();

                    if (Nd4j.getExecutioner() instanceof GridExecutioner)
                        ((GridExecutioner) Nd4j.getExecutioner()).flushQueueBlocking();

                    updater.setStateViewArray((MultiLayerNetwork) replicatedModel, viewD, false);
                }
            } else if (replicatedModel instanceof  ComputationGraph) {
                replicatedModel.setParams(model.params().dup());

                ComputationGraphUpdater updater = ((ComputationGraph) model).getUpdater();
                INDArray view = updater.getStateViewArray();

                if (view != null) {
                    INDArray viewD = view.dup();

                    if (Nd4j.getExecutioner() instanceof GridExecutioner)
                        ((GridExecutioner) Nd4j.getExecutioner()).flushQueueBlocking();

                    updater = ((ComputationGraph) replicatedModel).getUpdater();
                    updater.setStateViewArray(viewD);
                }
            }

            if (Nd4j.getExecutioner() instanceof GridExecutioner)
                ((GridExecutioner) Nd4j.getExecutioner()).flushQueueBlocking();
        }

        public boolean isRunning(){
            // if Trainer thread got exception during training - rethrow it here
            if (thrownException != null)
                throw new RuntimeException(thrownException);

            return running.get() == 0;
        }

        public void shutdown() {
            shouldStop.set(true);
        }

        @Override
        public void run() {
            try {
                // we create fresh network, with the same configuration, as initially created by user
                // however, we don't need clone or anything here
                if (originalModel instanceof MultiLayerNetwork) {
                    MultiLayerConfiguration conf = ((MultiLayerNetwork) originalModel).getLayerWiseConfigurations().clone();
                    this.replicatedModel = new MultiLayerNetwork(conf);

                    ((MultiLayerNetwork) replicatedModel).init();
                    Collection<IterationListener> oldListeners = ((MultiLayerNetwork) originalModel).getListeners();
                    Collection<IterationListener> replicatedListeners = new ArrayList<>();

                    for(IterationListener listener : oldListeners) {
                        if(listener instanceof RoutingIterationListener) {
                            RoutingIterationListener routingListener = ((RoutingIterationListener) listener).clone();
                            routingListener.setSessionID(((RoutingIterationListener) listener).getSessionID());
                            routingListener.setWorkerID(uuid);
                            routingListener.setStorageRouter(ParallelWrapper.this.storageRouter);
                            replicatedListeners.add(routingListener);
                        } else {
                            replicatedListeners.add(listener);
                        }
                    }

                    ((MultiLayerNetwork)this.replicatedModel).setListeners(replicatedListeners);
                } else if (originalModel instanceof ComputationGraph) {
                    this.replicatedModel = new ComputationGraph(((ComputationGraph) originalModel).getConfiguration().clone());

                    ((ComputationGraph) this.replicatedModel).init();
                    Collection<IterationListener> oldListeners = ((ComputationGraph) originalModel).getListeners();
                    Collection<IterationListener> replicatedListeners = new ArrayList<>();

                    for(IterationListener listener : oldListeners) {
                        if(listener instanceof RoutingIterationListener) {
                            RoutingIterationListener routingIterationListener = ((RoutingIterationListener) listener).clone();
                            routingIterationListener.setSessionID(((RoutingIterationListener) listener).getSessionID());
                            routingIterationListener.setWorkerID(uuid);
                            routingIterationListener.setStorageRouter(ParallelWrapper.this.storageRouter);
                            replicatedListeners.add(routingIterationListener);
                        } else {
                            replicatedListeners.add(listener);
                        }
                    }

                    ((ComputationGraph)this.replicatedModel).setListeners(replicatedListeners);
                }

                if (!useMDS) {
                    while (!shouldStop.get()) {
                        DataSet dataSet = queue.poll(100, TimeUnit.MILLISECONDS);
                        if (dataSet != null) {
                            if (replicatedModel instanceof MultiLayerNetwork) {
                                ((MultiLayerNetwork) replicatedModel).fit(dataSet);
                            } else if (replicatedModel instanceof ComputationGraph) {
                                ((ComputationGraph) replicatedModel).fit(dataSet);
                            }

                            if (Nd4j.getExecutioner() instanceof GridExecutioner)
                                ((GridExecutioner) Nd4j.getExecutioner()).flushQueueBlocking();

                            running.decrementAndGet();
                        }
                    }
                } else {
                    // loop for MultiDataSet
                    while (!shouldStop.get()) {
                        MultiDataSet dataSet = queueMDS.poll(100, TimeUnit.MILLISECONDS);
                        if (dataSet != null) {
                            if (replicatedModel instanceof ComputationGraph) {
                                ((ComputationGraph) replicatedModel).fit(dataSet);
                            } else throw new RuntimeException("MultiDataSet can be fit into ComputationGraph only");

                            if (Nd4j.getExecutioner() instanceof GridExecutioner)
                                ((GridExecutioner) Nd4j.getExecutioner()).flushQueueBlocking();

                            running.decrementAndGet();
                        }
                    }
                }
            } catch (Exception e) {
                this.thrownException = e;
            }
        }

        public void waitTillRunning() {
            while (running.get() != 0) {

                // if Trainer thread got exception during training - rethrow it here
                if (thrownException != null)
                    throw new RuntimeException(thrownException);

                try {
                    Thread.sleep(10);
                } catch (Exception e) {
                    ;
                }
            }
        }
    }
}

