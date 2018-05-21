package org.deeplearning4j.parallelism.trainer;

import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.api.storage.StatsStorageRouter;
import org.deeplearning4j.api.storage.listener.RoutingIterationListener;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.updater.graph.ComputationGraphUpdater;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Collection;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.LockSupport;

/**
 * Trains datasets using a standard in memory
 * parameter averaging technique.
 * Think of this worker as the simplest form of doing parameter averaging
 *
 * @author Adam Gibson
 */
@Builder
@Slf4j
@NoArgsConstructor
@AllArgsConstructor
public class DefaultTrainer extends Thread implements Trainer {

    protected Model replicatedModel;

    // TODO: make queue size configurable
    @Builder.Default
    protected LinkedBlockingQueue<DataSet> queue = new LinkedBlockingQueue<>(1);
    @Builder.Default
    protected LinkedBlockingQueue<MultiDataSet> queueMDS = new LinkedBlockingQueue<>(1);
    @Builder.Default
    protected AtomicInteger running = new AtomicInteger(0);
    @Builder.Default
    protected AtomicBoolean shouldUpdate = new AtomicBoolean(false);
    @Builder.Default
    protected AtomicBoolean shouldStop = new AtomicBoolean(false);
    protected Exception thrownException;
    @Builder.Default
    protected volatile boolean useMDS = false;
    @Getter protected String uuid;
    @Builder.Default
    protected boolean onRootModel = false;
    @Builder.Default
    protected volatile AtomicLong lastEtlTime = new AtomicLong(0);

    @Builder.Default
    protected AtomicBoolean nullMode = new AtomicBoolean(false);
    protected DataSet nullDataSet;

    @Builder.Default
    protected AtomicBoolean isStopped = new AtomicBoolean(false);

    protected ParallelWrapper parallelWrapper;
    protected WorkspaceMode workspaceMode;
    protected int averagingFrequency;
    protected int threadId;
    protected Model originalModel;


    @Override
    public void feedMultiDataSet(@NonNull MultiDataSet dataSet, long etlTime) {
        setupIfNeccessary();
        try {
            queueMDS.put(dataSet);
            running.incrementAndGet();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            // do nothing
        }

        if (lastEtlTime == null)
            lastEtlTime = new AtomicLong(0);

        lastEtlTime.set(etlTime);
    }

    @Override
    public void feedDataSet(DataSet dataSet, long etlTime) {
        setupIfNeccessary();
        if (dataSet != null) {
            try {
                queue.put(dataSet);
                running.incrementAndGet();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                // do nothing
            }
        } else {
            if (nullMode == null)
                nullMode = new AtomicBoolean(false);

            nullMode.set(true);
        }

        if (lastEtlTime == null)
            lastEtlTime = new AtomicLong(0);

        lastEtlTime.set(etlTime);
    }

    @Override
    public Model getModel() {
        return replicatedModel;
    }

    @Override
    public void updateModel(@NonNull Model model) {
        this.shouldUpdate.set(true);
        if (replicatedModel instanceof MultiLayerNetwork) {
            replicatedModel.setParams(model.params().dup());

            Updater updater = ((MultiLayerNetwork) model).getUpdater();
            INDArray view = updater.getStateViewArray();

            if (view != null) {
                updater = ((MultiLayerNetwork) replicatedModel).getUpdater();
                INDArray viewD = view.dup();

                Nd4j.getExecutioner().commit();

                updater.setStateViewArray((MultiLayerNetwork) replicatedModel, viewD, false);
            }
        } else if (replicatedModel instanceof ComputationGraph) {
            replicatedModel.setParams(model.params().dup());

            ComputationGraphUpdater updater = ((ComputationGraph) model).getUpdater();
            INDArray view = updater.getStateViewArray();

            if (view != null) {
                INDArray viewD = view.dup();

                Nd4j.getExecutioner().commit();

                updater = ((ComputationGraph) replicatedModel).getUpdater();
                updater.setStateViewArray(viewD);
            }
        }

        Nd4j.getExecutioner().commit();
    }



    protected void setupIfNeccessary() {
        if (queue == null)
            queue = new LinkedBlockingQueue<>(1);
        if (queueMDS == null)
            queueMDS = new LinkedBlockingQueue<>(1);
        if (running == null)
            running = new AtomicInteger(0);
        if (shouldStop == null)
            shouldStop = new AtomicBoolean(false);
        if (shouldUpdate == null)
            shouldUpdate = new AtomicBoolean(false);
        if (isStopped == null)
            isStopped = new AtomicBoolean(false);
        if (lastEtlTime == null)
            lastEtlTime = new AtomicLong(0);
    }

    @Override
    public boolean isRunning() {
        // if Trainer thread got exception during training - rethrow it here
        if (thrownException != null)
            throw new RuntimeException(thrownException);

        return running.get() == 0;
    }

    @Override
    public void shutdown() {
        shouldStop.set(true);
        while (!isStopped.get())
            LockSupport.parkNanos(1000L);

        shouldStop.set(false);
        isStopped.set(false);
    }

    protected void fit(DataSet dataSet) {
        if (replicatedModel instanceof MultiLayerNetwork) {
            if (lastEtlTime == null)
                lastEtlTime = new AtomicLong(0);

            ((MultiLayerNetwork) replicatedModel).setLastEtlTime(lastEtlTime.get());
            ((MultiLayerNetwork) replicatedModel).fit(dataSet);
        } else if (replicatedModel instanceof ComputationGraph) {
            if (lastEtlTime == null)
                lastEtlTime = new AtomicLong(0);

            ((ComputationGraph) replicatedModel).setLastEtlTime(lastEtlTime.get());
            ((ComputationGraph) replicatedModel).fit(dataSet);
        }
    }

    protected void fit(MultiDataSet dataSet) {
        if (lastEtlTime == null)
            lastEtlTime = new AtomicLong(0);

        ((ComputationGraph) replicatedModel).setLastEtlTime(lastEtlTime.get());
        ((ComputationGraph) replicatedModel).fit(dataSet);
    }

    /**
     * This method does post-initialization configuration of Model.
     * Good place to configure listeners and all such a things
     */
    protected void postInit() {
        Collection<TrainingListener> oldListeners = new ArrayList<>();
        Collection<TrainingListener> replicatedListeners = new ArrayList<>();

        if (parallelWrapper.getListeners() != null) {
            oldListeners.addAll(parallelWrapper.getListeners());
        }
        configureListeners(uuid, oldListeners, replicatedListeners);

        this.replicatedModel.setListeners(replicatedListeners);
    }

    @Override
    public void run() {
        setupIfNeccessary();
        AtomicInteger iterationsCounter = new AtomicInteger(0);

        // FIXME: make this thing CUDA-compatible, and avoid RC at originalModel relocation
        if (threadId == 0)
            onRootModel = true;

        try {
            // we create fresh network, with the same configuration, as initially created by user
            // however, we don't need clone or anything here
            if (originalModel instanceof MultiLayerNetwork) {
                if (!onRootModel) {
                    MultiLayerConfiguration conf = MultiLayerConfiguration.fromJson(
                                    ((MultiLayerNetwork) originalModel).getLayerWiseConfigurations().toJson());
                    conf.setTrainingWorkspaceMode(workspaceMode);
                    this.replicatedModel = new MultiLayerNetwork(conf);

                    replicatedModel.init();

                    // we replicate original model params & updater state, just in case it's pre-trained model
                    synchronized (originalModel) {
                        replicatedModel.setParams(originalModel.params().unsafeDuplication(true));

                        Updater updaterReplica = ((MultiLayerNetwork) replicatedModel).getUpdater();
                        Updater updaterOrigina = ((MultiLayerNetwork) originalModel).getUpdater();

                        if (updaterOrigina != null && updaterOrigina.getStateViewArray() != null)
                            updaterReplica.setStateViewArray((MultiLayerNetwork) replicatedModel,
                                            updaterOrigina.getStateViewArray().unsafeDuplication(true), false);

                        Nd4j.getExecutioner().commit();
                    }
                } else {
                    this.replicatedModel = originalModel;
                    if (!((MultiLayerNetwork) replicatedModel).isInitCalled())
                        this.replicatedModel.init();

                    ((MultiLayerNetwork) replicatedModel).getLayerWiseConfigurations()
                                    .setTrainingWorkspaceMode(workspaceMode);
                }
            } else if (originalModel instanceof ComputationGraph) {
                if (!onRootModel) {
                    ComputationGraphConfiguration conf = ComputationGraphConfiguration
                                    .fromJson(((ComputationGraph) originalModel).getConfiguration().toJson());
                    conf.setTrainingWorkspaceMode(workspaceMode);

                    this.replicatedModel = new ComputationGraph(conf);
                    this.replicatedModel.init();

                    // we replicate original model params & updater state, just in case it's pre-trained model
                    synchronized (originalModel) {
                        replicatedModel.setParams(originalModel.params().unsafeDuplication(true));

                        ComputationGraphUpdater updaterReplica = ((ComputationGraph) replicatedModel).getUpdater();
                        ComputationGraphUpdater updaterOrigina = ((ComputationGraph) originalModel).getUpdater();

                        if (updaterOrigina != null && updaterOrigina.getStateViewArray() != null)
                            updaterReplica.setStateViewArray(
                                            updaterOrigina.getStateViewArray().unsafeDuplication(true));

                        Nd4j.getExecutioner().commit();
                    }
                } else {
                    this.replicatedModel = originalModel;
                    this.replicatedModel.init();
                    ((ComputationGraph) replicatedModel).getConfiguration().setTrainingWorkspaceMode(workspaceMode);
                }
            }

            if (replicatedModel == null)
                log.error("replicatedModel is NULL at worker_{}", threadId);

            // classes that extend DefaultTrainer might hook something there
            postInit();

            if (!useMDS) {
                while (!shouldStop.get()) {
                    DataSet dataSet = null;
                    if (nullMode == null || !nullMode.get())
                        dataSet = queue.poll(10, TimeUnit.MILLISECONDS);
                    else {
                        // this code branch is for debugging only, please ignore :)
                        if (nullDataSet == null)
                            nullDataSet = new org.nd4j.linalg.dataset.DataSet(Nd4j.create(64, 28 * 28),
                                            Nd4j.create(64, 10));

                        dataSet = nullDataSet;
                    }
                    if (dataSet != null) {

                        fit(dataSet);

                        // if we don't support cross-device stuff (like multi-gpu on windows) - sync back to host
                        if (!Nd4j.getAffinityManager().isCrossDeviceAccessSupported() && (averagingFrequency == 0
                                        || iterationsCounter.incrementAndGet() % averagingFrequency == 0)
                                        && averagingRequired()) {
                            // we ensure all operations are finished in this training round
                            Nd4j.getExecutioner().commit();

                            // we ensure memory is updated on host side
                            Nd4j.getAffinityManager().ensureLocation(replicatedModel.params(),
                                            AffinityManager.Location.HOST);

                            if (replicatedModel instanceof MultiLayerNetwork) {
                                Updater updaterReplica = ((MultiLayerNetwork) replicatedModel).getUpdater();
                                if (updaterReplica.getStateViewArray() != null)
                                    Nd4j.getAffinityManager().ensureLocation(updaterReplica.getStateViewArray(),
                                                    AffinityManager.Location.HOST);
                            } else {
                                ComputationGraphUpdater updaterReplica =
                                                ((ComputationGraph) replicatedModel).getUpdater();

                                if (updaterReplica.getStateViewArray() != null)
                                    Nd4j.getAffinityManager().ensureLocation(updaterReplica.getStateViewArray(),
                                                    AffinityManager.Location.HOST);
                            }
                        }

                        running.decrementAndGet();
                    }
                }
            } else {
                // loop for MultiDataSet
                while (!shouldStop.get()) {
                    MultiDataSet dataSet = queueMDS.poll(10, TimeUnit.MILLISECONDS);
                    if (dataSet != null) {

                        // just fitting
                        fit(dataSet);

                        // if we don't support cross-device stuff (like multi-gpu on windows) - sync back to host
                        if (!Nd4j.getAffinityManager().isCrossDeviceAccessSupported() && (averagingFrequency == 0
                                        || iterationsCounter.incrementAndGet() % averagingFrequency == 0)
                                        && averagingRequired()) {
                            // we ensure all operations are finished in this training round
                            Nd4j.getExecutioner().commit();

                            // we ensure memory is updated on host side
                            Nd4j.getAffinityManager().ensureLocation(replicatedModel.params(),
                                            AffinityManager.Location.HOST);

                            ComputationGraphUpdater updaterReplica = ((ComputationGraph) replicatedModel).getUpdater();

                            if (updaterReplica.getStateViewArray() != null)
                                Nd4j.getAffinityManager().ensureLocation(updaterReplica.getStateViewArray(),
                                                AffinityManager.Location.HOST);
                        }

                        running.decrementAndGet();
                    }
                }
            }
        } catch (Exception e) {
            this.thrownException = e;
            throw new RuntimeException(e);
        } finally {
            log.debug("Terminating all workspaces for trainer_{}", threadId);
            Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
            isStopped.set(true);
        }
    }

    @Override
    public void waitTillRunning() {
        while (running.get() != 0) {
            // if Trainer thread got exception during training - rethrow it here
            //log.info("Thread {} running {}", Thread.currentThread().getId(), running.get());
            if (thrownException != null)
                throw new RuntimeException(thrownException);

            LockSupport.parkNanos(1000L);
        }
    }


    @Override
    public boolean averagingRequired() {
        return true;
    }

    protected static TrainingListener cloneListener(TrainingListener original) {
        if (original instanceof RoutingIterationListener) {
            return ((RoutingIterationListener) original).clone();
        }
        return original;
    }


    protected void configureListeners(String workerUUID, Collection<TrainingListener> oldListeners,
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
                    rl.setStorageRouter(parallelWrapper.getStorageRouter());
                }

            }
            if (!replicatedListeners.contains((l))) {
                replicatedListeners.add(l);
            }
        }
    }


    public static class DefaultTrainerBuilder {
        public DefaultTrainerBuilder() {}
    }

}
