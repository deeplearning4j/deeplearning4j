package org.deeplearning4j.parallelism.trainer;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.NoArgsConstructor;
import lombok.NonNull;
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
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Collection;
import java.util.UUID;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
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
    protected Model originalModel;
    protected Model replicatedModel;
    protected LinkedBlockingQueue<DataSet> queue = new LinkedBlockingQueue<>();
    protected LinkedBlockingQueue<MultiDataSet> queueMDS = new LinkedBlockingQueue<>();
    protected AtomicInteger running = new AtomicInteger(0);
    protected int threadId;
    protected AtomicBoolean shouldUpdate = new AtomicBoolean(false);
    protected AtomicBoolean shouldStop = new AtomicBoolean(false);
    protected Exception thrownException;
    protected volatile boolean useMDS = false;
    protected final String uuid = UUID.randomUUID().toString();
    protected boolean onRootModel = false;
    protected ParallelWrapper parallelWrapper;
    protected WorkspaceMode workspaceMode;



    @Override
    public void feedMultiDataSet(@NonNull MultiDataSet dataSet) {
        setupIfNeccessary();
        running.incrementAndGet();
        queueMDS.add(dataSet);
    }

    @Override
    public void feedDataSet(@NonNull DataSet dataSet) {
        setupIfNeccessary();
        running.incrementAndGet();
        queue.add(dataSet);
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

                if (Nd4j.getExecutioner() instanceof GridExecutioner)
                    ((GridExecutioner) Nd4j.getExecutioner()).flushQueueBlocking();

                updater.setStateViewArray((MultiLayerNetwork) replicatedModel, viewD, false);
            }
        }
        else if (replicatedModel instanceof ComputationGraph) {
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




    protected void setupIfNeccessary() {
        if(queue == null)
            queue = new LinkedBlockingQueue<>();
        if(queueMDS == null)
            queueMDS = new LinkedBlockingQueue<>();
        if(running == null)
            running = new AtomicInteger(0);
        if(shouldStop == null)
            shouldStop = new AtomicBoolean(false);
        if(shouldUpdate == null)
            shouldUpdate = new AtomicBoolean(false);
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
    }

    @Override
    public void run() {
        setupIfNeccessary();
        
        try {
            // we create fresh network, with the same configuration, as initially created by user
            // however, we don't need clone or anything here
            if (originalModel instanceof MultiLayerNetwork) {
                if (!onRootModel) {
                    MultiLayerConfiguration conf = MultiLayerConfiguration.fromJson(((MultiLayerNetwork) originalModel).getLayerWiseConfigurations().toJson());
                    conf.setWorkspaceMode(workspaceMode);
                    this.replicatedModel = new MultiLayerNetwork(conf);

                    replicatedModel.init();

                    // we replicate original model params & updater state, just in case it's pre-trained model
                    synchronized (originalModel) {
                        replicatedModel.setParams(originalModel.params());

                        Updater updaterReplica = ((MultiLayerNetwork) replicatedModel).getUpdater();
                        Updater updaterOrigina = ((MultiLayerNetwork) originalModel).getUpdater();

                        if (updaterOrigina != null && updaterOrigina.getStateViewArray() != null)
                            updaterReplica.setStateViewArray((MultiLayerNetwork) replicatedModel, updaterOrigina.getStateViewArray().dup(), false);

                        if (Nd4j.getExecutioner() instanceof GridExecutioner)
                            ((GridExecutioner) Nd4j.getExecutioner()).flushQueueBlocking();
                    }

                    Collection<IterationListener> oldListeners = ((MultiLayerNetwork) originalModel).getListeners();
                    oldListeners = (oldListeners == null ? new ArrayList<>() : new ArrayList<>(oldListeners));
                    Collection<IterationListener> replicatedListeners = new ArrayList<>();

                    if(parallelWrapper.getListeners() != null) {
                        oldListeners.addAll(parallelWrapper.getListeners());
                    }

                    configureListeners(uuid, oldListeners, replicatedListeners);

                    this.replicatedModel.setListeners(replicatedListeners);
                }
            }
            else if (originalModel instanceof ComputationGraph) {
                if (!onRootModel) {
                    ComputationGraphConfiguration conf = ComputationGraphConfiguration.fromJson(((ComputationGraph) originalModel).getConfiguration().toJson());
                    conf.setWorkspaceMode(workspaceMode);

                    this.replicatedModel = new ComputationGraph(conf);
                    this.replicatedModel.init();

                    // we replicate original model params & updater state, just in case it's pre-trained model
                    synchronized (originalModel) {
                        replicatedModel.setParams(originalModel.params());

                        ComputationGraphUpdater updaterReplica = ((ComputationGraph) replicatedModel).getUpdater();
                        ComputationGraphUpdater updaterOrigina = ((ComputationGraph) originalModel).getUpdater();

                        if (updaterOrigina != null && updaterOrigina.getStateViewArray() != null)
                            updaterReplica.setStateViewArray(updaterOrigina.getStateViewArray().dup());

                        if (Nd4j.getExecutioner() instanceof GridExecutioner)
                            ((GridExecutioner) Nd4j.getExecutioner()).flushQueueBlocking();
                    }

                    Collection<IterationListener> oldListeners = ((ComputationGraph) originalModel).getListeners();
                    oldListeners = (oldListeners == null ? new ArrayList<>() : new ArrayList<>(oldListeners));
                    Collection<IterationListener> replicatedListeners = new ArrayList<>();

                    if(parallelWrapper.getListeners() != null) {
                        oldListeners.addAll(parallelWrapper.getListeners());
                    }
                    configureListeners(uuid, oldListeners, replicatedListeners);

                    this.replicatedModel.setListeners(replicatedListeners);
                }
            }

            if (!useMDS) {
                while (!shouldStop.get()) {
                    DataSet dataSet = queue.poll(100, TimeUnit.MILLISECONDS);
                    if (dataSet != null) {

                        //if (Nd4j.getAffinityManager().getDeviceForCurrentThread() != Nd4j.getAffinityManager().getDeviceForArray(dataSet.getFeatures()))
                        //    log.debug("Thread: {}; Bad align for data: {}/{}", Thread.currentThread().getId(), Nd4j.getAffinityManager().getDeviceForCurrentThread(), Nd4j.getAffinityManager().getDeviceForArray(dataSet.getFeatures()));

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
            }
            else {
                // loop for MultiDataSet
                while (!shouldStop.get()) {
                    MultiDataSet dataSet = queueMDS.poll(100, TimeUnit.MILLISECONDS);
                    if (dataSet != null) {
                        if (replicatedModel instanceof ComputationGraph) {
                            ((ComputationGraph) replicatedModel).fit(dataSet);
                        } else
                            throw new RuntimeException("MultiDataSet can be fit into ComputationGraph only");

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

    @Override
    public void waitTillRunning() {
        while (running.get() != 0) {
            // if Trainer thread got exception during training - rethrow it here
            if (thrownException != null)
                throw new RuntimeException(thrownException);

            LockSupport.parkNanos(50000L);
        }
    }



    protected static IterationListener cloneListener(IterationListener original){
        if(original instanceof RoutingIterationListener){
            return ((RoutingIterationListener) original).clone();
        }
        return original;
    }


    protected void configureListeners(String workerUUID, Collection<IterationListener> oldListeners,
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
                    rl.setStorageRouter(parallelWrapper.getStorageRouter());
                }

            }
            replicatedListeners.add(l);
        }
    }


    public static class DefaultTrainerBuilder {
        public DefaultTrainerBuilder() {
        }
    }

}
