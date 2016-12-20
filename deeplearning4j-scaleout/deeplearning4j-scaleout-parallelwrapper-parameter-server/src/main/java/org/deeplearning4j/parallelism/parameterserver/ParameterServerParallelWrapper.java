package org.deeplearning4j.parallelism.parameterserver;

import io.aeron.driver.MediaDriver;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.agrona.CloseHelper;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.datasets.iterator.AsyncMultiDataSetIterator;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.parameterserver.client.ParameterServerClient;
import org.nd4j.parameterserver.node.ParameterServerNode;

import java.io.Closeable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Parallelwrapper using
 * a parameter server
 * for training
 *
 * @author Adam Gibson
 */
@Builder
@Data
@Slf4j
public class ParameterServerParallelWrapper implements AutoCloseable {
    private ExecutorService executorService;
    private int numWorkers;
    private Trainer[] parameterServerClient;
    private ParameterServerNode parameterServerNode;
    private MediaDriver mediaDriver;
    private MediaDriver.Context mediaDriverContext;
    private boolean init = false;
    private Model model;
    private ComputationGraph computationGraph;
    private MultiLayerNetwork multiLayerNetwork;
    //work queue for datasets
    private LinkedBlockingQueue<Object> linkedBlockingQueue;
    private AtomicBoolean running;
    private int preFetchSize;
    private String[] parameterServerArgs;
    private int numUpdatesPerEpoch;
    private int numEpochs;
    private int statusServerPort = 33000;
    public void fit(DataSetIterator source) {
        if(!init)
            init(source);
        DataSetIterator iterator;
        if(preFetchSize > 0 && source.asyncSupported())
            iterator = new AsyncDataSetIterator(source,preFetchSize);
        else
            iterator = source;
        for(int i = 0; i < numEpochs; i++) {
            while (iterator.hasNext()) {
                DataSet next = iterator.next();
                addObject(next);
            }

            iterator.reset();

            log.info(String.format("Completed epoch %d",i));
        }

    }




    public void fit(MultiDataSetIterator multiDataSetIterator) {
        if(!init)
            init(multiDataSetIterator);

        MultiDataSetIterator iterator = null;
        if (preFetchSize > 0 && multiDataSetIterator.asyncSupported()) {
            iterator = new AsyncMultiDataSetIterator(multiDataSetIterator, preFetchSize);
        } else iterator = multiDataSetIterator;

        while(iterator.hasNext()) {
            org.nd4j.linalg.dataset.api.MultiDataSet next = iterator.next();
            addObject(next);
        }
    }

    //poll when workers are at capacity
    private void addObject(Object next) {
        try {
            while(!linkedBlockingQueue.offer(next,1,TimeUnit.SECONDS))
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }


    private int numUpdatesPerEpoch(MultiDataSetIterator iterator) {
        if(!iterator.resetSupported())
            throw new IllegalStateException("Iterator must support reset()");
        int ret = 0;
        while(iterator.hasNext()) {
            iterator.next();
            ret++;
        }

        iterator.reset();

        return ret;
    }


    private int numUpdatesPerEpoch(DataSetIterator iterator) {
        if(!iterator.resetSupported())
            throw new IllegalStateException("Iterator must support reset()");
        int ret = 0;
        while(iterator.hasNext()) {
            iterator.next();
            ret++;
        }

        iterator.reset();

        return ret;
    }


    private void  init(Object iterator) {
        if(numEpochs < 1)
            throw new IllegalStateException("numEpochs must be >= 1");
        //determine the number of updates per epoch (number of minibatches total for this iterator)
        //TODO: make this efficient
        if(iterator instanceof DataSetIterator) {
            DataSetIterator dataSetIterator = (DataSetIterator) iterator;
            numUpdatesPerEpoch = numUpdatesPerEpoch(dataSetIterator);
        }
        else if(iterator instanceof MultiDataSetIterator){
            MultiDataSetIterator iterator1 = (MultiDataSetIterator) iterator;
            numUpdatesPerEpoch = numUpdatesPerEpoch(iterator1);

        }
        else
            throw new IllegalArgumentException("Illegal type of object passed in for initialization. Must be of type DataSetIterator or MultiDataSetIterator");

        mediaDriverContext = new MediaDriver.Context();
        mediaDriver = MediaDriver.launchEmbedded(mediaDriverContext);
        parameterServerNode = new ParameterServerNode(mediaDriver,statusServerPort,numWorkers);
        running = new AtomicBoolean(true);
        if(parameterServerArgs == null)
            parameterServerArgs = new String[] {
                    "-m","true",
                    "-s","1," + String.valueOf(model.numParams()),
                    "-p","40323",
                    "-h","localhost",
                    "-id","11",
                    "-md", mediaDriver.aeronDirectoryName(),
                    "-sh", "localhost",
                    "-sp", String.valueOf(statusServerPort),
                    "-u",String.valueOf(numUpdatesPerEpoch)
            };

        if(numWorkers == 0)
            numWorkers =Runtime.getRuntime().availableProcessors();

        linkedBlockingQueue = new LinkedBlockingQueue<>(numWorkers);

        //pass through args for the parameter server subscriber
        parameterServerNode.runMain(parameterServerArgs);

        while(!parameterServerNode.subscriberLaunched()) {
            try {
                Thread.sleep(10000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }


        try {
            Thread.sleep(10000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }


        log.info("Parameter server started");

        parameterServerClient = new Trainer[numWorkers];
        executorService = Executors.newFixedThreadPool(numWorkers);

        for(int i = 0; i < numWorkers; i++) {
            Model model = null;
            if(this.model instanceof ComputationGraph) {
                ComputationGraph computationGraph = (ComputationGraph) this.model;
                model = computationGraph.clone();
            }
            else if(this.model instanceof MultiLayerNetwork) {
                MultiLayerNetwork multiLayerNetwork = (MultiLayerNetwork) this.model;
                model = multiLayerNetwork.clone();
            }
            parameterServerClient[i] = new Trainer(ParameterServerClient.builder()
                    .aeron(parameterServerNode.getAeron())
                    .ndarrayRetrieveUrl(parameterServerNode.getSubscriber()[i].getResponder().connectionUrl())
                    .ndarraySendUrl(parameterServerNode.getSubscriber()[i].getSubscriber().connectionUrl())
                    .subscriberHost("localhost").masterStatusHost("localhost").masterStatusPort(statusServerPort)
                    .subscriberPort(40625 + i)
                    .subscriberStream(12 + i).build(),running,linkedBlockingQueue,model);
            final int j = i;
            executorService.submit(() -> parameterServerClient[j].start());

        }

        init = true;
        log.info("Initialized wrapper");
    }

    /**
     * Closes this resource, relinquishing any underlying resources.
     * This method is invoked automatically on objects managed by the
     * {@code try}-with-resources statement.
     * <p>
     * <p>While this interface method is declared to throw {@code
     * Exception}, implementers are <em>strongly</em> encouraged to
     * declare concrete implementations of the {@code close} method to
     * throw more specific exceptions, or to throw no exception at all
     * if the close operation cannot fail.
     * <p>
     * <p> Cases where the close operation may fail require careful
     * attention by implementers. It is strongly advised to relinquish
     * the underlying resources and to internally <em>mark</em> the
     * resource as closed, prior to throwing the exception. The {@code
     * close} method is unlikely to be invoked more than once and so
     * this ensures that the resources are released in a timely manner.
     * Furthermore it reduces problems that could arise when the resource
     * wraps, or is wrapped, by another resource.
     * <p>
     * <p><em>Implementers of this interface are also strongly advised
     * to not have the {@code close} method throw {@link
     * InterruptedException}.</em>
     * <p>
     * This exception interacts with a thread's interrupted status,
     * and runtime misbehavior is likely to occur if an {@code
     * InterruptedException} is {@linkplain Throwable#addSuppressed
     * suppressed}.
     * <p>
     * More generally, if it would cause problems for an
     * exception to be suppressed, the {@code AutoCloseable.close}
     * method should not throw it.
     * <p>
     * <p>Note that unlike the {@link Closeable#close close}
     * method of {@link Closeable}, this {@code close} method
     * is <em>not</em> required to be idempotent.  In other words,
     * calling this {@code close} method more than once may have some
     * visible side effect, unlike {@code Closeable.close} which is
     * required to have no effect if called more than once.
     * <p>
     * However, implementers of this interface are strongly encouraged
     * to make their {@code close} methods idempotent.
     *
     * @throws Exception if this resource cannot be closed
     */
    @Override
    public void close() throws Exception {
        if(executorService != null)
            executorService.shutdown();
        if(mediaDriver != null)
            CloseHelper.close(mediaDriver);
        if(parameterServerNode != null)
            parameterServerNode.close();
    }


    @AllArgsConstructor
    public static class Trainer implements AutoCloseable {
        private ParameterServerClient parameterServerClient;
        private AtomicBoolean running;
        private LinkedBlockingQueue<Object> work;
        private Model model;

        public void start() {
            log.info("Begin polling running queue");
            while(running.get()) {
                try {
                    Object next = work.poll(1, TimeUnit.SECONDS);
                    if(next == null)
                        continue;
                    //send new parameters
                    if(parameterServerClient.isReadyForNext()) {
                        log.info("Retrieving new array");
                        //get the new parameters from the server
                        INDArray newParams = parameterServerClient.getArray();
                        model.setParams(newParams);
                        log.info("Set new params");
                    }
                    else
                        log.debug("Continuing training");


                    if(next instanceof DataSet) {
                        DataSet dataSet = (DataSet) next;
                        if(model instanceof ComputationGraph) {
                            ComputationGraph computationGraph = (ComputationGraph) model;
                            computationGraph.fit(dataSet);
                        }
                        else {
                            MultiLayerNetwork multiLayerNetwork = (MultiLayerNetwork) model;
                            log.info("Calling fit on multi layer network");
                            multiLayerNetwork.fit(dataSet);

                        }

                        log.info("About to send params in");
                        //send the updated params
                        parameterServerClient.pushNDArray(model.params());
                        log.info("Sent params");

                    }
                    else {
                        MultiDataSet dataSet = (MultiDataSet) next;
                        if(model instanceof ComputationGraph) {
                            ComputationGraph computationGraph = (ComputationGraph) model;
                            computationGraph.fit(dataSet);
                        }
                        else {
                            throw new IllegalArgumentException("MultiLayerNetworks can't fit multi datasets");
                        }

                        log.info("Sending parameters");
                        //send the updated params
                        parameterServerClient.pushNDArray(model.params());

                    }
                } catch (InterruptedException e) {
                    e.printStackTrace();
                    Thread.currentThread().interrupt();
                }
            }

            log.info("Worker finished");
        }

        /**
         * Closes this resource, relinquishing any underlying resources.
         * This method is invoked automatically on objects managed by the
         * {@code try}-with-resources statement.
         * <p>
         * <p>While this interface method is declared to throw {@code
         * Exception}, implementers are <em>strongly</em> encouraged to
         * declare concrete implementations of the {@code close} method to
         * throw more specific exceptions, or to throw no exception at all
         * if the close operation cannot fail.
         * <p>
         * <p> Cases where the close operation may fail require careful
         * attention by implementers. It is strongly advised to relinquish
         * the underlying resources and to internally <em>mark</em> the
         * resource as closed, prior to throwing the exception. The {@code
         * close} method is unlikely to be invoked more than once and so
         * this ensures that the resources are released in a timely manner.
         * Furthermore it reduces problems that could arise when the resource
         * wraps, or is wrapped, by another resource.
         * <p>
         * <p><em>Implementers of this interface are also strongly advised
         * to not have the {@code close} method throw {@link
         * InterruptedException}.</em>
         * <p>
         * This exception interacts with a thread's interrupted status,
         * and runtime misbehavior is likely to occur if an {@code
         * InterruptedException} is {@linkplain Throwable#addSuppressed
         * suppressed}.
         * <p>
         * More generally, if it would cause problems for an
         * exception to be suppressed, the {@code AutoCloseable.close}
         * method should not throw it.
         * <p>
         * <p>Note that unlike the {@link Closeable#close close}
         * method of {@link Closeable}, this {@code close} method
         * is <em>not</em> required to be idempotent.  In other words,
         * calling this {@code close} method more than once may have some
         * visible side effect, unlike {@code Closeable.close} which is
         * required to have no effect if called more than once.
         * <p>
         * However, implementers of this interface are strongly encouraged
         * to make their {@code close} methods idempotent.
         *
         * @throws Exception if this resource cannot be closed
         */
        @Override
        public void close() throws Exception {
        }
    }


}
