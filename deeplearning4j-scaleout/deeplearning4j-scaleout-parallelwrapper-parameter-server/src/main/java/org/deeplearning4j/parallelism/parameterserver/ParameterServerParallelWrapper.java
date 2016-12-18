package org.deeplearning4j.parallelism.parameterserver;

import io.aeron.driver.MediaDriver;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
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

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Parallelwrapper using a parameter server
 * for training
 *
 * @author Adam Gibson
 */
@Builder
@Data
public class ParameterServerParallelWrapper {
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

    public void fit(DataSetIterator source) {
        if(!init)
            init(source);
        DataSetIterator iterator;
        if(preFetchSize > 0 && source.asyncSupported())
            iterator = new AsyncDataSetIterator(source,preFetchSize);
        else
            iterator = source;

        while(iterator.hasNext()) {
            DataSet next = iterator.next();
            addObject(next);
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
        parameterServerNode = new ParameterServerNode(mediaDriver);
        running = new AtomicBoolean(true);
        if(parameterServerArgs == null)
            parameterServerArgs = new String[] {
                    "-m","true",
                    "-s","1," + String.valueOf(model.numParams()),
                    "-p","40323",
                    "-h","localhost",
                    "-id","11",
                    "-md", mediaDriver.aeronDirectoryName(),
                    "-sp", "33000",
                    "-u",String.valueOf(numUpdatesPerEpoch)
            };


        linkedBlockingQueue = new LinkedBlockingQueue<>(numWorkers);

        //pass through args for the parameter server subscriber
        parameterServerNode.runMain(parameterServerArgs);

        parameterServerClient = new Trainer[numWorkers];
        executorService = Executors.newFixedThreadPool(numWorkers);

        for(int i = 0; i < numWorkers; i++) {
            parameterServerClient[i] = new Trainer(ParameterServerClient.builder()
                    .aeron(parameterServerNode.getAeron())
                    .ndarrayRetrieveUrl(parameterServerNode.getSubscriber().getResponder().connectionUrl())
                    .ndarraySendUrl(parameterServerNode.getSubscriber().getSubscriber().connectionUrl())
                    .subscriberHost("localhost")
                    .subscriberPort(40625 + i)
                    .subscriberStream(12 + i).build(),running,linkedBlockingQueue,model);
            final int j = i;
            executorService.submit(() -> parameterServerClient[j].start());

        }

        init = true;
    }


    @AllArgsConstructor
    public static class Trainer {
        private ParameterServerClient parameterServerClient;
        private AtomicBoolean running;
        private LinkedBlockingQueue<Object> work;
        private Model model;

        public void start() {
            while(running.get()) {
                try {
                    Object next = work.poll(1, TimeUnit.SECONDS);
                    //send new parameters
                    if(parameterServerClient.isReadyForNext()) {
                        //get the new parameters from the server
                        INDArray newParams = parameterServerClient.getArray();
                        model.setParams(newParams);
                    }


                    if(next instanceof DataSet) {
                        DataSet dataSet = (DataSet) next;
                        if(model instanceof ComputationGraph) {
                            ComputationGraph computationGraph = (ComputationGraph) model;
                            computationGraph.fit(dataSet);
                        }
                        else {
                            MultiLayerNetwork multiLayerNetwork = (MultiLayerNetwork) model;
                            multiLayerNetwork.fit(dataSet);
                        }

                        //send the updated params
                        parameterServerClient.pushNDArray(model.params());

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

                        //send the updated params
                        parameterServerClient.pushNDArray(model.params());

                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
        }

    }


}
