package org.deeplearning4j.spark.api.worker;

import org.apache.spark.api.java.function.FlatMapFunction;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.api.TrainingResult;
import org.deeplearning4j.spark.api.TrainingWorker;
import org.deeplearning4j.spark.api.WorkerConfiguration;
import org.nd4j.linalg.dataset.api.DataSet;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Created by Alex on 14/06/2016.
 */
public class ExecuteWorkerFlatMap<R extends TrainingResult> implements FlatMapFunction<Iterator<DataSet>, R> {

    private final TrainingWorker<R> worker;

    public ExecuteWorkerFlatMap(TrainingWorker<R> worker){
        this.worker = worker;
    }

    @Override
    public Iterable<R> call(Iterator<DataSet> dataSetIterator) throws Exception {
        WorkerConfiguration dataConfig = worker.getDataConfiguration();
        int batchSize = dataConfig.getBatchSizePerWorker();


        //TODO: handle prefetching
        //TODO: move this data set iterator functionality into an actual DataSetIterator

        MultiLayerNetwork net = worker.getInitialModel();

        //TODO replace this with something more elegant
        while(dataSetIterator.hasNext()){
            List<DataSet> list = new ArrayList<>();
            int batchSizeSoFar = 0;
            while(dataSetIterator.hasNext() && batchSizeSoFar < batchSize){
                DataSet next =
            }
        }
    }
}
