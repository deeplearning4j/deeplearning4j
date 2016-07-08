package org.deeplearning4j.spark.api.worker;

import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.input.PortableDataStream;
import org.deeplearning4j.spark.api.TrainingResult;
import org.deeplearning4j.spark.api.TrainingWorker;
import org.deeplearning4j.spark.iterator.PortableDataStreamMultiDataSetIterator;
import org.nd4j.linalg.dataset.api.MultiDataSet;

import java.util.Iterator;

/**
 * A FlatMapFunction for executing training on serialized MultiDataSet objects, that can be loaded using a PortableDataStream
 * Used for SparkComputationGraph implementations only
 *
 * @author Alex Black
 */
public class ExecuteWorkerPDSMDSFlatMap<R extends TrainingResult> implements FlatMapFunction<Iterator<PortableDataStream>, R> {
    private final FlatMapFunction<Iterator<MultiDataSet>, R> workerFlatMap;

    public ExecuteWorkerPDSMDSFlatMap(TrainingWorker<R> worker){
        this.workerFlatMap = new ExecuteWorkerMultiDataSetFlatMap<>(worker);
    }

    @Override
    public Iterable<R> call(Iterator<PortableDataStream> iter) throws Exception {
        return workerFlatMap.call(new PortableDataStreamMultiDataSetIterator(iter));
    }
}
