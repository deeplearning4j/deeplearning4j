package org.deeplearning4j.spark.api.worker;

import org.apache.spark.input.PortableDataStream;
import org.datavec.spark.functions.FlatMapFunctionAdapter;
import org.datavec.spark.transform.BaseFlatMapFunctionAdaptee;
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
public class ExecuteWorkerPDSMDSFlatMap<R extends TrainingResult>
                extends BaseFlatMapFunctionAdaptee<Iterator<PortableDataStream>, R> {

    public ExecuteWorkerPDSMDSFlatMap(TrainingWorker<R> worker) {
        super(new ExecuteWorkerPDSMDSFlatMapAdapter<>(worker));
    }
}


/**
 * A FlatMapFunction for executing training on serialized MultiDataSet objects, that can be loaded using a PortableDataStream
 * Used for SparkComputationGraph implementations only
 *
 * @author Alex Black
 */
class ExecuteWorkerPDSMDSFlatMapAdapter<R extends TrainingResult>
                implements FlatMapFunctionAdapter<Iterator<PortableDataStream>, R> {
    private final FlatMapFunctionAdapter<Iterator<MultiDataSet>, R> workerFlatMap;

    public ExecuteWorkerPDSMDSFlatMapAdapter(TrainingWorker<R> worker) {
        this.workerFlatMap = new ExecuteWorkerMultiDataSetFlatMapAdapter<>(worker);
    }

    @Override
    public Iterable<R> call(Iterator<PortableDataStream> iter) throws Exception {
        return workerFlatMap.call(new PortableDataStreamMultiDataSetIterator(iter));
    }
}
