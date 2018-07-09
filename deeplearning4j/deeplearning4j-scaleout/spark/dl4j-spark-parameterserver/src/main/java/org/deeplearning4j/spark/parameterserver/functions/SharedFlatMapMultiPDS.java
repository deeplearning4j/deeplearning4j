package org.deeplearning4j.spark.parameterserver.functions;

import org.apache.spark.input.PortableDataStream;
import org.datavec.spark.functions.FlatMapFunctionAdapter;
import org.datavec.spark.transform.BaseFlatMapFunctionAdaptee;
import org.deeplearning4j.spark.api.TrainingResult;
import org.deeplearning4j.spark.api.TrainingWorker;
import org.deeplearning4j.spark.parameterserver.callbacks.MultiDataSetDeserializationCallback;
import org.deeplearning4j.spark.parameterserver.callbacks.PortableDataStreamMDSCallback;
import org.deeplearning4j.spark.parameterserver.iterators.MultiPdsIterator;
import org.deeplearning4j.spark.parameterserver.pw.SharedTrainingWrapper;
import org.deeplearning4j.spark.parameterserver.training.SharedTrainingResult;
import org.deeplearning4j.spark.parameterserver.training.SharedTrainingWorker;

import java.util.Collections;
import java.util.Iterator;

/**
 * Created by raver119 on 14.06.17.
 */
public class SharedFlatMapMultiPDS<R extends TrainingResult>
                extends BaseFlatMapFunctionAdaptee<Iterator<PortableDataStream>, R> {

    public SharedFlatMapMultiPDS(TrainingWorker<R> worker) {
        this(worker, null);
    }

    public SharedFlatMapMultiPDS(TrainingWorker<R> worker, PortableDataStreamMDSCallback callback) {
        super(new SharedFlatMapMultiPDSAdapter<R>(worker, callback));
    }
}


class SharedFlatMapMultiPDSAdapter<R extends TrainingResult>
                implements FlatMapFunctionAdapter<Iterator<PortableDataStream>, R> {

    protected final SharedTrainingWorker worker;
    protected final PortableDataStreamMDSCallback callback;

    public SharedFlatMapMultiPDSAdapter(TrainingWorker<R> worker) {
        this(worker, null);
    }

    public SharedFlatMapMultiPDSAdapter(TrainingWorker<R> worker, PortableDataStreamMDSCallback callback) {
        // we're not going to have anything but Shared classes here ever
        this.worker = (SharedTrainingWorker) worker;


        if (callback == null) {
            this.callback = new MultiDataSetDeserializationCallback();
        } else {
            this.callback = callback;
        }
    }

    @Override
    public Iterable<R> call(Iterator<PortableDataStream> dataSetIterator) throws Exception {
        //Under some limited circumstances, we might have an empty partition. In this case, we should return immediately
        if(!dataSetIterator.hasNext()){
            return Collections.emptyList();
        }
        // we want to process PDS somehow, and convert to DataSet after all

        // iterator should be silently attached to VirtualDataSetIterator, and used appropriately
        SharedTrainingWrapper.getInstance().attachMDS(new MultiPdsIterator(dataSetIterator, callback));

        // first callee will become master, others will obey and die
        SharedTrainingResult result = SharedTrainingWrapper.getInstance().run(worker);

        return Collections.singletonList((R) result);
    }
}
