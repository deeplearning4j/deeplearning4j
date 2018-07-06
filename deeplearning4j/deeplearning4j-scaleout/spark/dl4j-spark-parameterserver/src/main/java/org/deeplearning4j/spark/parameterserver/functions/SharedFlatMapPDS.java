package org.deeplearning4j.spark.parameterserver.functions;

import org.apache.spark.input.PortableDataStream;
import org.datavec.spark.functions.FlatMapFunctionAdapter;
import org.datavec.spark.transform.BaseFlatMapFunctionAdaptee;
import org.deeplearning4j.spark.api.TrainingResult;
import org.deeplearning4j.spark.api.TrainingWorker;
import org.deeplearning4j.spark.parameterserver.callbacks.DataSetDeserializationCallback;
import org.deeplearning4j.spark.parameterserver.callbacks.PortableDataStreamCallback;
import org.deeplearning4j.spark.parameterserver.iterators.PdsIterator;
import org.deeplearning4j.spark.parameterserver.pw.SharedTrainingWrapper;
import org.deeplearning4j.spark.parameterserver.training.SharedTrainingResult;
import org.deeplearning4j.spark.parameterserver.training.SharedTrainingWorker;

import java.util.Collections;
import java.util.Iterator;

public class SharedFlatMapPDS<R extends TrainingResult>
                extends BaseFlatMapFunctionAdaptee<Iterator<PortableDataStream>, R> {

    public SharedFlatMapPDS(TrainingWorker<R> worker) {
        this(worker, null);
    }

    public SharedFlatMapPDS(TrainingWorker<R> worker, PortableDataStreamCallback callback) {
        super(new SharedFlatMapPDSAdapter<R>(worker, callback));
    }
}


class SharedFlatMapPDSAdapter<R extends TrainingResult>
                implements FlatMapFunctionAdapter<Iterator<PortableDataStream>, R> {

    protected final SharedTrainingWorker worker;
    protected final PortableDataStreamCallback callback;

    public SharedFlatMapPDSAdapter(TrainingWorker<R> worker) {
        this(worker, null);
    }

    public SharedFlatMapPDSAdapter(TrainingWorker<R> worker, PortableDataStreamCallback callback) {
        // we're not going to have anything but Shared classes here ever
        this.worker = (SharedTrainingWorker) worker;


        if (callback == null) {
            this.callback = new DataSetDeserializationCallback();
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
        SharedTrainingWrapper.getInstance().attachDS(new PdsIterator(dataSetIterator, callback));

        // first callee will become master, others will obey and die
        SharedTrainingResult result = SharedTrainingWrapper.getInstance().run(worker);

        return Collections.singletonList((R) result);
    }
}
