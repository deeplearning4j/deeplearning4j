package org.deeplearning4j.spark.parameterserver.functions;

import org.apache.spark.input.PortableDataStream;
import org.datavec.spark.functions.FlatMapFunctionAdapter;
import org.datavec.spark.transform.BaseFlatMapFunctionAdaptee;
import org.deeplearning4j.spark.api.TrainingResult;
import org.deeplearning4j.spark.api.TrainingWorker;
import org.deeplearning4j.spark.parameterserver.callbacks.PortableDataStreamCallback;
import org.deeplearning4j.spark.parameterserver.pw.SharedTrainingWrapper;
import org.deeplearning4j.spark.parameterserver.training.SharedTrainingWorker;
import org.nd4j.linalg.dataset.DataSet;

import java.util.Iterator;

public class SharedFlatMapPDS<R extends TrainingResult> extends BaseFlatMapFunctionAdaptee<Iterator<PortableDataStream>, R> {

    public SharedFlatMapPDS(TrainingWorker<R> worker, PortableDataStreamCallback callback) {
        super(new SharedFlatMapPDSAdapter<R>(worker, callback));
    }
}


class SharedFlatMapPDSAdapter<R extends TrainingResult> implements FlatMapFunctionAdapter<Iterator<PortableDataStream>, R> {

    protected final SharedTrainingWorker worker;
    protected final PortableDataStreamCallback callback;

    public SharedFlatMapPDSAdapter(TrainingWorker<R> worker, PortableDataStreamCallback callback) {
        // we're not going to have anything but Shared classes here ever
        this.worker = (SharedTrainingWorker) worker;
        this.callback = callback;
    }

    @Override
    public Iterable<R> call(Iterator<PortableDataStream> dataSetIterator) throws Exception {
        // we want to process PDS somehow, and convert to DataSet after all

        // iterator should be silently attached to VirtualDataSetIterator, and used appropriately
        //SharedTrainingWrapper.getInstance().attachDS(dataSetIterator);

        // first callee will become master, others will obey and die
        SharedTrainingWrapper.getInstance().run(worker);

        // all threads in this executor will be blocked here until training finished
        SharedTrainingWrapper.getInstance().blockUntilFinished();

        // TODO: return result here, probably singleton list though
        return null;
    }
}
