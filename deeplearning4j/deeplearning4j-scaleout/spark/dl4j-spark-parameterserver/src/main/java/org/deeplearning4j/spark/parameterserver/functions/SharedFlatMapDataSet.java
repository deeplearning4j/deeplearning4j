package org.deeplearning4j.spark.parameterserver.functions;

import org.datavec.spark.functions.FlatMapFunctionAdapter;
import org.datavec.spark.transform.BaseFlatMapFunctionAdaptee;
import org.deeplearning4j.spark.api.TrainingResult;
import org.deeplearning4j.spark.api.TrainingWorker;
import org.deeplearning4j.spark.parameterserver.pw.SharedTrainingWrapper;
import org.deeplearning4j.spark.parameterserver.training.SharedTrainingResult;
import org.deeplearning4j.spark.parameterserver.training.SharedTrainingWorker;
import org.nd4j.linalg.dataset.DataSet;

import java.util.Collections;
import java.util.Iterator;

/**
 * @author raver119@gmail.com
 */

public class SharedFlatMapDataSet<R extends TrainingResult> extends BaseFlatMapFunctionAdaptee<Iterator<DataSet>, R> {

    public SharedFlatMapDataSet(TrainingWorker<R> worker) {
        super(new SharedFlatMapDataSetAdapter<R>(worker));
    }
}


class SharedFlatMapDataSetAdapter<R extends TrainingResult> implements FlatMapFunctionAdapter<Iterator<DataSet>, R> {

    private final SharedTrainingWorker worker;

    public SharedFlatMapDataSetAdapter(TrainingWorker<R> worker) {
        // we're not going to have anything but Shared classes here ever
        this.worker = (SharedTrainingWorker) worker;
    }

    @Override
    public Iterable<R> call(Iterator<DataSet> dataSetIterator) throws Exception {
        /*
            That's the place where we do our stuff. Here's the plan:
            1) we pass given iterator to VirtualDataSetIterator, which acts as holder for them
            2) Virtual iterator will provide load balancing between available devices
            3) we'll lock out here
         */

        // iterator should be silently attached to VirtualDataSetIterator, and used appropriately
        SharedTrainingWrapper.getInstance().attachDS(dataSetIterator);

        // first callee will become master, others will obey and die
        // all threads in this executor will be blocked here until training finished
        SharedTrainingResult result = SharedTrainingWrapper.getInstance().run(worker);

        return Collections.singletonList((R) result);
    }
}
