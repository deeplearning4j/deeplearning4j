package org.deeplearning4j.spark.parameterserver.functions;

import org.datavec.spark.functions.FlatMapFunctionAdapter;
import org.datavec.spark.transform.BaseFlatMapFunctionAdaptee;
import org.deeplearning4j.spark.api.TrainingResult;
import org.deeplearning4j.spark.api.TrainingWorker;
import org.deeplearning4j.spark.iterator.PathSparkDataSetIterator;
import org.deeplearning4j.spark.parameterserver.pw.SharedTrainingWrapper;
import org.deeplearning4j.spark.parameterserver.training.SharedTrainingResult;
import org.deeplearning4j.spark.parameterserver.training.SharedTrainingWorker;

import java.util.Collections;
import java.util.Iterator;

/**
 *
 * @author raver119@gmail.com
 */
public class SharedFlatMapPaths<R extends TrainingResult> extends BaseFlatMapFunctionAdaptee<Iterator<String>, R> {

    public SharedFlatMapPaths(TrainingWorker<R> worker) {
        super(new SharedFlatMapPathsAdapter<R>(worker));
    }
}


class SharedFlatMapPathsAdapter<R extends TrainingResult> implements FlatMapFunctionAdapter<Iterator<String>, R> {

    protected final SharedTrainingWorker worker;

    public SharedFlatMapPathsAdapter(TrainingWorker<R> worker) {
        // we're not going to have anything but Shared classes here ever
        this.worker = (SharedTrainingWorker) worker;
    }

    @Override
    public Iterable<R> call(Iterator<String> dataSetIterator) throws Exception {
        // here we'll be converting out Strings coming out of iterator to DataSets
        // PathSparkDataSetIterator does that for us

        // iterator should be silently attached to VirtualDataSetIterator, and used appropriately
        SharedTrainingWrapper.getInstance().attachDS(new PathSparkDataSetIterator(dataSetIterator));

        // first callee will become master, others will obey and die
        SharedTrainingResult result = SharedTrainingWrapper.getInstance().run(worker);

        return Collections.singletonList((R) result);
    }
}
