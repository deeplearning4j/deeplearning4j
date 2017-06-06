package org.deeplearning4j.spark.parameterserver.pw;

import org.deeplearning4j.parallelism.ParallelWrapper;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;

/**
 * This class maintains ParallelWrapper instance in Spark environment, and provides primitives for inter-executor
 * communication during training over partitions.
 *
 * @author raver119@gmail.com
 */
public class SharedTrainingWrapper {
    public static SharedTrainingWrapper INSTANCE = new SharedTrainingWrapper();
    protected ParallelWrapper wrapper;

    protected SharedTrainingWrapper() {
        // do nothing here
    }

    public static SharedTrainingWrapper getInstance() {
        return INSTANCE;
    }

    public void run() {
        /*
            first call instantiates pw, messenger etc, and gets in charge here.
         */
    }

    public void passDataSet(DataSet dataSet) {
        // we're going to save this dataset into VirtualDataSetIterator
    }

    public void passDataSet(MultiDataSet dataSet) {
        // we're going to save this dataset into VirtualMultiDataSetIterator
    }
}
