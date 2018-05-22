package org.deeplearning4j.spark.api;

import lombok.AllArgsConstructor;
import lombok.Data;

import java.io.Serializable;

/**
 * A simple configuration object (common settings for workers)
 *
 * @author Alex Black
 */
@AllArgsConstructor
@Data
public class WorkerConfiguration implements Serializable {

    protected final boolean isGraphNetwork;
    protected final int dataSetObjectSizeExamples; //Number of examples in each DataSet object
    protected final int batchSizePerWorker;
    protected final int maxBatchesPerWorker;
    protected final int prefetchNumBatches;
    protected final boolean collectTrainingStats;

}
