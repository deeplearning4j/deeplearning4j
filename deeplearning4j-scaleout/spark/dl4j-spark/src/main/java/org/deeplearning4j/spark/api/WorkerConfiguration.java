package org.deeplearning4j.spark.api;

import lombok.AllArgsConstructor;
import lombok.Data;

import java.io.Serializable;

/**
 * Created by Alex on 14/06/2016.
 */
@AllArgsConstructor @Data
public class WorkerConfiguration implements Serializable {

    protected final boolean isGraphNetwork;
    protected final int batchSizePerWorker;
    protected final int maxBatchesPerWorker;
    protected final int prefetchNumBatches;
    protected final boolean collectTrainingStats;

}
