package org.nd4j.linalg.dataset.api;

/**PreProcessor interface for MultiDataSets
 */
public interface MultiDataSetPreProcessor {

    /** Preprocess the MultiDataSet */
    void preProcess(MultiDataSet multiDataSet);

}
