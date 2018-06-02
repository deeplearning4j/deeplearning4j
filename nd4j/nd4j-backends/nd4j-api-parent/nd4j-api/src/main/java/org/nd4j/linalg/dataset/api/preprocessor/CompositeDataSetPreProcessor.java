package org.nd4j.linalg.dataset.api.preprocessor;

import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;

/**
 * A simple Composite DataSetPreProcessor - allows you to apply multiple DataSetPreProcessors sequentially
 * on the one DataSet, in the order they are passed to the constructor
 *
 * @author Alex Black
 */
public class CompositeDataSetPreProcessor implements DataSetPreProcessor {

    private DataSetPreProcessor[] preProcessors;

    /**
     * @param preProcessors Preprocessors to apply. They will be applied in this order
     */
    public CompositeDataSetPreProcessor(DataSetPreProcessor... preProcessors){
        this.preProcessors = preProcessors;
    }

    @Override
    public void preProcess(DataSet dataSet) {
        for(DataSetPreProcessor p : preProcessors){
            p.preProcess(dataSet);
        }
    }
}
