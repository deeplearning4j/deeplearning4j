package org.nd4j.linalg.dataset.api.preprocessor;

import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;

/**
 * A simple Composite MultiDataSetPreProcessor - allows you to apply multiple MultiDataSetPreProcessors sequentially
 * on the one MultiDataSet, in the order they are passed to the constructor
 *
 * @author Alex Black
 */
public class CompositeMultiDataSetPreProcessor implements MultiDataSetPreProcessor {

    private MultiDataSetPreProcessor[] preProcessors;

    /**
     * @param preProcessors Preprocessors to apply. They will be applied in this order
     */
    public CompositeMultiDataSetPreProcessor(MultiDataSetPreProcessor... preProcessors){
        this.preProcessors = preProcessors;
    }

    @Override
    public void preProcess(MultiDataSet multiDataSet) {
        for(MultiDataSetPreProcessor p : preProcessors){
            p.preProcess(multiDataSet);
        }
    }
}
