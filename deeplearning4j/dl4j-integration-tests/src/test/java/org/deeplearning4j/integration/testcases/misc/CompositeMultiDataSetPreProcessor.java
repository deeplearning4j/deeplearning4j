package org.deeplearning4j.integration.testcases.misc;

import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;

public class CompositeMultiDataSetPreProcessor implements MultiDataSetPreProcessor {

    private MultiDataSetPreProcessor[] preProcessors;

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
