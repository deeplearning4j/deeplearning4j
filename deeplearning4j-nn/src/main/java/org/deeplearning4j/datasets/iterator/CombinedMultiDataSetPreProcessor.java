package org.deeplearning4j.datasets.iterator;

import lombok.NonNull;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;

import java.util.ArrayList;
import java.util.List;

/**
 * Combines various multidataset preprocessors
 * Applied in the order they are specified to in the builder
 */
public class CombinedMultiDataSetPreProcessor implements MultiDataSetPreProcessor {

    private List<MultiDataSetPreProcessor> preProcessors;

    private CombinedMultiDataSetPreProcessor() {

    }

    @Override
    public void preProcess(MultiDataSet multiDataSet) {
        for (MultiDataSetPreProcessor preProcessor : preProcessors) {
            preProcessor.preProcess(multiDataSet);
        }
    }

    public static class Builder {
        private List<MultiDataSetPreProcessor> preProcessors = new ArrayList<>();

        public Builder() {

        }

        /**
         * @param preProcessor to be added to list of preprocessors to be applied
         */
        public Builder addPreProcessor(@NonNull MultiDataSetPreProcessor preProcessor) {
            preProcessors.add(preProcessor);
            return this;
        }

        /**
         * Inserts the specified preprocessor at the specified position to the list of preprocessors to be applied
         * @param idx the position to apply the specified preprocessor at
         * @param preProcessor to be added to list of preprocessors to be applied
         */
        public Builder addPreProcessor(int idx, @NonNull MultiDataSetPreProcessor preProcessor) {
            preProcessors.add(idx, preProcessor);
            return this;
        }

        public CombinedMultiDataSetPreProcessor build() {
            CombinedMultiDataSetPreProcessor preProcessor = new CombinedMultiDataSetPreProcessor();
            preProcessor.preProcessors = this.preProcessors;
            return preProcessor;
        }
    }
}
