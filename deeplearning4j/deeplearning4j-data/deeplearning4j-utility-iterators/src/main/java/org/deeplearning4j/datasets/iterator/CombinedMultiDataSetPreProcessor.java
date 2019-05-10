/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

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
