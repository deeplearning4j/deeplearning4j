/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.deeplearning4j.rl4j.observation.preprocessors;

import lombok.NonNull;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;

import java.util.ArrayList;
import java.util.List;

/**
 * The PipelineDataSetPreProcessor allows to call a serie of DataSetPreProcessors sequentially.
 * <br />
 * NOTE: This DataSetPreProcessor will return immediately if one of the preProcesosrs return an empty DataSet.
 * Empty datasets should ignored by the Policy/Learning class and other DataSetPreProcessors
 *
 * @author Alexandre Boulanger
 */
public class PipelineDataSetPreProcessor implements DataSetPreProcessor {
    private List<DataSetPreProcessor> preProcessors;

    private PipelineDataSetPreProcessor() {

    }

    /**
     * Pre process a dataset sequentially
     *
     * @param dataSet the data set to pre process
     */
    @Override
    public void preProcess(DataSet dataSet) {
        Preconditions.checkNotNull(dataSet, "Encountered null dataSet");

        for (DataSetPreProcessor preProcessor : preProcessors) {
            if (dataSet.isEmpty()) {
                return;
            }
            preProcessor.preProcess(dataSet);
        }
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private List<DataSetPreProcessor> preProcessors = new ArrayList<>();

        public Builder() {

        }

        public Builder addPreProcessor(@NonNull DataSetPreProcessor preProcessor) {
            preProcessors.add(preProcessor);
            return this;
        }

        public Builder addPreProcessor(int idx, @NonNull DataSetPreProcessor preProcessor) {
            preProcessors.add(idx, preProcessor);
            return this;
        }


        public PipelineDataSetPreProcessor build() {
            PipelineDataSetPreProcessor preProcessor = new PipelineDataSetPreProcessor();
            preProcessor.preProcessors = this.preProcessors;
            return preProcessor;
        }
    }
}


