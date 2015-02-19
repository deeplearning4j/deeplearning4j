/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.deeplearning4j.nn.conf.preprocessor;

import org.deeplearning4j.nn.conf.OutputPreProcessor;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Aggregate output pre processor
 * @author Adam Gibson
 */
public class AggregatePreProcessor implements OutputPreProcessor {
    private OutputPreProcessor[] preProcessor;

    public AggregatePreProcessor(OutputPreProcessor[] preProcessor) {
        this.preProcessor = preProcessor;
    }

    @Override
    public INDArray preProcess(INDArray output) {
        for(OutputPreProcessor outputPreProcessor : preProcessor)
            output = outputPreProcessor.preProcess(output);
        return output;
    }
}
