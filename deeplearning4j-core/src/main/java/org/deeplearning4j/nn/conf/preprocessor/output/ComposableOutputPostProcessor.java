/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.nn.conf.preprocessor.output;

import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.OutputPostProcessor;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Composable output post processor
 *
 * @author Adam Gibson
 */
public class ComposableOutputPostProcessor implements OutputPostProcessor {
    private InputPreProcessor[] inputPreProcessors;
    private OutputPostProcessor[] outputPostProcessors;

    public ComposableOutputPostProcessor(InputPreProcessor[] inputPreProcessor, OutputPostProcessor[] outputPostProcessors) {
        this.inputPreProcessors = inputPreProcessors;
        this.outputPostProcessors = outputPostProcessors;
    }

    @Override
    public INDArray preProcess(INDArray output) {
        for(OutputPostProcessor outputPostProcessor : outputPostProcessors)
          output = outputPostProcessor.preProcess(output);
        return output;
    }

    @Override
    public INDArray backprop(INDArray input) {
        for (InputPreProcessor preProcessor : inputPreProcessors)
            input = preProcessor.preProcess(input);
        return input;

    }
}
