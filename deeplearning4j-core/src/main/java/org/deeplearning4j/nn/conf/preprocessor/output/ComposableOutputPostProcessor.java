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

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.conf.OutputPostProcessor;
import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Composable output post processor
 *
 * @author Adam Gibson
 */
public class ComposableOutputPostProcessor implements OutputPostProcessor {
	private static final long serialVersionUID = 4406818294012989149L;
	private OutputPostProcessor[] outputPostProcessors;

    public ComposableOutputPostProcessor(OutputPostProcessor[] outputPostProcessors) {
        this.outputPostProcessors = outputPostProcessors;
    }

    @Override
    public INDArray process(INDArray output) {
        for(OutputPostProcessor outputPostProcessor : outputPostProcessors)
          output = outputPostProcessor.process(output);
        return output;
    }

    @Override
    public Pair<Gradient,INDArray> backprop(Pair<Gradient,INDArray> input) {
        for (OutputPostProcessor preProcessor : outputPostProcessors)
            input = preProcessor.backprop(input);
        return input;
    }
}
