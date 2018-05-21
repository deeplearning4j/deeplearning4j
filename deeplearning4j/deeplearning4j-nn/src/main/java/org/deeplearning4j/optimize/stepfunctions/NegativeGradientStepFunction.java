/*-
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

package org.deeplearning4j.optimize.stepfunctions;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.optimize.api.StepFunction;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Subtract the line
 *
 * @author Adam Gibson
 */
@Slf4j
public class NegativeGradientStepFunction implements StepFunction {
    @Override
    public void step(INDArray x, INDArray line, double step) {
        step(x, line);
    }

    @Override
    public void step(INDArray x, INDArray line) {
        x.subi(line);
    }

    @Override
    public void step() {

    }
}
