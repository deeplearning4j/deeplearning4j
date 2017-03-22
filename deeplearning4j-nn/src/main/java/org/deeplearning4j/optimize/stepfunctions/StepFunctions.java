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

import org.deeplearning4j.optimize.api.StepFunction;

public class StepFunctions {

    private static final DefaultStepFunction DEFAULT_STEP_FUNCTION_INSTANCE = new DefaultStepFunction();
    private static final GradientStepFunction GRADIENT_STEP_FUNCTION_INSTANCE = new GradientStepFunction();
    private static final NegativeDefaultStepFunction NEGATIVE_DEFAULT_STEP_FUNCTION_INSTANCE =
                    new NegativeDefaultStepFunction();
    private static final NegativeGradientStepFunction NEGATIVE_GRADIENT_STEP_FUNCTION_INSTANCE =
                    new NegativeGradientStepFunction();

    private StepFunctions() {}

    public static StepFunction createStepFunction(org.deeplearning4j.nn.conf.stepfunctions.StepFunction stepFunction) {
        if (stepFunction == null)
            return null;
        if (stepFunction instanceof org.deeplearning4j.nn.conf.stepfunctions.DefaultStepFunction)
            return DEFAULT_STEP_FUNCTION_INSTANCE;
        if (stepFunction instanceof org.deeplearning4j.nn.conf.stepfunctions.GradientStepFunction)
            return GRADIENT_STEP_FUNCTION_INSTANCE;
        if (stepFunction instanceof org.deeplearning4j.nn.conf.stepfunctions.NegativeDefaultStepFunction)
            return NEGATIVE_DEFAULT_STEP_FUNCTION_INSTANCE;
        if (stepFunction instanceof org.deeplearning4j.nn.conf.stepfunctions.NegativeGradientStepFunction)
            return NEGATIVE_GRADIENT_STEP_FUNCTION_INSTANCE;

        throw new RuntimeException("unknown step function: " + stepFunction);
    }
}
