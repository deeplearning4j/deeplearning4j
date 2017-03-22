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

package org.deeplearning4j.optimize.terminations;

import org.deeplearning4j.optimize.api.TerminationCondition;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Terminate if the norm2 of the gradient is < a certain tolerance
 */
public class Norm2Termination implements TerminationCondition {
    private double gradientTolerance = 1e-3;

    public Norm2Termination(double gradientTolerance) {
        this.gradientTolerance = gradientTolerance;
    }

    @Override
    public boolean terminate(double cost, double oldCost, Object[] otherParams) {
        INDArray line = (INDArray) otherParams[0];
        double norm2 = line.norm2(Integer.MAX_VALUE).getDouble(0);
        return norm2 < gradientTolerance;
    }
}
