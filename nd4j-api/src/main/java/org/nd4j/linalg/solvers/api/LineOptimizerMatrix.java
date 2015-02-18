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

package org.nd4j.linalg.solvers.api;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.solvers.exception.InvalidStepException;

/**
 * Line optimizer interface adapted from mallet
 *
 * @author Adam Gibson
 */
public interface LineOptimizerMatrix {
    /**
     * Returns the last step size used.
     */
    public double optimize(INDArray line, int iteration, double initialStep) throws InvalidStepException;

    public interface ByGradient {
        /**
         * Returns the last step size used.
         */
        public double optimize(INDArray line, int iteration, double initialStep) throws InvalidStepException;
    }
}
