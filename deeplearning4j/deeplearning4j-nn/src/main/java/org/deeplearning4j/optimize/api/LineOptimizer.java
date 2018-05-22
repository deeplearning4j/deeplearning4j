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

package org.deeplearning4j.optimize.api;

import org.deeplearning4j.exception.InvalidStepException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

import java.io.Serializable;

/**
 * Line optimizer interface adapted from mallet
 * @author Adam Gibson
 *
 */
public interface LineOptimizer extends Serializable {
    /**
     * Line optimizer
     * @param parameters the parameters to optimize
     * @param gradient the gradient
     * @param searchDirection  the point/direction to go in
     * @return the last step size used
     * @throws InvalidStepException
     */
    double optimize(INDArray parameters, INDArray gradient, INDArray searchDirection, LayerWorkspaceMgr workspaceMgr) throws InvalidStepException;



}
