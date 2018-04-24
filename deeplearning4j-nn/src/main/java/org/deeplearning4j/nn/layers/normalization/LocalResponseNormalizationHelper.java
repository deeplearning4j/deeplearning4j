/*-
 *
 *  * Copyright 2016 Skymind,Inc.
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
package org.deeplearning4j.nn.layers.normalization;

import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

/**
 * Helper for the local response normalization layer.
 *
 * @author saudet
 */
public interface LocalResponseNormalizationHelper {
    boolean checkSupported(double k, double n, double alpha, double beta);

    Pair<Gradient, INDArray> backpropGradient(INDArray input, INDArray epsilon, double k, double n, double alpha,
                    double beta, LayerWorkspaceMgr workspaceMgr);

    INDArray activate(INDArray x, boolean training, double k, double n, double alpha, double beta, LayerWorkspaceMgr workspaceMgr);
}
