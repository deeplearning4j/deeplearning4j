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
 * Helper for the batch normalization layer.
 *
 * @author saudet
 */
public interface BatchNormalizationHelper {
    boolean checkSupported(double eps);

    Pair<Gradient, INDArray> backpropGradient(INDArray input, INDArray epsilon, int[] shape, INDArray gamma,
                    INDArray dGammaView, INDArray dBetaView, double eps, LayerWorkspaceMgr workspaceMgr);

    INDArray preOutput(INDArray x, boolean training, int[] shape, INDArray gamma, INDArray beta, INDArray mean,
                    INDArray var, double decay, double eps, LayerWorkspaceMgr workspaceMgr);
}
