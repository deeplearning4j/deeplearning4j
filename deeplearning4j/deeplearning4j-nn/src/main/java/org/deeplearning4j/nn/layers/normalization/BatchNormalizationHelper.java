/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.nn.layers.normalization;

import org.deeplearning4j.nn.conf.CNN2DFormat;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.LayerHelper;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.common.primitives.Pair;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

/**
 * Helper for the batch normalization layer.
 *
 * @author saudet
 */
public interface BatchNormalizationHelper extends LayerHelper {
    boolean checkSupported(double eps, boolean fixedGammaBeta);

    Pair<Gradient, INDArray> backpropGradient(INDArray input, INDArray epsilon, long[] shape, INDArray gamma, INDArray beta,
                                              INDArray dGammaView, INDArray dBetaView, double eps, CNN2DFormat format,
                                              LayerWorkspaceMgr workspaceMgr);

    INDArray preOutput(INDArray x, boolean training, long[] shape, INDArray gamma, INDArray beta, INDArray mean,
                    INDArray var, double decay, double eps, CNN2DFormat format, LayerWorkspaceMgr workspaceMgr);

    INDArray getMeanCache(DataType dataType);

    INDArray getVarCache(DataType dataType);
}
