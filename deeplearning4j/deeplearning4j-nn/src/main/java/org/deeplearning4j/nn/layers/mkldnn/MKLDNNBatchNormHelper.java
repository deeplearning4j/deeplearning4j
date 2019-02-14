/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.nn.layers.mkldnn;

import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.normalization.BatchNormalizationHelper;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.BatchNorm;
import org.nd4j.linalg.api.ops.impl.layers.convolution.BatchNormDerivative;
import org.nd4j.linalg.api.ops.impl.summarystats.Variance;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.util.Collections;
import java.util.Map;

public class MKLDNNBatchNormHelper implements BatchNormalizationHelper {
    private static final int[] RANK2_DIMS = {0};
    private static final int[] RANK4_DIMS = {0,2,3};

    private INDArray meanCache;
    private INDArray varCache;

    @Override
    public boolean checkSupported(double eps) {
        return BaseMKLDNNHelper.mklDnnEnabled();
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray input, INDArray epsilon, int[] shape, INDArray gamma,
                                                     INDArray dGammaView, INDArray dBetaView, double eps, LayerWorkspaceMgr workspaceMgr) {
        /*
        INDArray[] in = gamma == null ? new INDArray[]{input, mean, var, epsilon} : new INDArray[]{input, mean, var, gamma, beta, epsilon};

        INDArray gradAtInput = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, input.dataType(), input.shape());

        INDArray[] out = gamma == null ? new INDArray[]{gradAtInput, }

        BatchNormDerivative bn = BatchNormDerivative.derivativeBuilder()
                .applyBeta(gamma != null)
                .applyGamma(gamma != null)
                .axis(new int[]{1})     //4d: is channels: NCHW; 2d: is nIn - axis 1 in both cases
                .epsilon(eps)
                .inputArrays(in)
                .outputArrays(new INDArray[]{out})
                .build();
        Nd4j.exec(bn);
        */

        return null;
    }

    @Override
    public INDArray preOutput(INDArray x, boolean training, int[] shape, INDArray gamma, INDArray beta, INDArray mean, INDArray var,
                              double decay, double eps, LayerWorkspaceMgr workspaceMgr) {

        //Mean and variance: args here are *global*. Depending on train/test mode we might need to use batch mean/var
        INDArray m, v;
        if(training){
            if(meanCache == null){
                try(MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                    meanCache = Nd4j.createUninitialized(x.dataType(), x.size(1));
                    varCache = Nd4j.createUninitialized(x.dataType(), x.size(1));
                }
            }
            x.mean(meanCache, x.rank() == 2 ? RANK2_DIMS : RANK4_DIMS);
            Nd4j.exec(new Variance(x, varCache, false, x.rank() == 2 ? RANK2_DIMS : RANK4_DIMS));

            m = meanCache;
            v = varCache;
        } else {
            m = mean.reshape(mean.length());
            v = var.reshape(var.length());
        }

        //Note: batchnorm op expects rank 1 inputs for mean/var etc, not rank 2 shape [1,x]
        INDArray[] input = gamma == null ? new INDArray[]{x, m, v} : new INDArray[]{x, m, v, gamma.reshape(gamma.length()), beta.reshape(beta.length())};
        INDArray out = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, x.dataType(), x.shape());

        BatchNorm bn = BatchNorm.builder()
                .applyBeta(beta != null)
                .applyGamma(gamma != null)
                .axis(new int[]{1})     //4d: is channels: NCHW; 2d: is nIn - axis 1 in both cases
                .epsilon(eps)
                .inputArrays(input)
                .outputArrays(new INDArray[]{out})
                .build();
        Nd4j.exec(bn);
        return out;
    }

    @Override
    public INDArray getMeanCache() {
        return meanCache;
    }

    @Override
    public INDArray getVarCache() {
        return varCache;
    }

    @Override
    public Map<String, Long> helperMemoryUse() {
        return Collections.emptyMap();
    }
}
