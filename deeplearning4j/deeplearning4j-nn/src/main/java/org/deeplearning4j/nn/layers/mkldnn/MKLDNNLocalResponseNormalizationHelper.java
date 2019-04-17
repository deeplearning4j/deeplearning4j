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

import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.normalization.LocalResponseNormalizationHelper;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.ops.impl.layers.convolution.LocalResponseNormalization;
import org.nd4j.linalg.api.ops.impl.layers.convolution.LocalResponseNormalizationDerivative;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.LocalResponseNormalizationConfig;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.util.Collections;
import java.util.Map;

/**
 * MKL-DNN Local response normalization helper
 *
 * @author Alex Black
 */
public class MKLDNNLocalResponseNormalizationHelper extends BaseMKLDNNHelper implements LocalResponseNormalizationHelper {

    protected OpContext context;

    public MKLDNNLocalResponseNormalizationHelper(DataType dataType){

    }

    @Override
    public boolean checkSupported(double k, double n, double alpha, double beta) {
        return BaseMKLDNNHelper.mklDnnEnabled();
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray input, INDArray epsilon, double k, double n, double alpha, double beta, LayerWorkspaceMgr workspaceMgr) {
        INDArray gradAtInput = workspaceMgr.createUninitialized(ArrayType.ACTIVATION_GRAD, input.dataType(), input.shape());

        if(context == null){
            context = Nd4j.getExecutioner().buildContext();
            context.setTArguments(k, alpha, beta);
            context.setIArguments((int)n);
        }

        LocalResponseNormalization op = new LocalResponseNormalization();

        context.setInputArray(0, input);
        context.setInputArray(0, epsilon);
        context.setOutputArray(0, gradAtInput);

        Nd4j.exec(op, context);
        Gradient g = new DefaultGradient();
        return new Pair<>(g, gradAtInput);
    }

    @Override
    public INDArray activate(INDArray x, boolean training, double k, double n, double alpha, double beta, LayerWorkspaceMgr workspaceMgr) {
        INDArray out = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, x.dataType(), x.shape());

        if(context == null){
            context = Nd4j.getExecutioner().buildContext();
            context.setTArguments(k, alpha, beta);
            context.setIArguments((int)n);
        }

        context.setInputArray(0, x);
        context.setOutputArray(0, out);

        LocalResponseNormalization op = new LocalResponseNormalization();

        Nd4j.exec(op, context);
        return out;
    }

    @Override
    public Map<String, Long> helperMemoryUse() {
        return Collections.emptyMap();
    }
}
