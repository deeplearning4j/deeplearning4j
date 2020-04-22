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

import org.deeplearning4j.nn.conf.CNN2DFormat;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.normalization.BatchNormalizationHelper;
import org.deeplearning4j.nn.params.BatchNormalizationParamInitializer;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.ops.impl.layers.convolution.BatchNorm;
import org.nd4j.linalg.api.ops.impl.summarystats.Variance;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * MKL-DNN batch normalization helper implementation
 *
 * @author Alex Black
 */
public class MKLDNNBatchNormHelper implements BatchNormalizationHelper {
    private static final int[] RANK2_DIMS = {0};
    private static final int[] RANK4_DIMS_NCHW = {0,2,3};
    private static final int[] RANK4_DIMS_NHWC = {0,1,2};

    protected OpContext context;
    private INDArray meanCache;
    private INDArray varCache;

    public MKLDNNBatchNormHelper(DataType dataType){

    }

    @Override
    public boolean checkSupported(double eps, boolean fixedGammaBeta) {
        return !fixedGammaBeta && BaseMKLDNNHelper.mklDnnEnabled();
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray input, INDArray epsilon, long[] shape, INDArray gamma,
                                                     INDArray beta, INDArray dGammaView, INDArray dBetaView, double eps,
                                                     CNN2DFormat format, LayerWorkspaceMgr workspaceMgr) {

        //Workaround for: https://github.com/eclipse/deeplearning4j/issues/8860
        if(!Shape.hasDefaultStridesForShape(epsilon))
            epsilon = epsilon.dup('c');

        if(input.dataType() != DataType.FLOAT)
            return null;    //MKL-DNN only supports float

        int axis = (input.rank() != 4 || format == CNN2DFormat.NCHW) ? 1 : 3;

        List<INDArray> args = new ArrayList<>();
        args.add(input);
        args.add(meanCache);
        args.add(varCache);
        if(gamma != null)
            args.add(gamma.reshape(gamma.length()));
        if(beta != null)
            args.add(beta.reshape(beta.length()));
        args.add(epsilon);


        DynamicCustomOp op = DynamicCustomOp.builder("batchnorm_bp")
                .addInputs(args.toArray(new INDArray[0]))
                .addIntegerArguments(
                        gamma == null ? 0 : 1,          //Apply scale
                        beta == null ? 0 : 1,           //Apply beta
                        axis)                              //Axis (NCHW) - 1=NCHW, 3=NHWC
                .addFloatingPointArguments(eps)
                .build();

        INDArray epsAtInput = workspaceMgr.createUninitialized(ArrayType.ACTIVATION_GRAD, input.dataType(), input.shape());
        INDArray dLdm = workspaceMgr.createUninitialized(ArrayType.BP_WORKING_MEM, meanCache.dataType(), meanCache.shape());
        INDArray dLdv = workspaceMgr.createUninitialized(ArrayType.BP_WORKING_MEM, meanCache.dataType(), meanCache.shape());

        op.setOutputArgument(0, epsAtInput);
        op.setOutputArgument(1, dLdm);
        op.setOutputArgument(2, dLdv);
        if(dGammaView != null) {
            //Both are always null/not null simultaneously
            op.setOutputArgument(3, dGammaView.reshape(dGammaView.length()));
            op.setOutputArgument(4, dBetaView.reshape(dBetaView.length()));
        }


        Nd4j.exec(op);

        Gradient g = new DefaultGradient();
        g.setGradientFor(BatchNormalizationParamInitializer.GAMMA, dGammaView);
        g.setGradientFor(BatchNormalizationParamInitializer.BETA, dBetaView);

        return new Pair<>(g, epsAtInput);
    }

    @Override
    public INDArray preOutput(INDArray x, boolean training, long[] shape, INDArray gamma, INDArray beta, INDArray mean, INDArray var,
                              double decay, double eps, CNN2DFormat format, LayerWorkspaceMgr workspaceMgr) {
        if(x.dataType() != DataType.FLOAT)
            return null;    //MKL-DNN only supports float

        int axis = (x.rank() != 4 || format == CNN2DFormat.NCHW) ? 1 : 3;

        if(context == null){
            context = Nd4j.getExecutioner().buildContext();
            context.setIArguments(
                    ArrayUtil.fromBoolean(gamma != null),
                    ArrayUtil.fromBoolean(beta != null),
                    axis);   //Axis - 1 = NCHW, 3 = NHWC
            context.setTArguments(eps);
        }

        //Mean and variance: args here are *global*. Depending on train/test mode we might need to use batch mean/var
        INDArray m, v;
        if(training){
            if(meanCache == null){
                try(MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                    meanCache = Nd4j.createUninitialized(x.dataType(), x.size(axis));
                    varCache = Nd4j.createUninitialized(x.dataType(), x.size(axis));
                }
            }

            int[] dims;
            if(x.rank() == 2){
                dims = RANK2_DIMS;
            } else if(format == CNN2DFormat.NCHW){
                dims = RANK4_DIMS_NCHW;
            } else {
                dims = RANK4_DIMS_NHWC;
            }

            x.mean(meanCache, dims);
            Nd4j.exec(new Variance(x, varCache, false, dims));

            m = meanCache;
            v = varCache;
        } else {
            m = mean.reshape(mean.length());
            v = var.reshape(var.length());
        }

        //Note: batchnorm op expects rank 1 inputs for mean/var etc, not rank 2 shape [1,x]
        context.purge();
        context.setInputArray(0, x);
        context.setInputArray(1, m);
        context.setInputArray(2, v);
        if(gamma != null && beta != null) {
            context.setInputArray(3, gamma.rank() == 2 ? gamma.reshape(gamma.length()) : gamma);
            context.setInputArray(4, beta.rank() == 2 ? beta.reshape(beta.length()) : beta);
        }

        INDArray out = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, x.dataType(), x.shape());
        context.setOutputArray(0, out);

        BatchNorm bn = new BatchNorm();
        Nd4j.exec(bn, context);
        return out;
    }

    @Override
    public INDArray getMeanCache(DataType dataType) {
        return meanCache;
    }

    @Override
    public INDArray getVarCache(DataType dataType) {
        return varCache;
    }

    @Override
    public Map<String, Long> helperMemoryUse() {
        return Collections.emptyMap();
    }
}
