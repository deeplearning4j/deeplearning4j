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

package org.deeplearning4j.nn.layers.convolution.upsampling;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.AbstractLayer;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.primitives.Pair;

import java.util.Arrays;


/**
 * 3D Upsampling layer.
 * <p>
 * Used for upsampling a 3D convolution
 *
 * @author Max Pumperla
 */
@Slf4j
public class Upsampling3D extends AbstractLayer<org.deeplearning4j.nn.conf.layers.Upsampling3D> {


    public Upsampling3D(NeuralNetConfiguration conf, DataType dataType) {
        super(conf, dataType);
    }

    @Override
    public double calcRegularizationScore(boolean backpropParamsOnly){
        return 0;
    }

    @Override
    public Type type() {
        return Type.UPSAMPLING;
    }


    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);

        boolean ncdhw = layerConf().getDataFormat() == org.deeplearning4j.nn.conf.layers.Convolution3D.DataFormat.NCDHW;
        // Assumes NCDHW order
        long miniBatch = input.size(0);
        long inChannels, inD, inH, inW;
        int[] intArgs;
        if(ncdhw){
            inChannels = input.size(1);
            inD = input.size(2);
            inH = input.size(3);
            inW = input.size(4);
            intArgs = new int[] {1}; // 1 is channels first
        } else {
            inD = input.size(1);
            inH = input.size(2);
            inW = input.size(3);
            inChannels = input.size(4);
            intArgs = new int[] {0}; // 0 is channels last
        }



        INDArray epsOut;
        if(ncdhw){
            epsOut = workspaceMgr.createUninitialized(
                    ArrayType.ACTIVATION_GRAD, epsilon.dataType(), new long[]{miniBatch, inChannels, inD, inH, inW}, 'c');
        } else {
            epsOut = workspaceMgr.createUninitialized(
                    ArrayType.ACTIVATION_GRAD, epsilon.dataType(), new long[]{miniBatch, inD, inH, inW, inChannels}, 'c');
        }


        Gradient gradient = new DefaultGradient();

        CustomOp op = DynamicCustomOp.builder("upsampling3d_bp")
                .addIntegerArguments(intArgs)
                .addInputs(input, epsilon)
                .addOutputs(epsOut)
                .callInplace(false)
                .build();
        Nd4j.getExecutioner().exec(op);

        epsOut = backpropDropOutIfPresent(epsOut);
        return new Pair<>(gradient, epsOut);
    }

    protected int[] getSize() {
        return layerConf().getSize();
    }

    protected INDArray preOutput(boolean training, boolean forBackprop, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(false);
        applyDropOutIfNecessary(training, workspaceMgr);

        if (input.rank() != 5) {
            throw new DL4JInvalidInputException("Got rank " + input.rank()
                    + " array as input to Upsampling3DLayer with shape " + Arrays.toString(input.shape())
                    + ". Expected rank 5 array with shape "
                    + "[minibatchSize, channels, inputDepth, inputHeight, inputWidth]. "
                    + layerId());
        }

        if (preOutput != null && forBackprop) {
            return preOutput;
        }

        boolean ncdhw = layerConf().getDataFormat() == org.deeplearning4j.nn.conf.layers.Convolution3D.DataFormat.NCDHW;
        long miniBatch = input.size(0);
        long inChannels, inD, inH, inW;
        int[] intArgs;
        int[] size = getSize();
        if(ncdhw){
            inChannels = (int) input.size(1);
            inD = (int) input.size(2);
            inH = (int) input.size(3);
            inW = (int) input.size(4);
            intArgs = new int[] {size[0], size[1], size[2], 1}; // 1 is channels first
        } else {
            inD = (int) input.size(1);
            inH = (int) input.size(2);
            inW = (int) input.size(3);
            inChannels = (int) input.size(4);
            intArgs = new int[] {size[0], size[1], size[2], 0}; // 0 is channels last
        }


        long outD = inD * size[0];
        long outH = inH * size[1];
        long outW = inW * size[2];

        INDArray output;
        if(ncdhw){
            output = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS,
                    input.dataType(), new long[]{miniBatch, inChannels, outD, outH, outW}, 'c');
        } else {
            output = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS,
                    input.dataType(), new long[]{miniBatch, outD, outH, outW, inChannels}, 'c');
        }



        CustomOp upsampling = DynamicCustomOp.builder("upsampling3d")
                .addIntegerArguments(intArgs)
                .addInputs(input)
                .addOutputs(output)
                .callInplace(false)
                .build();
        Nd4j.getExecutioner().exec(upsampling);

        return output;
    }

    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(false);
        applyDropOutIfNecessary(training, workspaceMgr);

        if (cacheMode == null)
            cacheMode = CacheMode.NONE;

        INDArray z = preOutput(training, false, workspaceMgr);

        // we do cache only if cache workspace exists. Skip otherwise
        if (training && cacheMode != CacheMode.NONE && workspaceMgr.hasConfiguration(ArrayType.FF_CACHE)
                && workspaceMgr.isWorkspaceOpen(ArrayType.FF_CACHE)) {
            try (MemoryWorkspace wsB = workspaceMgr.notifyScopeBorrowed(ArrayType.FF_CACHE)) {
                preOutput = z.unsafeDuplication();
            }
        }
        return z;
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }

    @Override
    public void clearNoiseWeightParams() {
        //No op
    }

    @Override
    public Gradient gradient() {
        throw new UnsupportedOperationException("Not supported - no parameters");
    }

    @Override
    public void fit() {

    }

    @Override
    public long numParams() {
        return 0;
    }

    @Override
    public void fit(INDArray input, LayerWorkspaceMgr workspaceMgr) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public double score() {
        return 0;
    }

    @Override
    public void update(INDArray gradient, String paramType) {

    }

    @Override
    public INDArray params() {
        return null;
    }

    @Override
    public INDArray getParam(String param) {
        return params();
    }

    @Override
    public void setParams(INDArray params) {

    }

}
