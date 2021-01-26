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
import org.deeplearning4j.nn.conf.CNN2DFormat;
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
 * 2D Upsampling layer.
 * <p>
 * Used for upsampling a 2D convolution
 *
 * @author Max Pumperla
 */
@Slf4j
public class Upsampling2D extends AbstractLayer<org.deeplearning4j.nn.conf.layers.Upsampling2D> {


    public Upsampling2D(NeuralNetConfiguration conf, DataType dataType) {
        super(conf, dataType);
    }

    @Override
    public Type type() {
        return Type.UPSAMPLING;
    }


    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);

        CNN2DFormat format = getFormat();
        boolean nchw = format == CNN2DFormat.NCHW;

        long miniBatch = (int) input.size(0);
        long inDepth = (int) input.size(nchw ? 1 : 3);
        long inH = (int) input.size(nchw ? 2 : 1);
        long inW = (int) input.size(nchw ? 3 : 2);

        long[] epsShape = nchw ? new long[]{miniBatch, inDepth, inH, inW} : new long[]{miniBatch, inH, inW, inDepth};
        INDArray epsOut =  workspaceMgr.createUninitialized(ArrayType.ACTIVATION_GRAD, epsilon.dataType(), epsShape, 'c');

        Gradient gradient = new DefaultGradient();

        CustomOp op = DynamicCustomOp.builder("upsampling_bp")
                .addIntegerArguments(nchw ? 1 : 0)      //1=NCHW, 0=NHWC
                .addInputs(input, epsilon)
                .addOutputs(epsOut)
                .callInplace(false)
                .build();
        Nd4j.getExecutioner().exec(op);

        epsOut = backpropDropOutIfPresent(epsOut);

        return new Pair<>(gradient, epsOut);
    }

    protected int[] getSize(){
        return layerConf().getSize();
    }

    protected CNN2DFormat getFormat(){
        //Here so it can be overridden by Upsampling1D
        return layerConf().getFormat();
    }

    protected INDArray preOutput(boolean training, boolean forBackprop, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(false);
        applyDropOutIfNecessary(training, workspaceMgr);

        if (input.rank() != 4) {
            throw new DL4JInvalidInputException("Got rank " + input.rank()
                    + " array as input to SubsamplingLayer with shape " + Arrays.toString(input.shape())
                    + ". Expected rank 4 array with shape " + layerConf().getFormat().dimensionNames() + ". "
                    + layerId());
        }

        if (preOutput != null && forBackprop) {
            return preOutput;
        }

        CNN2DFormat format = getFormat();
        boolean nchw = format == CNN2DFormat.NCHW;

        long miniBatch = (int) input.size(0);
        long inDepth = (int) input.size(nchw ? 1 : 3);
        long inH = (int) input.size(nchw ? 2 : 1);
        long inW = (int) input.size(nchw ? 3 : 2);

        int[] size = getSize();
        int outH = (int)inH * size[0];
        int outW = (int)inW * size[1];

        long[] outShape = nchw ? new long[]{miniBatch, inDepth, outH, outW} : new long[]{miniBatch, outH, outW, inDepth};
        INDArray reshapedOutput = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, input.dataType(), outShape, 'c');

        int[] intArgs = new int[] {size[0], size[1], nchw ? 1 : 0}; // 1 = NCHW, 0 = NHWC

        CustomOp upsampling = DynamicCustomOp.builder("upsampling2d")
                .addIntegerArguments(intArgs)
                .addInputs(input)
                .addOutputs(reshapedOutput)
                .callInplace(false)
                .build();
        Nd4j.getExecutioner().exec(upsampling);

        return reshapedOutput;
    }

    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(false);
        applyDropOutIfNecessary(training, workspaceMgr);

        if (cacheMode == null)
            cacheMode = CacheMode.NONE;

        INDArray z = preOutput(training, false, workspaceMgr);

        // we do cache only if cache workspace exists. Skip otherwise
        if (training && cacheMode != CacheMode.NONE && workspaceMgr.hasConfiguration(ArrayType.FF_CACHE) && workspaceMgr.isWorkspaceOpen(ArrayType.FF_CACHE)) {
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
