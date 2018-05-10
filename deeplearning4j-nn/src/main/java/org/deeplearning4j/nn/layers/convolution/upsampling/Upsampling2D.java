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

package org.deeplearning4j.nn.layers.convolution.upsampling;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.AbstractLayer;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Upsampling;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.nn.workspace.ArrayType;


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


    public Upsampling2D(NeuralNetConfiguration conf) {
        super(conf);
    }

    public Upsampling2D(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
    }


    @Override
    public double calcL2(boolean backpropParamsOnly) {
        return 0;
    }

    @Override
    public double calcL1(boolean backpropParamsOnly) {
        return 0;
    }

    @Override
    public Type type() {
        return Type.UPSAMPLING;
    }


    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);

        int miniBatch = input.size(0);
        int inDepth = input.size(1);
        int inH = input.size(2);
        int inW = input.size(3);

        INDArray reshapedEpsilon =  workspaceMgr.createUninitialized(ArrayType.ACTIVATION_GRAD, new int[]{miniBatch, inDepth, inH, inW}, 'c');

        Gradient gradient = new DefaultGradient();

        int[] intArgs = new int[] {1}; // 1 is for NCHW


        CustomOp op = DynamicCustomOp.builder("upsampling_bp")
                .addIntegerArguments(intArgs)
                .addInputs(input, epsilon)
                .addOutputs(reshapedEpsilon)
                .callInplace(false)
                .build();
        Nd4j.getExecutioner().exec(op);

        return new Pair<>(gradient, reshapedEpsilon);
    }

    protected int[] getSize(){
        return layerConf().getSize();
    }

    protected INDArray preOutput(boolean training, boolean forBackprop, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(false);
        applyDropOutIfNecessary(training, workspaceMgr);

        if (input.rank() != 4) {
            throw new DL4JInvalidInputException("Got rank " + input.rank()
                    + " array as input to SubsamplingLayer with shape " + Arrays.toString(input.shape())
                    + ". Expected rank 4 array with shape [minibatchSize, channels, inputHeight, inputWidth]. "
                    + layerId());
        }

        if (preOutput != null && forBackprop) {
            return preOutput;
        }

        int miniBatch = input.size(0);
        int inDepth = input.size(1);
        int inH = input.size(2);
        int inW = input.size(3);

        int[] size = getSize();
        int outH = inH * size[0];
        int outW = inW * size[1];

        INDArray reshapedOutput = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, new int[]{miniBatch, inDepth, outH, outW}, 'c');

        int[] intArgs = new int[] {size[0], size[1], 1}; // 1 is for NCHW

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
    public Layer transpose() {
        throw new UnsupportedOperationException(layerId());
    }

    @Override
    public Layer clone() {
        return new Upsampling2D(conf.clone());
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
    public int numParams() {
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
    public void accumulateScore(double accum) {
        throw new UnsupportedOperationException(layerId());
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
