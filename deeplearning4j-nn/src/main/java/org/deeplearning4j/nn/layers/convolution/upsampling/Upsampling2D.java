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
import org.deeplearning4j.nn.api.activations.Activations;
import org.deeplearning4j.nn.api.activations.ActivationsFactory;
import org.deeplearning4j.nn.api.gradients.Gradients;
import org.deeplearning4j.nn.api.gradients.GradientsFactory;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.BaseUpsamplingLayer;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.AbstractLayer;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ops.DynamicCustomOp;


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


    @Override
    public double calcL2(boolean backpropParamsOnly) {
        return 0;
    }

    @Override
    public double calcL1(boolean backpropParamsOnly) {
        return 0;
    }


    @Override
    public Gradients backpropGradient(Gradients gradients) {
        INDArray input = this.input.get(0);
        INDArray epsilon = gradients.get(0);

        int miniBatch = input.size(0);
        int inDepth = input.size(1);
        int inH = input.size(2);
        int inW = input.size(3);

        int size = ((BaseUpsamplingLayer) layerConf()).getSize();   //Required to avoid casting issue in subclasses

        INDArray outEpsilon = Nd4j.createUninitialized(miniBatch * inDepth * inH * inW);
        INDArray reshapedEpsilon = outEpsilon.reshape('c', miniBatch, inDepth, inH, inW);

        INDArray forwardOutput  = preOutput(true, true);

        Gradient gradient = new DefaultGradient();

        CustomOp op = DynamicCustomOp.builder("upsampling_bp")
                .addIntegerArguments(size)
                .addInputs(forwardOutput, epsilon)
                .addOutputs(reshapedEpsilon)
                .callInplace(false)
                .build();
        Nd4j.getExecutioner().exec(op);

        Gradients g = GradientsFactory.getInstance().create(reshapedEpsilon, gradient);
        return backpropPreprocessor(g);
    }

    public INDArray preOutput(boolean training, boolean forBackprop) {
        applyPreprocessorIfNecessary(training);
        applyDropOutIfNecessary(training);
        INDArray input = this.input.get(0);

        if (input.rank() != 4) {
            throw new DL4JInvalidInputException("Got rank " + input.rank()
                    + " array as input to SubsamplingLayer with shape " + Arrays.toString(input.shape())
                    + ". Expected rank 4 array with shape [minibatchSize, depth, inputHeight, inputWidth]. "
                    + layerId());
        }

        if (preOutput != null && forBackprop) {
            return preOutput;
        }

        int miniBatch = input.size(0);
        int inDepth = input.size(1);
        int inH = input.size(2);
        int inW = input.size(3);

        int size = ((BaseUpsamplingLayer) layerConf()).getSize();
        int outH = inH * size;
        int outW = inW * size;

        INDArray output = Nd4j.createUninitialized(miniBatch * inDepth * outH * outW);
        INDArray reshapedOutput = output.reshape('c', miniBatch, inDepth, outH, outW);

        CustomOp op = DynamicCustomOp.builder("upsampling")
                .addIntegerArguments(size)
                .addInputs(input)
                .addOutputs(reshapedOutput)
                .callInplace(false)
                .build();

        Nd4j.getExecutioner().exec(op);

        return reshapedOutput;
    }

    @Override
    public Activations activate(boolean training) {
        applyPreprocessorIfNecessary(training);
        applyDropOutIfNecessary(training);

        if (cacheMode == null)
            cacheMode = CacheMode.NONE;

        INDArray z = preOutput(training, false);

        // we do cache only if cache workspace exists. Skip otherwise
        if (training && cacheMode != CacheMode.NONE
                && Nd4j.getWorkspaceManager().checkIfWorkspaceExists(ComputationGraph.workspaceCache)) {
            try (MemoryWorkspace wsB = Nd4j.getWorkspaceManager()
                    .getWorkspaceForCurrentThread(ComputationGraph.workspaceCache).notifyScopeBorrowed()) {
                preOutput = z.unsafeDuplication();
            }
        }
        return ActivationsFactory.getInstance().create(z);
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
    public int numParams() {
        return 0;
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
