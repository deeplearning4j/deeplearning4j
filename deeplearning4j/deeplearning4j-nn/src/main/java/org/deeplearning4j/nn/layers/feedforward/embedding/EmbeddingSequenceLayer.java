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

package org.deeplearning4j.nn.layers.feedforward.embedding;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.custom.ScatterUpdate;
import org.nd4j.linalg.factory.Broadcast;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.util.Arrays;

import static org.nd4j.linalg.api.shape.Shape.hasDefaultStridesForShape;

/**
 * Embedding layer for sequences: feed-forward layer that expects fixed-length number (inputLength) of integers/indices
 * per example as input, ranged from 0 to numClasses - 1. This input thus has shape [numExamples, inputLength].
 * The output of this layer is 3D, namely of shape [numExamples, nOut, inputLength].
 * <b>Note</b>: can only be used as the first layer for a network<br>
 * <b>Note 2</b>: For a given example index i, the output is activationFunction(weights.getRow(i) + bias), hence the
 * weight rows can be considered a vector/embedding of each index.
 *
 * @author Max Pumperla
 */
@Slf4j
public class EmbeddingSequenceLayer extends BaseLayer<org.deeplearning4j.nn.conf.layers.EmbeddingSequenceLayer> {
    private static final int[] WEIGHT_DIM = new int[]{1};

    public EmbeddingSequenceLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);
        INDArray z = preOutput(true, workspaceMgr);
        INDArray delta = layerConf().getActivationFn().backprop(z, epsilon).getFirst(); //Shape: [mb, vector, seqLength]

        if (maskArray != null) {
            delta = Broadcast.mul(delta, maskArray, delta, 0, 2);
        }

        int inputLength = layerConf().getInputLength();
        int numSamples = input.rows();
        val nOut = layerConf().getNOut();
        delta = delta.permute(2, 0, 1);
        delta = delta.reshape(inputLength * numSamples, nOut);

        INDArray weightGradients = gradientViews.get(DefaultParamInitializer.WEIGHT_KEY);
        weightGradients.assign(0);

        if (!hasDefaultStridesForShape(input))
            input = workspaceMgr.dup(ArrayType.ACTIVATIONS, input, 'f');
        int[] indexes = input.data().asInt();

        ScatterUpdate op = new ScatterUpdate(weightGradients, delta, indexes, WEIGHT_DIM, ScatterUpdate.UpdateOp.ADD);
        Nd4j.getExecutioner().exec(op);

        Gradient ret = new DefaultGradient();
        ret.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY, weightGradients);

        if (hasBias()) {
            INDArray biasGradientsView = gradientViews.get(DefaultParamInitializer.BIAS_KEY);
            delta.sum(biasGradientsView, 0); //biasGradientView is initialized/zeroed first in sum op
            ret.gradientForVariable().put(DefaultParamInitializer.BIAS_KEY, biasGradientsView);
        }

        return new Pair<>(ret, null);
    }

    @Override
    protected INDArray preOutput(boolean training, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(false);

        // if inference is true, override input length config with input data columns
        boolean inferInputLength = layerConf().isInferInputLength();
        if (inferInputLength) {
            layerConf().setInputLength(input.columns());
        }

        if (input.columns() != layerConf().getInputLength()) {
            //Assume shape is [numExamples, inputLength], and each entry is an integer index
            throw new DL4JInvalidInputException("Sequence length of embedding input has to be equal to the specified "
                    + "input length: " + layerConf().getInputLength()
                    + " i.e. we expect input shape [numExamples, inputDim] with each entry being an integer index, "
                    + " got [" + input.rows() + ", " + input.columns() + "] instead, "
                    + "for layer with id: " + layerId());
        }

        val nIn = layerConf().getNIn();
        val numRows = input.rows();
        val inputLength = layerConf().getInputLength();
        if (!hasDefaultStridesForShape(input))
            input = workspaceMgr.dup(ArrayType.ACTIVATIONS, input, 'f');

        int[] indexes = input.data().asInt();

        for (int i = 0; i < indexes.length; i++) {
            indexes[i] = input.getInt(i % numRows, i / numRows);
            if (indexes[i] < 0 || indexes[i] >= nIn) {
                throw new DL4JInvalidInputException("Invalid index for embedding layer: got index " + indexes[i]
                        + " for entry " + i + " in minibatch; indexes must be between 0 and nIn-1 inclusive (0 to "
                        + (nIn - 1) + ")");
            }
        }

        INDArray weights = getParam(DefaultParamInitializer.WEIGHT_KEY);

        val nOut = layerConf().getNOut();
        INDArray destination = workspaceMgr.createUninitialized(
                ArrayType.ACTIVATIONS, numRows * inputLength, nOut);
        INDArray rows = Nd4j.pullRows(weights, destination, 1, indexes);

        if (hasBias()) {
            INDArray bias = getParam(DefaultParamInitializer.BIAS_KEY);
            rows.addiRowVector(bias);
        }


        val shape = new long[]{inputLength, numRows, nOut};
        INDArray ret = workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, rows.reshape('c', shape));
        ret = ret.permute(1, 2 , 0);

        return ret;
    }

    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        INDArray rows = preOutput(training, workspaceMgr);

        INDArray ret = layerConf().getActivationFn().getActivation(rows, training);
        if (maskArray != null) {
            if(maskArray.rank() != 2 || !maskArray.equalShapes(input)){
                throw new IllegalStateException("Mask array for EmbeddingSequenceLayer (when defined) must be rank 2 and" +
                        "have shape equal to input shape. Input shape: " + Arrays.toString(input.shape()) +
                        ", mask shape: " + Arrays.toString(maskArray.shape()));
            }
            //Returned array: rank 3, shape [mb, vector, seqLength]. mask shape: [mb, seqLength]
            Broadcast.mul(ret, maskArray, ret, 0, 2);
//            ret.muliColumnVector(maskArray);
        }
        return ret;
    }

    @Override
    public boolean hasBias() {
        return layerConf().hasBias();
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }

    @Override
    protected void applyDropOutIfNecessary(boolean training, LayerWorkspaceMgr workspaceMgr) {
        throw new UnsupportedOperationException("Dropout not supported with EmbeddingLayer " + layerId());
    }


    @Override
    public Type type() {
        return Type.RECURRENT;
    }
}
