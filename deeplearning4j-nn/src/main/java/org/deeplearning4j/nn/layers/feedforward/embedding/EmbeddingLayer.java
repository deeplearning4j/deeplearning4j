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
import org.nd4j.linalg.api.ops.custom.ScatterUpdate;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.nn.workspace.ArrayType;

/**Embedding layer: feed-forward layer that expects single integers per example as input (class numbers, in range 0 to numClass-1)
 * as input. This input has shape [numExamples,1] instead of [numExamples,numClasses] for the equivalent one-hot representation.
 * Mathematically, EmbeddingLayer is equivalent to using a DenseLayer with a one-hot representation for the input; however,
 * it can be much more efficient with a large number of classes (as a dense layer + one-hot input does a matrix multiply
 * with all but one value being zero).<br>
 * <b>Note</b>: can only be used as the first layer for a network<br>
 * <b>Note 2</b>: For a given example index i, the output is activationFunction(weights.getRow(i) + bias), hence the
 * weight rows can be considered a vector/embedding for each example.
 * @author Alex Black
 */
@Slf4j
public class EmbeddingLayer extends BaseLayer<org.deeplearning4j.nn.conf.layers.EmbeddingLayer> {
    private static final int[] DIM_1 = new int[]{1};

    public EmbeddingLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);
        //If this layer is layer L, then epsilon is (w^(L+1)*(d^(L+1))^T) (or equivalent)
        INDArray z = preOutput(true, workspaceMgr);
        INDArray delta = layerConf().getActivationFn().backprop(z, epsilon).getFirst(); //TODO handle activation function params

        if (maskArray != null) {
            delta.muliColumnVector(maskArray);
        }

        INDArray weightGradients = gradientViews.get(DefaultParamInitializer.WEIGHT_KEY);
        weightGradients.assign(0);

        // FIXME: int cast
        int[] indexes = new int[(int) input.length()];
        for (int i = 0; i < indexes.length; i++) {
            indexes[i] = input.getInt(i, 0);
        }

        ScatterUpdate op = new ScatterUpdate(weightGradients, delta, indexes, DIM_1, ScatterUpdate.UpdateOp.ADD);
        Nd4j.getExecutioner().exec(op);

        Gradient ret = new DefaultGradient();
        ret.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY, weightGradients);

        if(hasBias()) {
            INDArray biasGradientsView = gradientViews.get(DefaultParamInitializer.BIAS_KEY);
            delta.sum(biasGradientsView, 0); //biasGradientView is initialized/zeroed first in sum op
            ret.gradientForVariable().put(DefaultParamInitializer.BIAS_KEY, biasGradientsView);
        }

        return new Pair<>(ret, null); //Don't bother returning epsilons: no layer below this one...
    }

    @Override
    protected INDArray preOutput(boolean training, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(false);
        if (input.columns() != 1) {
            //Assume shape is [numExamples,1], and each entry is an integer index
            throw new DL4JInvalidInputException(
                            "Cannot do forward pass for embedding layer with input more than one column. "
                                            + "Expected input shape: [numExamples,1] with each entry being an integer index "
                                            + layerId());
        }

        int nIn = layerConf().getNIn();

        // FIXME: int cast
        int[] indexes = new int[(int) input.length()];
        for (int i = 0; i < indexes.length; i++) {
            indexes[i] = input.getInt(i, 0);

            if (indexes[i] < 0 || indexes[i] >= nIn) {
                throw new DL4JInvalidInputException("Invalid index for embedding layer: got index " + indexes[i]
                        + " for entry " + i + " in minibatch; indexes must be between 0 and nIn-1 inclusive (0 to "
                        + (nIn  -1) + ")");
            }
        }

        INDArray weights = getParam(DefaultParamInitializer.WEIGHT_KEY);
        INDArray bias = getParam(DefaultParamInitializer.BIAS_KEY);

        INDArray destination = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, input.size(0), weights.size(1));
        INDArray rows = Nd4j.pullRows(weights, destination, 1, indexes);
        if(hasBias()){
            rows.addiRowVector(bias);
        }

        return rows;
    }

    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        INDArray rows = preOutput(training, workspaceMgr);

        //INDArray ret =  Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getLayer().getActivationFunction(), rows));
        INDArray ret = layerConf().getActivationFn().getActivation(rows, training);
        if (maskArray != null) {
            ret.muliColumnVector(maskArray);
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

}
