/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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

package org.deeplearning4j.regressiontest.customlayer100a;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Pair;

/**
 * Layer (implementation) class for the custom layer example
 *
 * @author Alex Black
 */
public class CustomLayerImpl extends BaseLayer<CustomLayer> { //Generic parameter here: the configuration class type

    public CustomLayerImpl(NeuralNetConfiguration conf) {
        super(conf);
    }


    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        /*
        The activate method is used for doing forward pass. Note that it relies on the pre-output method;
        essentially we are just applying the activation function (or, functions in this example).
        In this particular (contrived) example, we have TWO activation functions - one for the first half of the outputs
        and another for the second half.
         */

        INDArray output = preOutput(training, workspaceMgr);
        int columns = output.columns();

        INDArray firstHalf = output.get(NDArrayIndex.all(), NDArrayIndex.interval(0, columns / 2));
        INDArray secondHalf = output.get(NDArrayIndex.all(), NDArrayIndex.interval(columns / 2, columns));

        IActivation activation1 = layerConf().getActivationFn();
        IActivation activation2 = ((CustomLayer) conf.getLayer()).getSecondActivationFunction();

        //IActivation function instances modify the activation functions in-place
        activation1.getActivation(firstHalf, training);
        activation2.getActivation(secondHalf, training);

        return output;
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }


    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        /*
        The baockprop gradient method here is very similar to the BaseLayer backprop gradient implementation
        The only major difference is the two activation functions we have added in this example.

        Note that epsilon is dL/da - i.e., the derivative of the loss function with respect to the activations.
        It has the exact same shape as the activation arrays (i.e., the output of preOut and activate methods)
        This is NOT the 'delta' commonly used in the neural network literature; the delta is obtained from the
        epsilon ("epsilon" is dl4j's notation) by doing an element-wise product with the activation function derivative.

        Note the following:
        1. Is it very important that you use the gradientViews arrays for the results.
           Note the gradientViews.get(...) and the in-place operations here.
           This is because DL4J uses a single large array for the gradients for efficiency. Subsets of this array (views)
           are distributed to each of the layers for efficient backprop and memory management.
        2. The method returns two things, as a Pair:
           (a) a Gradient object (essentially a Map<String,INDArray> of the gradients for each parameter (again, these
               are views of the full network gradient array)
           (b) an INDArray. This INDArray is the 'epsilon' to pass to the layer below. i.e., it is the gradient with
               respect to the input to this layer

        */

        INDArray activationDerivative = preOutput(true, workspaceMgr);
        int columns = activationDerivative.columns();

        INDArray firstHalf = activationDerivative.get(NDArrayIndex.all(), NDArrayIndex.interval(0, columns / 2));
        INDArray secondHalf = activationDerivative.get(NDArrayIndex.all(), NDArrayIndex.interval(columns / 2, columns));

        INDArray epsilonFirstHalf = epsilon.get(NDArrayIndex.all(), NDArrayIndex.interval(0, columns / 2));
        INDArray epsilonSecondHalf = epsilon.get(NDArrayIndex.all(), NDArrayIndex.interval(columns / 2, columns));

        IActivation activation1 = layerConf().getActivationFn();
        IActivation activation2 = ((CustomLayer) conf.getLayer()).getSecondActivationFunction();

        //IActivation backprop method modifies the 'firstHalf' and 'secondHalf' arrays in-place, to contain dL/dz
        activation1.backprop(firstHalf, epsilonFirstHalf);
        activation2.backprop(secondHalf, epsilonSecondHalf);

        //The remaining code for this method: just copy & pasted from BaseLayer.backpropGradient
//        INDArray delta = epsilon.muli(activationDerivative);
        if (maskArray != null) {
            activationDerivative.muliColumnVector(maskArray);
        }

        Gradient ret = new DefaultGradient();

        INDArray weightGrad = gradientViews.get(DefaultParamInitializer.WEIGHT_KEY);    //f order
        Nd4j.gemm(input, activationDerivative, weightGrad, true, false, 1.0, 0.0);
        INDArray biasGrad = gradientViews.get(DefaultParamInitializer.BIAS_KEY);
        biasGrad.assign(activationDerivative.sum(0));  //TODO: do this without the assign

        ret.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY, weightGrad);
        ret.gradientForVariable().put(DefaultParamInitializer.BIAS_KEY, biasGrad);

        INDArray epsilonNext = params.get(DefaultParamInitializer.WEIGHT_KEY).mmul(activationDerivative.transpose()).transpose();

        return new Pair<>(ret, epsilonNext);
    }

}
