/*
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

package org.deeplearning4j.nn.conf.override;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.OutputLayer;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Classifier over ride
 * @author Adam Gibson
 */
public class ClassifierOverride implements ConfOverride {
    private int finalLayer = -1;

    public ClassifierOverride(int finalLayer) {
        this.finalLayer = finalLayer;
    }

    @Override
    public void overrideLayer(int i, NeuralNetConfiguration.Builder builder) {
        if(i == finalLayer) {
            builder.activationFunction("softmax");
            builder.weightInit(WeightInit.ZERO);
            builder.layerFactory(LayerFactories.getFactory(OutputLayer.class));
            builder.lossFunction(LossFunctions.LossFunction.MCXENT);
        }
    }
}
