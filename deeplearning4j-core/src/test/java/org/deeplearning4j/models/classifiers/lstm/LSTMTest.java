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

package org.deeplearning4j.models.classifiers.lstm;

import java.util.Arrays;

import org.deeplearning4j.nn.api.LayerFactory;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.stepfunctions.NegativeDefaultStepFunction;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.layers.recurrent.LSTM;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.util.FeatureUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.junit.Test;


/**
 * Created by agibsonccc on 12/30/14.
 */
public class LSTMTest {

    private static final Logger log = LoggerFactory.getLogger(LSTMTest.class);

    @Test
    public void testTraffic() {
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder().activationFunction("tanh")
                .layer(new org.deeplearning4j.nn.conf.layers.LSTM()).optimizationAlgo(OptimizationAlgorithm.LBFGS)
                .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
                .nIn(4).nOut(4).build();
        LSTM l = LayerFactories.getFactory(conf.getLayer()).create(conf, Arrays.<IterationListener>asList(new ScoreIterationListener(10)));
        INDArray predict = FeatureUtil.toOutcomeMatrix(new int[]{0,1,2,3},4);
        l.fit(predict);
        INDArray out = l.activate(predict);
        log.info("Out " + out);
    }

}
