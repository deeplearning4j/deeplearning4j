/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.deeplearning4j.nn.graph;

import org.deeplearning4j.datasets.iterator.INDArrayDataSetIterator;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.jupiter.api.Test;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Collections;
import java.util.List;

public class TestCompGraphWorkSpaces {
    @Test
    public void testWorkspaces() {

        try {
            ComputationGraphConfiguration computationGraphConf = new NeuralNetConfiguration.Builder()
                    .seed(123)
                    .updater(new Nesterovs(0.1, 0.9))
                    .graphBuilder()
                    .addInputs("input")
                    .appendLayer("L1", new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1}).nIn(1).nOut(1).hasBias(false).build())
                    .appendLayer("out", new CnnLossLayer.Builder()
                            .activation(Activation.SIGMOID)
                            .lossFunction(LossFunctions.LossFunction.XENT)
                            .build())
                    .setOutputs("out")
                    .build();

            ComputationGraph graph = new ComputationGraph(computationGraphConf);

            INDArray data1 = Nd4j.create(1, 1, 256, 256);
            INDArray data2 = Nd4j.create(1, 1, 256, 256);
            INDArray label1 = Nd4j.create(1, 1, 256, 256);
            INDArray label2 = Nd4j.create(1, 1, 256, 256);
            List<Pair<INDArray, INDArray>> trainData = Collections.singletonList(new Pair<>(data1, label1));
            List<Pair<INDArray, INDArray>> testData = Collections.singletonList(new Pair<>(data2, label2));
            DataSetIterator trainIter = new INDArrayDataSetIterator(trainData, 1);
            DataSetIterator testIter = new INDArrayDataSetIterator(testData, 1);

            graph.fit(trainIter);

            while (testIter.hasNext()) {
                graph.score(testIter.next());
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}
