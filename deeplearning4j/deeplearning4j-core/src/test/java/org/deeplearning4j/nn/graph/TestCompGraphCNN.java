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

package org.deeplearning4j.nn.graph;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.exception.DL4JInvalidConfigException;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.*;


/**
 * Created by nyghtowl on 1/15/16.
 */
//@Ignore
public class TestCompGraphCNN extends BaseDL4JTest {

    protected ComputationGraphConfiguration conf;
    protected ComputationGraph graph;
    protected DataSetIterator dataSetIterator;
    protected DataSet ds;

    protected static ComputationGraphConfiguration getMultiInputGraphConfig() {
        ComputationGraphConfiguration conf =
                        new NeuralNetConfiguration.Builder()
                                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                                        .graphBuilder().addInputs("input")
                                        .setInputTypes(InputType.convolutional(32, 32, 3))
                                        .addLayer("cnn1",
                                                        new ConvolutionLayer.Builder(4, 4).stride(2, 2).nIn(3).nOut(3)
                                                                        .build(),
                                                        "input")
                                        .addLayer("cnn2",
                                                        new ConvolutionLayer.Builder(4, 4).stride(2, 2).nIn(3).nOut(3)
                                                                        .build(),
                                                        "input")
                                        .addLayer("max1",
                                                        new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                                                                        .stride(1, 1).kernelSize(2, 2).build(),
                                                        "cnn1", "cnn2")
                                        .addLayer("dnn1", new DenseLayer.Builder().nOut(7).build(), "max1")
                                        .addLayer("output", new OutputLayer.Builder().nIn(7).nOut(10).build(), "dnn1")
                                        .setOutputs("output").pretrain(false).backprop(true).build();

        return conf;
    }

    protected static DataSetIterator getDS() {

        List<DataSet> list = new ArrayList<>(5);
        for (int i = 0; i < 5; i++) {
            INDArray f = Nd4j.create(1, 32 * 32 * 3);
            INDArray l = Nd4j.create(1, 10);
            l.putScalar(i, 1.0);
            list.add(new DataSet(f, l));
        }
        return new ListDataSetIterator(list, 5);
    }

    protected static int getNumParams() {
        return 2 * (3 * 1 * 4 * 4 * 3 + 3) + (7 * 14 * 14 * 6 + 7) + (7 * 10 + 10);
    }

    @Before
    @Ignore
    public void beforeDo() {
        conf = getMultiInputGraphConfig();
        graph = new ComputationGraph(conf);
        graph.init();

        dataSetIterator = getDS();
        ds = dataSetIterator.next();

    }

    @Test
    public void testConfigBasic() {
        //Check the order. there are 2 possible valid orders here
        int[] order = graph.topologicalSortOrder();
        int[] expOrder1 = new int[] {0, 1, 2, 4, 3, 5, 6}; //First of 2 possible valid orders
        int[] expOrder2 = new int[] {0, 2, 1, 4, 3, 5, 6}; //Second of 2 possible valid orders
        boolean orderOK = Arrays.equals(expOrder1, order) || Arrays.equals(expOrder2, order);
        assertTrue(orderOK);

        INDArray params = graph.params();
        assertNotNull(params);

        // confirm param shape is what is expected
        int nParams = getNumParams();
        assertEquals(nParams, params.length());

        INDArray arr = Nd4j.linspace(0, nParams, nParams);
        assertEquals(nParams, arr.length());

        // params are set
        graph.setParams(arr);
        params = graph.params();
        assertEquals(arr, params);

        //Number of inputs and outputs:
        assertEquals(1, graph.getNumInputArrays());
        assertEquals(1, graph.getNumOutputArrays());

    }

    @Test(expected = DL4JInvalidConfigException.class)
    public void testCNNComputationGraphKernelTooLarge() {
        int imageWidth = 23;
        int imageHeight = 19;
        int nChannels = 1;
        int classes = 2;
        int numSamples = 200;

        int kernelHeight = 3;
        int kernelWidth = imageWidth;


        DataSet trainInput;

        ComputationGraphConfiguration conf =
                        new NeuralNetConfiguration.Builder()
                                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                                        .seed(123).graphBuilder().addInputs("input")
                                        .setInputTypes(InputType.convolutional(nChannels, imageWidth,
                                                        imageHeight))
                                        .addLayer("conv1", new ConvolutionLayer.Builder()
                                                        .kernelSize(kernelHeight, kernelWidth).stride(1, 1)
                                                        .nIn(nChannels).nOut(2).weightInit(WeightInit.XAVIER)
                                                        .activation(Activation.RELU).build(), "input")
                                        .addLayer("pool1",
                                                        new SubsamplingLayer.Builder()
                                                                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                                                                        .kernelSize(imageHeight - kernelHeight + 1, 1)
                                                                        .stride(1, 1).build(),
                                                        "conv1")
                                        .addLayer("output", new OutputLayer.Builder().nOut(classes).build(), "pool1")
                                        .setOutputs("output").backprop(true).pretrain(false).build();


        ComputationGraph model = new ComputationGraph(conf);
        model.init();


        INDArray emptyFeatures = Nd4j.zeros(numSamples, imageWidth * imageHeight * nChannels);
        INDArray emptyLables = Nd4j.zeros(numSamples, classes);

        trainInput = new DataSet(emptyFeatures, emptyLables);

        model.fit(trainInput);
    }
}
