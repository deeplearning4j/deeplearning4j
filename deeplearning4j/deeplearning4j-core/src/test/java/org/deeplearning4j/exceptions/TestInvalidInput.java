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

package org.deeplearning4j.exceptions;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.exception.DL4JException;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.fail;

/**
 * A set of tests to ensure that useful exceptions are thrown on invalid input
 */
public class TestInvalidInput extends BaseDL4JTest {

    @Test
    public void testInputNinMismatchDense() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().list()
                        .layer(0, new DenseLayer.Builder().nIn(10).nOut(10).build())
                        .layer(1, new OutputLayer.Builder().nIn(10).nOut(10).activation(Activation.SOFTMAX).build()).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        try {
            net.feedForward(Nd4j.create(1, 20));
            fail("Expected DL4JException");
        } catch (DL4JException e) {
            System.out.println("testInputNinMismatchDense(): " + e.getMessage());
        } catch (Exception e) {
            e.printStackTrace();
            fail("Expected DL4JException");
        }
    }

    @Test
    public void testInputNinMismatchOutputLayer() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().list()
                        .layer(0, new DenseLayer.Builder().nIn(10).nOut(20).build())
                        .layer(1, new OutputLayer.Builder().nIn(10).nOut(10).activation(Activation.SOFTMAX).build()).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        try {
            net.feedForward(Nd4j.create(1, 10));
            fail("Expected DL4JException");
        } catch (DL4JException e) {
            System.out.println("testInputNinMismatchOutputLayer(): " + e.getMessage());
        } catch (Exception e) {
            e.printStackTrace();
            fail("Expected DL4JException");
        }
    }

    @Test
    public void testLabelsNOutMismatchOutputLayer() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().list()
                        .layer(0, new DenseLayer.Builder().nIn(10).nOut(10).build())
                        .layer(1, new OutputLayer.Builder().nIn(10).nOut(10).activation(Activation.SOFTMAX).build()).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        try {
            net.fit(Nd4j.create(1, 10), Nd4j.create(1, 20));
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            //From loss function
            System.out.println("testLabelsNOutMismatchOutputLayer(): " + e.getMessage());
        } catch (Exception e) {
            e.printStackTrace();
            fail("Expected DL4JException");
        }
    }

    @Test
    public void testLabelsNOutMismatchRnnOutputLayer() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().list()
                        .layer(0, new GravesLSTM.Builder().nIn(5).nOut(5).build())
                        .layer(1, new RnnOutputLayer.Builder().nIn(5).nOut(5).activation(Activation.SOFTMAX).build()).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        try {
            net.fit(Nd4j.create(1, 5, 8), Nd4j.create(1, 10, 8));
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            //From loss function
            System.out.println("testLabelsNOutMismatchRnnOutputLayer(): " + e.getMessage());
        } catch (Exception e) {
            e.printStackTrace();
            fail("Expected DL4JException");
        }
    }

    @Test
    public void testInputNinMismatchConvolutional() {
        //Rank 4 input, but input channels does not match nIn channels

        int h = 16;
        int w = 16;
        int d = 3;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().list()
                        .layer(0, new ConvolutionLayer.Builder().nIn(d).nOut(5).build())
                        .layer(1, new OutputLayer.Builder().nOut(10).activation(Activation.SOFTMAX).build())
                        .setInputType(InputType.convolutional(h, w, d)).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        try {
            net.feedForward(Nd4j.create(1, 5, h, w));
            fail("Expected DL4JException");
        } catch (DL4JException e) {
            System.out.println("testInputNinMismatchConvolutional(): " + e.getMessage());
        } catch (Exception e) {
            e.printStackTrace();
            fail("Expected DL4JException");
        }
    }

    @Test
    public void testInputNinRank2Convolutional() {
        //Rank 2 input, instead of rank 4 input. For example, forgetting the

        int h = 16;
        int w = 16;
        int d = 3;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().list()
                        .layer(0, new ConvolutionLayer.Builder().nIn(d).nOut(5).build())
                        .layer(1, new OutputLayer.Builder().nOut(10).activation(Activation.SOFTMAX).build())
                        .setInputType(InputType.convolutional(h, w, d)).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        try {
            net.feedForward(Nd4j.create(1, 5 * h * w));
            fail("Expected DL4JException");
        } catch (DL4JException e) {
            System.out.println("testInputNinRank2Convolutional(): " + e.getMessage());
        } catch (Exception e) {
            e.printStackTrace();
            fail("Expected DL4JException");
        }
    }

    @Test
    public void testInputNinRank2Subsampling() {
        //Rank 2 input, instead of rank 4 input. For example, using the wrong input type
        int h = 16;
        int w = 16;
        int d = 3;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().list()
                        .layer(0, new SubsamplingLayer.Builder().kernelSize(2, 2).build())
                        .layer(1, new OutputLayer.Builder().nOut(10).activation(Activation.SOFTMAX).build())
                        .setInputType(InputType.convolutional(h, w, d)).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        try {
            net.feedForward(Nd4j.create(1, 5 * h * w));
            fail("Expected DL4JException");
        } catch (DL4JException e) {
            System.out.println("testInputNinRank2Subsampling(): " + e.getMessage());
        } catch (Exception e) {
            e.printStackTrace();
            fail("Expected DL4JException");
        }
    }


    @Test
    public void testInputNinMismatchLSTM() {

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().list()
                        .layer(0, new GravesLSTM.Builder().nIn(5).nOut(5).build())
                        .layer(1, new RnnOutputLayer.Builder().nIn(5).nOut(5).activation(Activation.SOFTMAX).build()).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        try {
            net.fit(Nd4j.create(1, 10, 5), Nd4j.create(1, 5, 5));
            fail("Expected DL4JException");
        } catch (DL4JException e) {
            System.out.println("testInputNinMismatchLSTM(): " + e.getMessage());
        } catch (Exception e) {
            e.printStackTrace();
            fail("Expected DL4JException");
        }
    }

    @Test
    public void testInputNinMismatchBidirectionalLSTM() {

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().list()
                        .layer(0, new GravesBidirectionalLSTM.Builder().nIn(5).nOut(5).build())
                        .layer(1, new RnnOutputLayer.Builder().nIn(5).nOut(5).activation(Activation.SOFTMAX).build()).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        try {
            net.fit(Nd4j.create(1, 10, 5), Nd4j.create(1, 5, 5));
            fail("Expected DL4JException");
        } catch (DL4JException e) {
            System.out.println("testInputNinMismatchBidirectionalLSTM(): " + e.getMessage());
        } catch (Exception e) {
            e.printStackTrace();
            fail("Expected DL4JException");
        }

    }

    @Test
    public void testInputNinMismatchEmbeddingLayer() {

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().list()
                        .layer(0, new EmbeddingLayer.Builder().nIn(10).nOut(10).build())
                        .layer(1, new OutputLayer.Builder().nIn(10).nOut(10).activation(Activation.SOFTMAX).build()).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        try {
            net.feedForward(Nd4j.create(10, 5));
            fail("Expected DL4JException");
        } catch (DL4JException e) {
            System.out.println("testInputNinMismatchEmbeddingLayer(): " + e.getMessage());
        } catch (Exception e) {
            e.printStackTrace();
            fail("Expected DL4JException");
        }
    }


    @Test
    public void testInvalidRnnTimeStep() {
        //Idea: Using rnnTimeStep with a different number of examples between calls
        //(i.e., not calling reset between time steps)

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().list()
                        .layer(0, new GravesLSTM.Builder().nIn(5).nOut(5).build())
                        .layer(1, new RnnOutputLayer.Builder().nIn(5).nOut(5).activation(Activation.SOFTMAX).build()).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        net.rnnTimeStep(Nd4j.create(3, 5, 10));

        try {
            net.rnnTimeStep(Nd4j.create(5, 5, 10));
            fail("Expected DL4JException");
        } catch (DL4JException e) {
            System.out.println("testInvalidRnnTimeStep(): " + e.getMessage());
        } catch (Exception e) {
            e.printStackTrace();
            fail("Expected DL4JException");
        }
    }
}
