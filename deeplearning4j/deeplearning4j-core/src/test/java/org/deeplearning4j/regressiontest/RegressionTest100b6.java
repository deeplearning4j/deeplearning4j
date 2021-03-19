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

package org.deeplearning4j.regressiontest;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.TestUtils;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.graph.LayerVertex;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.convolutional.Cropping2D;
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional;
import org.deeplearning4j.nn.conf.layers.recurrent.SimpleRnn;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.impl.MergeVertex;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInitXavier;
import org.deeplearning4j.regressiontest.customlayer100a.CustomLayer;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.activations.impl.*;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.learning.regularization.L2Regularization;
import org.nd4j.linalg.lossfunctions.impl.LossMAE;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;
import org.nd4j.common.resources.Resources;

import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;

import static org.junit.jupiter.api.Assertions.*;
@Disabled
public class RegressionTest100b6 extends BaseDL4JTest {

    @Override
    public DataType getDataType() {
        return DataType.FLOAT;
    }

    @Override
    public long getTimeoutMilliseconds() {
        return 180000L;  //Most tests should be fast, but slow download may cause timeout on slow connections
    }

    @Test
    public void testCustomLayer() throws Exception {

        for (DataType dtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {

            String dtypeName = dtype.toString().toLowerCase();

            File f = Resources.asFile("regression_testing/100b6/CustomLayerExample_100b6_" + dtypeName + ".bin");
            MultiLayerNetwork.load(f, true);

            MultiLayerNetwork net = MultiLayerNetwork.load(f, true);
//            net = net.clone();

            DenseLayer l0 = (DenseLayer) net.getLayer(0).conf().getLayer();
            assertEquals(new ActivationTanH(), l0.getActivationFn());
            assertEquals(new L2Regularization(0.03), TestUtils.getL2Reg(l0));
            assertEquals(new RmsProp(0.95), l0.getIUpdater());

            CustomLayer l1 = (CustomLayer) net.getLayer(1).conf().getLayer();
            assertEquals(new ActivationTanH(), l1.getActivationFn());
            assertEquals(new ActivationSigmoid(), l1.getSecondActivationFunction());
            assertEquals(new RmsProp(0.95), l1.getIUpdater());

            INDArray outExp;
            File f2 = Resources
                    .asFile("regression_testing/100b6/CustomLayerExample_Output_100b6_" + dtypeName + ".bin");
            try (DataInputStream dis = new DataInputStream(new FileInputStream(f2))) {
                outExp = Nd4j.read(dis);
            }

            INDArray in;
            File f3 = Resources.asFile("regression_testing/100b6/CustomLayerExample_Input_100b6_" + dtypeName + ".bin");
            try (DataInputStream dis = new DataInputStream(new FileInputStream(f3))) {
                in = Nd4j.read(dis);
            }

            assertEquals(dtype, in.dataType());
            assertEquals(dtype, outExp.dataType());
            assertEquals(dtype, net.params().dataType());
            assertEquals(dtype, net.getFlattenedGradients().dataType());
            assertEquals(dtype, net.getUpdater().getStateViewArray().dataType());

            //System.out.println(Arrays.toString(net.params().data().asFloat()));

            INDArray outAct = net.output(in);
            assertEquals(dtype, outAct.dataType());

            assertEquals(dtype, net.getLayerWiseConfigurations().getDataType());
            assertEquals(dtype, net.params().dataType());
            boolean eq = outExp.equalsWithEps(outAct, 0.01);
            assertTrue(eq, "Test for dtype: " + dtypeName + " - " + outExp + " vs " + outAct);
        }
    }


    @Test
    public void testLSTM() throws Exception {

        File f = Resources.asFile("regression_testing/100b6/GravesLSTMCharModelingExample_100b6.bin");
        MultiLayerNetwork net = MultiLayerNetwork.load(f, true);

        LSTM l0 = (LSTM) net.getLayer(0).conf().getLayer();
        assertEquals(new ActivationTanH(), l0.getActivationFn());
        assertEquals(200, l0.getNOut());
        assertEquals(new WeightInitXavier(), l0.getWeightInitFn());
        assertEquals(new L2Regularization(0.0001), TestUtils.getL2Reg(l0));
        assertEquals(new Adam(0.005), l0.getIUpdater());

        LSTM l1 = (LSTM) net.getLayer(1).conf().getLayer();
        assertEquals(new ActivationTanH(), l1.getActivationFn());
        assertEquals(200, l1.getNOut());
        assertEquals(new WeightInitXavier(), l1.getWeightInitFn());
        assertEquals(new L2Regularization(0.0001), TestUtils.getL2Reg(l1));
        assertEquals(new Adam(0.005), l1.getIUpdater());

        RnnOutputLayer l2 = (RnnOutputLayer) net.getLayer(2).conf().getLayer();
        assertEquals(new ActivationSoftmax(), l2.getActivationFn());
        assertEquals(77, l2.getNOut());
        assertEquals(new WeightInitXavier(), l2.getWeightInitFn());
        assertEquals(new L2Regularization(0.0001), TestUtils.getL2Reg(l2));
        assertEquals(new Adam(0.005), l2.getIUpdater());

        assertEquals(BackpropType.TruncatedBPTT, net.getLayerWiseConfigurations().getBackpropType());
        assertEquals(50, net.getLayerWiseConfigurations().getTbpttBackLength());
        assertEquals(50, net.getLayerWiseConfigurations().getTbpttFwdLength());

        INDArray outExp;
        File f2 = Resources.asFile("regression_testing/100b6/GravesLSTMCharModelingExample_Output_100b6.bin");
        try (DataInputStream dis = new DataInputStream(new FileInputStream(f2))) {
            outExp = Nd4j.read(dis);
        }

        INDArray in;
        File f3 = Resources.asFile("regression_testing/100b6/GravesLSTMCharModelingExample_Input_100b6.bin");
        try (DataInputStream dis = new DataInputStream(new FileInputStream(f3))) {
            in = Nd4j.read(dis);
        }

        INDArray outAct = net.output(in);

        assertEquals(outExp, outAct);
    }

    @Test
    public void testVae() throws Exception {

        File f = Resources.asFile("regression_testing/100b6/VaeMNISTAnomaly_100b6.bin");
        MultiLayerNetwork net = MultiLayerNetwork.load(f, true);

        VariationalAutoencoder l0 = (VariationalAutoencoder) net.getLayer(0).conf().getLayer();
        assertEquals(new ActivationLReLU(), l0.getActivationFn());
        assertEquals(32, l0.getNOut());
        assertArrayEquals(new int[]{256, 256}, l0.getEncoderLayerSizes());
        assertArrayEquals(new int[]{256, 256}, l0.getDecoderLayerSizes());
        assertEquals(new WeightInitXavier(), l0.getWeightInitFn());
        assertEquals(new L2Regularization(0.0001), TestUtils.getL2Reg(l0));
        assertEquals(new Adam(1e-3), l0.getIUpdater());

        INDArray outExp;
        File f2 = Resources.asFile("regression_testing/100b6/VaeMNISTAnomaly_Output_100b6.bin");
        try (DataInputStream dis = new DataInputStream(new FileInputStream(f2))) {
            outExp = Nd4j.read(dis);
        }

        INDArray in;
        File f3 = Resources.asFile("regression_testing/100b6/VaeMNISTAnomaly_Input_100b6.bin");
        try (DataInputStream dis = new DataInputStream(new FileInputStream(f3))) {
            in = Nd4j.read(dis);
        }

        INDArray outAct = net.output(in);

        assertEquals(outExp, outAct);
    }


    @Test
    public void testYoloHouseNumber() throws Exception {

        File f = Resources.asFile("regression_testing/100b6/HouseNumberDetection_100b6.bin");
        ComputationGraph net = ComputationGraph.load(f, true);

        int nBoxes = 5;
        int nClasses = 10;

        ConvolutionLayer cl = (ConvolutionLayer) ((LayerVertex) net.getConfiguration().getVertices()
                .get("convolution2d_9")).getLayerConf().getLayer();
        assertEquals(nBoxes * (5 + nClasses), cl.getNOut());
        assertEquals(new ActivationIdentity(), cl.getActivationFn());
        assertEquals(ConvolutionMode.Same, cl.getConvolutionMode());
        assertEquals(new WeightInitXavier(), cl.getWeightInitFn());
        assertArrayEquals(new int[]{1, 1}, cl.getKernelSize());

        INDArray outExp;
        File f2 = Resources.asFile("regression_testing/100b6/HouseNumberDetection_Output_100b6.bin");
        try (DataInputStream dis = new DataInputStream(new FileInputStream(f2))) {
            outExp = Nd4j.read(dis);
        }

        INDArray in;
        File f3 = Resources.asFile("regression_testing/100b6/HouseNumberDetection_Input_100b6.bin");
        try (DataInputStream dis = new DataInputStream(new FileInputStream(f3))) {
            in = Nd4j.read(dis);
        }

        INDArray outAct = net.outputSingle(in);

        boolean eq = outExp.equalsWithEps(outAct.castTo(outExp.dataType()), 1e-3);
        assertTrue(eq);
    }

    @Test
    public void testSyntheticCNN() throws Exception {

        File f = Resources.asFile("regression_testing/100b6/SyntheticCNN_100b6.bin");
        MultiLayerNetwork net = MultiLayerNetwork.load(f, true);

        ConvolutionLayer l0 = (ConvolutionLayer) net.getLayer(0).conf().getLayer();
        assertEquals(new ActivationReLU(), l0.getActivationFn());
        assertEquals(4, l0.getNOut());
        assertEquals(new WeightInitXavier(), l0.getWeightInitFn());
        assertEquals(new L2Regularization(0.0001), TestUtils.getL2Reg(l0));
        assertEquals(new Adam(0.005), l0.getIUpdater());
        assertArrayEquals(new int[]{3, 3}, l0.getKernelSize());
        assertArrayEquals(new int[]{2, 1}, l0.getStride());
        assertArrayEquals(new int[]{1, 1}, l0.getDilation());
        assertArrayEquals(new int[]{0, 0}, l0.getPadding());

        SeparableConvolution2D l1 = (SeparableConvolution2D) net.getLayer(1).conf().getLayer();
        assertEquals(new ActivationReLU(), l1.getActivationFn());
        assertEquals(8, l1.getNOut());
        assertEquals(new WeightInitXavier(), l1.getWeightInitFn());
        assertEquals(new L2Regularization(0.0001), TestUtils.getL2Reg(l1));
        assertEquals(new Adam(0.005), l1.getIUpdater());
        assertArrayEquals(new int[]{3, 3}, l1.getKernelSize());
        assertArrayEquals(new int[]{1, 1}, l1.getStride());
        assertArrayEquals(new int[]{1, 1}, l1.getDilation());
        assertArrayEquals(new int[]{0, 0}, l1.getPadding());
        assertEquals(ConvolutionMode.Same, l1.getConvolutionMode());
        assertEquals(1, l1.getDepthMultiplier());

        SubsamplingLayer l2 = (SubsamplingLayer) net.getLayer(2).conf().getLayer();
        assertArrayEquals(new int[]{3, 3}, l2.getKernelSize());
        assertArrayEquals(new int[]{2, 2}, l2.getStride());
        assertArrayEquals(new int[]{1, 1}, l2.getDilation());
        assertArrayEquals(new int[]{0, 0}, l2.getPadding());
        assertEquals(PoolingType.MAX, l2.getPoolingType());

        ZeroPaddingLayer l3 = (ZeroPaddingLayer) net.getLayer(3).conf().getLayer();
        assertArrayEquals(new int[]{4, 4, 4, 4}, l3.getPadding());

        Upsampling2D l4 = (Upsampling2D) net.getLayer(4).conf().getLayer();
        assertArrayEquals(new int[]{3, 3}, l4.getSize());

        DepthwiseConvolution2D l5 = (DepthwiseConvolution2D) net.getLayer(5).conf().getLayer();
        assertEquals(new ActivationReLU(), l5.getActivationFn());
        assertEquals(16, l5.getNOut());
        assertEquals(new WeightInitXavier(), l5.getWeightInitFn());
        assertEquals(new L2Regularization(0.0001), TestUtils.getL2Reg(l5));
        assertEquals(new Adam(0.005), l5.getIUpdater());
        assertArrayEquals(new int[]{3, 3}, l5.getKernelSize());
        assertArrayEquals(new int[]{1, 1}, l5.getStride());
        assertArrayEquals(new int[]{1, 1}, l5.getDilation());
        assertArrayEquals(new int[]{0, 0}, l5.getPadding());
        assertEquals(2, l5.getDepthMultiplier());

        SubsamplingLayer l6 = (SubsamplingLayer) net.getLayer(6).conf().getLayer();
        assertArrayEquals(new int[]{2, 2}, l6.getKernelSize());
        assertArrayEquals(new int[]{2, 2}, l6.getStride());
        assertArrayEquals(new int[]{1, 1}, l6.getDilation());
        assertArrayEquals(new int[]{0, 0}, l6.getPadding());
        assertEquals(PoolingType.MAX, l6.getPoolingType());

        Cropping2D l7 = (Cropping2D) net.getLayer(7).conf().getLayer();
        assertArrayEquals(new int[]{3, 3, 2, 2}, l7.getCropping());

        ConvolutionLayer l8 = (ConvolutionLayer) net.getLayer(8).conf().getLayer();
        assertEquals(4, l8.getNOut());
        assertEquals(new WeightInitXavier(), l8.getWeightInitFn());
        assertEquals(new L2Regularization(0.0001), TestUtils.getL2Reg(l8));
        assertEquals(new Adam(0.005), l8.getIUpdater());
        assertArrayEquals(new int[]{4, 4}, l8.getKernelSize());
        assertArrayEquals(new int[]{1, 1}, l8.getStride());
        assertArrayEquals(new int[]{1, 1}, l8.getDilation());
        assertArrayEquals(new int[]{0, 0}, l8.getPadding());

        CnnLossLayer l9 = (CnnLossLayer) net.getLayer(9).conf().getLayer();
        assertEquals(new WeightInitXavier(), l9.getWeightInitFn());
        assertEquals(new L2Regularization(0.0001), TestUtils.getL2Reg(l9));
        assertEquals(new Adam(0.005), l9.getIUpdater());
        assertEquals(new LossMAE(), l9.getLossFn());

        INDArray outExp;
        File f2 = Resources.asFile("regression_testing/100b6/SyntheticCNN_Output_100b6.bin");
        try (DataInputStream dis = new DataInputStream(new FileInputStream(f2))) {
            outExp = Nd4j.read(dis);
        }

        INDArray in;
        File f3 = Resources.asFile("regression_testing/100b6/SyntheticCNN_Input_100b6.bin");
        try (DataInputStream dis = new DataInputStream(new FileInputStream(f3))) {
            in = Nd4j.read(dis);
        }

        INDArray outAct = net.output(in);

        //19 layers - CPU vs. GPU difference accumulates notably, but appears to be correct
        if(Nd4j.getBackend().getClass().getName().toLowerCase().contains("native")){
            assertEquals(outExp, outAct);
        } else {
            boolean eq = outExp.equalsWithEps(outAct, 0.1);
            assertTrue(eq);
        }
    }

    @Test
    public void testSyntheticBidirectionalRNNGraph() throws Exception {

        File f = Resources.asFile("regression_testing/100b6/SyntheticBidirectionalRNNGraph_100b6.bin");
        ComputationGraph net = ComputationGraph.load(f, true);

        Bidirectional l0 = (Bidirectional) net.getLayer("rnn1").conf().getLayer();

        LSTM l1 = (LSTM) l0.getFwd();
        assertEquals(16, l1.getNOut());
        assertEquals(new ActivationReLU(), l1.getActivationFn());
        assertEquals(new L2Regularization(0.0001), TestUtils.getL2Reg(l1));

        LSTM l2 = (LSTM) l0.getBwd();
        assertEquals(16, l2.getNOut());
        assertEquals(new ActivationReLU(), l2.getActivationFn());
        assertEquals(new L2Regularization(0.0001), TestUtils.getL2Reg(l2));

        Bidirectional l3 = (Bidirectional) net.getLayer("rnn2").conf().getLayer();

        SimpleRnn l4 = (SimpleRnn) l3.getFwd();
        assertEquals(16, l4.getNOut());
        assertEquals(new ActivationReLU(), l4.getActivationFn());
        assertEquals(new L2Regularization(0.0001), TestUtils.getL2Reg(l4));

        SimpleRnn l5 = (SimpleRnn) l3.getBwd();
        assertEquals(16, l5.getNOut());
        assertEquals(new ActivationReLU(), l5.getActivationFn());
        assertEquals(new L2Regularization(0.0001), TestUtils.getL2Reg(l5));

        MergeVertex mv = (MergeVertex) net.getVertex("concat");

        GlobalPoolingLayer gpl = (GlobalPoolingLayer) net.getLayer("pooling").conf().getLayer();
        assertEquals(PoolingType.MAX, gpl.getPoolingType());
        assertArrayEquals(new int[]{2}, gpl.getPoolingDimensions());
        assertTrue(gpl.isCollapseDimensions());

        OutputLayer outl = (OutputLayer) net.getLayer("out").conf().getLayer();
        assertEquals(3, outl.getNOut());
        assertEquals(new LossMCXENT(), outl.getLossFn());

        INDArray outExp;
        File f2 = Resources.asFile("regression_testing/100b6/SyntheticBidirectionalRNNGraph_Output_100b6.bin");
        try (DataInputStream dis = new DataInputStream(new FileInputStream(f2))) {
            outExp = Nd4j.read(dis);
        }

        INDArray in;
        File f3 = Resources.asFile("regression_testing/100b6/SyntheticBidirectionalRNNGraph_Input_100b6.bin");
        try (DataInputStream dis = new DataInputStream(new FileInputStream(f3))) {
            in = Nd4j.read(dis);
        }

        INDArray outAct = net.output(in)[0];

        assertEquals(outExp, outAct);
    }
}
