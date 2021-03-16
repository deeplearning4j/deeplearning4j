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

package org.deeplearning4j.nn.layers.recurrent;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.TestUtils;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.RNNFormat;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.LastTimeStep;
import org.deeplearning4j.nn.conf.layers.recurrent.SimpleRnn;
import org.deeplearning4j.nn.conf.layers.util.MaskZeroLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.jupiter.api.Test;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.primitives.Pair;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;

@AllArgsConstructor
public class RnnDataFormatTests extends BaseDL4JTest {


    public static Stream<Arguments> params() {
        List<Object[]> ret = new ArrayList<>();
        for (boolean helpers: new boolean[]{true, false})
            for (boolean lastTimeStep: new boolean[]{true, false})
                for (boolean maskZero: new boolean[]{true, false})
                    ret.add(new Object[]{helpers, lastTimeStep, maskZero});
        return ret.stream().map(Arguments::of);
    }


    @Test
    @MethodSource("#params")
    @ParameterizedTest
    public void testSimpleRnn(boolean helpers,
             boolean lastTimeStep,
             boolean maskZeros
    ) {
        try {

                    Nd4j.getRandom().setSeed(12345);
                    Nd4j.getEnvironment().allowHelpers(helpers);
                    String msg = "Helpers: " + helpers + ", lastTimeStep: " + lastTimeStep + ", maskZeros: " + maskZeros;
                    System.out.println(" --- " + msg + " ---");

                    INDArray inNCW = Nd4j.rand(DataType.FLOAT, 2, 3, 12);

                    INDArray labelsNWC = (lastTimeStep) ?TestUtils.randomOneHot(2, 10): TestUtils.randomOneHot(2 * 12, 10).reshape(2, 12, 10);

                    TestCase tc = TestCase.builder()
                            .msg(msg)
                            .net1(getSimpleRnnNet(RNNFormat.NCW, true, lastTimeStep, maskZeros))
                            .net2(getSimpleRnnNet(RNNFormat.NCW, false, lastTimeStep, maskZeros))
                            .net3(getSimpleRnnNet(RNNFormat.NWC, true, lastTimeStep, maskZeros))
                            .net4(getSimpleRnnNet(RNNFormat.NWC, false, lastTimeStep, maskZeros))
                            .inNCW(inNCW)
                            .labelsNCW((lastTimeStep)? labelsNWC: labelsNWC.permute(0, 2, 1))
                            .labelsNWC(labelsNWC)
                            .testLayerIdx(1)
                            .build();

                    TestCase.testHelper(tc);


        } finally {
            Nd4j.getEnvironment().allowHelpers(true);
        }
    }

    @Test
    @ParameterizedTest
    @MethodSource("#params")
    public void testLSTM(boolean helpers,
                         boolean lastTimeStep,
                         boolean maskZeros) {
        try {

            Nd4j.getRandom().setSeed(12345);
            Nd4j.getEnvironment().allowHelpers(helpers);
            String msg = "Helpers: " + helpers + ", lastTimeStep: " + lastTimeStep + ", maskZeros: " + maskZeros;
            System.out.println(" --- " + msg + " ---");

            INDArray inNCW = Nd4j.rand(DataType.FLOAT, 2, 3, 12);

            INDArray labelsNWC = (lastTimeStep) ?TestUtils.randomOneHot(2, 10): TestUtils.randomOneHot(2 * 12, 10).reshape(2, 12, 10);

            TestCase tc = TestCase.builder()
                    .msg(msg)
                    .net1(getLstmNet(RNNFormat.NCW, true, lastTimeStep, maskZeros))
                    .net2(getLstmNet(RNNFormat.NCW, false, lastTimeStep, maskZeros))
                    .net3(getLstmNet(RNNFormat.NWC, true, lastTimeStep, maskZeros))
                    .net4(getLstmNet(RNNFormat.NWC, false, lastTimeStep, maskZeros))
                    .inNCW(inNCW)
                    .labelsNCW((lastTimeStep)? labelsNWC: labelsNWC.permute(0, 2, 1))
                    .labelsNWC(labelsNWC)
                    .testLayerIdx(1)
                    .build();

            TestCase.testHelper(tc);


        } finally {
            Nd4j.getEnvironment().allowHelpers(true);
        }
    }


    @Test
    @MethodSource("#params")
    @ParameterizedTest
    public void testGraveLSTM(boolean helpers,
                              boolean lastTimeStep,
                              boolean maskZeros) {
        try {

            Nd4j.getRandom().setSeed(12345);
            Nd4j.getEnvironment().allowHelpers(helpers);
            String msg = "Helpers: " + helpers + ", lastTimeStep: " + lastTimeStep + ", maskZeros: " + maskZeros;
            System.out.println(" --- " + msg + " ---");

            INDArray inNCW = Nd4j.rand(DataType.FLOAT, 2, 3, 12);

            INDArray labelsNWC = (lastTimeStep) ?TestUtils.randomOneHot(2, 10): TestUtils.randomOneHot(2 * 12, 10).reshape(2, 12, 10);

            TestCase tc = TestCase.builder()
                    .msg(msg)
                    .net1(getGravesLstmNet(RNNFormat.NCW, true, lastTimeStep, maskZeros))
                    .net2(getGravesLstmNet(RNNFormat.NCW, false, lastTimeStep, maskZeros))
                    .net3(getGravesLstmNet(RNNFormat.NWC, true, lastTimeStep, maskZeros))
                    .net4(getGravesLstmNet(RNNFormat.NWC, false, lastTimeStep, maskZeros))
                    .inNCW(inNCW)
                    .labelsNCW((lastTimeStep)? labelsNWC: labelsNWC.permute(0, 2, 1))
                    .labelsNWC(labelsNWC)
                    .testLayerIdx(1)
                    .build();

            TestCase.testHelper(tc);


        } finally {
            Nd4j.getEnvironment().allowHelpers(true);
        }
    }


    @Test
    @MethodSource("#params")
    @ParameterizedTest
    public void testGraveBiLSTM(boolean helpers,
                                boolean lastTimeStep,
                                boolean maskZeros) {
        try {

            Nd4j.getRandom().setSeed(12345);
            Nd4j.getEnvironment().allowHelpers(helpers);
            String msg = "Helpers: " + helpers + ", lastTimeStep: " + lastTimeStep + ", maskZeros: " + maskZeros;
            System.out.println(" --- " + msg + " ---");

            INDArray inNCW = Nd4j.rand(DataType.FLOAT, 2, 3, 12);

            INDArray labelsNWC = (lastTimeStep) ?TestUtils.randomOneHot(2, 10): TestUtils.randomOneHot(2 * 12, 10).reshape(2, 12, 10);

            TestCase tc = TestCase.builder()
                    .msg(msg)
                    .net1(getGravesBidirectionalLstmNet(RNNFormat.NCW, true, lastTimeStep, maskZeros))
                    .net2(getGravesBidirectionalLstmNet(RNNFormat.NCW, false, lastTimeStep, maskZeros))
                    .net3(getGravesBidirectionalLstmNet(RNNFormat.NWC, true, lastTimeStep, maskZeros))
                    .net4(getGravesBidirectionalLstmNet(RNNFormat.NWC, false, lastTimeStep, maskZeros))
                    .inNCW(inNCW)
                    .labelsNCW((lastTimeStep)? labelsNWC: labelsNWC.permute(0, 2, 1))
                    .labelsNWC(labelsNWC)
                    .testLayerIdx(1)
                    .build();

            TestCase.testHelper(tc);


        } finally {
            Nd4j.getEnvironment().allowHelpers(true);
        }
    }


    private MultiLayerNetwork getGravesBidirectionalLstmNet(RNNFormat format, boolean setOnLayerAlso, boolean lastTimeStep, boolean maskZeros) {
        if (setOnLayerAlso) {
            return getNetWithLayer(new GravesBidirectionalLSTM.Builder().nOut(3)
                    .dataFormat(format).build(), format, lastTimeStep, maskZeros);
        } else {
            return getNetWithLayer(new  GravesBidirectionalLSTM.Builder().nOut(3).build(), format, lastTimeStep, maskZeros);
        }
    }
    private MultiLayerNetwork getGravesLstmNet(RNNFormat format, boolean setOnLayerAlso, boolean lastTimeStep, boolean maskZeros) {
        if (setOnLayerAlso) {
            return getNetWithLayer(new GravesLSTM.Builder().nOut(3)
                    .dataFormat(format).build(), format, lastTimeStep, maskZeros);
        } else {
            return getNetWithLayer(new GravesLSTM.Builder().nOut(3).build(), format, lastTimeStep, maskZeros);
        }
    }

    private MultiLayerNetwork getLstmNet(RNNFormat format, boolean setOnLayerAlso, boolean lastTimeStep, boolean maskZeros) {
        if (setOnLayerAlso) {
            return getNetWithLayer(new LSTM.Builder().nOut(3)
                    .dataFormat(format).build(), format, lastTimeStep, maskZeros);
        } else {
            return getNetWithLayer(new LSTM.Builder().nOut(3).build(), format, lastTimeStep, maskZeros);
        }
    }

    private MultiLayerNetwork getSimpleRnnNet(RNNFormat format, boolean setOnLayerAlso, boolean lastTimeStep, boolean maskZeros) {
        if (setOnLayerAlso) {
            return getNetWithLayer(new SimpleRnn.Builder().nOut(3)
                    .dataFormat(format).build(), format, lastTimeStep, maskZeros);
        } else {
            return getNetWithLayer(new SimpleRnn.Builder().nOut(3).build(), format, lastTimeStep, maskZeros);
        }
    }
    private MultiLayerNetwork getNetWithLayer(Layer layer, RNNFormat format, boolean lastTimeStep, boolean maskZeros) {
        if (maskZeros){
            layer = new MaskZeroLayer.Builder().setMaskValue(0.).setUnderlying(layer).build();
        }
        if(lastTimeStep){
            layer = new LastTimeStep(layer);
        }
        NeuralNetConfiguration.ListBuilder builder = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .list()
                .layer(new LSTM.Builder()
                        .nIn(3)
                        .activation(Activation.TANH)
                        .dataFormat(format)
                        .nOut(3)
                        .helperAllowFallback(false)
                        .build())
                .layer(layer)
                .layer(
                        (lastTimeStep)?new OutputLayer.Builder().activation(Activation.SOFTMAX).nOut(10).build():
        new RnnOutputLayer.Builder().activation(Activation.SOFTMAX).nOut(10).dataFormat(format).build()
                )
                .setInputType(InputType.recurrent(3, 12, format));

        MultiLayerNetwork net = new MultiLayerNetwork(builder.build());
        net.init();
        return net;
    }

    @AllArgsConstructor
    @Data
    @NoArgsConstructor
    @Builder
    private static class TestCase {
        private String msg;
        private MultiLayerNetwork net1;
        private MultiLayerNetwork net2;
        private MultiLayerNetwork net3;
        private MultiLayerNetwork net4;
        private INDArray inNCW;
        private INDArray labelsNCW;
        private INDArray labelsNWC;
        private int testLayerIdx;
        private boolean nwcOutput;

        public static void testHelper(TestCase tc) {

            tc.net2.params().assign(tc.net1.params());
            tc.net3.params().assign(tc.net1.params());
            tc.net4.params().assign(tc.net1.params());

            INDArray inNCW = tc.inNCW;
            INDArray inNWC = tc.inNCW.permute(0, 2, 1).dup();

            INDArray l0_1 = tc.net1.feedForward(inNCW).get(tc.testLayerIdx + 1);
            INDArray l0_2 = tc.net2.feedForward(inNCW).get(tc.testLayerIdx + 1);
            INDArray l0_3 = tc.net3.feedForward(inNWC).get(tc.testLayerIdx + 1);
            INDArray l0_4 = tc.net4.feedForward(inNWC).get(tc.testLayerIdx + 1);

            boolean rank3Out = tc.labelsNCW.rank() == 3;
            assertEquals(l0_1, l0_2, tc.msg);
            if (rank3Out){
                assertEquals(l0_1, l0_3.permute(0, 2, 1), tc.msg);
                assertEquals(l0_1, l0_4.permute(0, 2, 1), tc.msg);
            }
            else{
                assertEquals(l0_1, l0_3, tc.msg);
                assertEquals(l0_1, l0_4, tc.msg);
            }
            INDArray out1 = tc.net1.output(inNCW);
            INDArray out2 = tc.net2.output(inNCW);
            INDArray out3 = tc.net3.output(inNWC);
            INDArray out4 = tc.net4.output(inNWC);

            assertEquals(out1, out2, tc.msg);
            if (rank3Out){
                assertEquals(out1, out3.permute(0, 2, 1), tc.msg);      //NWC to NCW
                assertEquals(out1, out4.permute(0, 2, 1), tc.msg);
            }
            else{
                assertEquals(out1, out3, tc.msg);      //NWC to NCW
                assertEquals(out1, out4, tc.msg);
            }


            //Test backprop
            Pair<Gradient, INDArray> p1 = tc.net1.calculateGradients(inNCW, tc.labelsNCW, null, null);
            Pair<Gradient, INDArray> p2 = tc.net2.calculateGradients(inNCW, tc.labelsNCW, null, null);
            Pair<Gradient, INDArray> p3 = tc.net3.calculateGradients(inNWC, tc.labelsNWC, null, null);
            Pair<Gradient, INDArray> p4 = tc.net4.calculateGradients(inNWC, tc.labelsNWC, null, null);

            //Inpput gradients
            assertEquals(p1.getSecond(), p2.getSecond(), tc.msg);

            assertEquals(p1.getSecond(), p3.getSecond().permute(0, 2, 1), tc.msg);  //Input gradients for NWC input are also in NWC format
            assertEquals(p1.getSecond(), p4.getSecond().permute(0, 2, 1), tc.msg);


            List<String> diff12 = differentGrads(p1.getFirst(), p2.getFirst());
            List<String> diff13 = differentGrads(p1.getFirst(), p3.getFirst());
            List<String> diff14 = differentGrads(p1.getFirst(), p4.getFirst());
            assertEquals(0, diff12.size(),tc.msg + " " + diff12);
            assertEquals(0, diff13.size(),tc.msg + " " + diff13);
            assertEquals( 0, diff14.size(),tc.msg + " " + diff14);

            assertEquals(p1.getFirst().gradientForVariable(), p2.getFirst().gradientForVariable(), tc.msg);
            assertEquals(p1.getFirst().gradientForVariable(), p3.getFirst().gradientForVariable(), tc.msg);
            assertEquals(p1.getFirst().gradientForVariable(), p4.getFirst().gradientForVariable(), tc.msg);

            tc.net1.fit(inNCW, tc.labelsNCW);
            tc.net2.fit(inNCW, tc.labelsNCW);
            tc.net3.fit(inNWC, tc.labelsNWC);
            tc.net4.fit(inNWC, tc.labelsNWC);

            assertEquals(tc.net1.params(), tc.net2.params(), tc.msg);
            assertEquals(tc.net1.params(), tc.net3.params(), tc.msg);
            assertEquals(tc.net1.params(), tc.net4.params(), tc.msg);

            //Test serialization
            MultiLayerNetwork net1a = TestUtils.testModelSerialization(tc.net1);
            MultiLayerNetwork net2a = TestUtils.testModelSerialization(tc.net2);
            MultiLayerNetwork net3a = TestUtils.testModelSerialization(tc.net3);
            MultiLayerNetwork net4a = TestUtils.testModelSerialization(tc.net4);

            out1 = tc.net1.output(inNCW);
            assertEquals(out1, net1a.output(inNCW), tc.msg);
            assertEquals(out1, net2a.output(inNCW), tc.msg);

            if (rank3Out) {
                assertEquals(out1, net3a.output(inNWC).permute(0, 2, 1), tc.msg);   //NWC to NCW
                assertEquals(out1, net4a.output(inNWC).permute(0, 2, 1), tc.msg);
            }
            else{
                assertEquals(out1, net3a.output(inNWC), tc.msg);   //NWC to NCW
                assertEquals(out1, net4a.output(inNWC), tc.msg);
            }
        }

    }
    private static List<String> differentGrads(Gradient g1, Gradient g2){
        List<String> differs = new ArrayList<>();
        Map<String,INDArray> m1 = g1.gradientForVariable();
        Map<String,INDArray> m2 = g2.gradientForVariable();
        for(String s : m1.keySet()){
            INDArray a1 = m1.get(s);
            INDArray a2 = m2.get(s);
            if(!a1.equals(a2)){
                differs.add(s);
            }
        }
        return differs;
    }
}
