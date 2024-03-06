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
package org.eclipse.deeplearning4j.dl4jcore.nn.layers.recurrent;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.utilty.SingletonMultiDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.saver.InMemoryModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingGraphTrainer;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional;
import org.deeplearning4j.nn.conf.layers.recurrent.SimpleRnn;
import org.deeplearning4j.nn.conf.layers.variational.BernoulliReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.updater.MultiLayerUpdater;
import org.deeplearning4j.nn.updater.graph.ComputationGraphUpdater;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.util.TimeSeriesUtils;
import org.junit.jupiter.api.Tag;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.common.primitives.Pair;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.util.*;
import java.util.stream.Stream;

import static java.util.stream.Collectors.groupingBy;
import static org.deeplearning4j.nn.conf.RNNFormat.NCW;
import static org.deeplearning4j.nn.conf.RNNFormat.NWC;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.DisplayName;
import org.nd4j.linalg.profiler.data.array.event.NDArrayEvent;
import org.nd4j.linalg.profiler.data.array.event.dict.*;
import org.nd4j.linalg.profiler.data.array.eventlog.Nd4jEventLog;

@Slf4j
@DisplayName("Bidirectional Test")
@NativeTag
@Tag(TagNames.DL4J_OLD_API)
class BidirectionalTest extends BaseDL4JTest {


    public static Stream<Arguments> params() {
        List<Arguments> args = new ArrayList<>();
        for (Nd4jBackend nd4jBackend : BaseNd4jTestWithBackends.BACKENDS) {
            for (RNNFormat rnnFormat : new RNNFormat[]{NWC, NCW}) {
                for (WorkspaceMode workspaceMode : new WorkspaceMode[] {WorkspaceMode.ENABLED}) {
                    for (Bidirectional.Mode mode :new Bidirectional.Mode[] { Bidirectional.Mode.CONCAT,Bidirectional.Mode.ADD,Bidirectional.Mode.MUL,
                            Bidirectional.Mode.AVERAGE}) {
                        args.add(Arguments.of(rnnFormat, mode, workspaceMode, nd4jBackend));
                    }
                }
            }
        }
        return args.stream();
    }








    @DisplayName("Test Simple Bidirectional")
    @ParameterizedTest
    @MethodSource("params")
    public void testSimpleBidirectional(RNNFormat rnnDataFormat, Bidirectional.Mode mode, WorkspaceMode workspaceMode, Nd4jBackend backend) {
        log.info("*** Starting workspace mode: " + workspaceMode);
        Nd4j.getRandom().setSeed(12345);

        long[] inshape = rnnDataFormat == NCW ? new long[]{3, 10, 6} : new long[]{3, 6, 10};
        INDArray in1 = Nd4j.linspace(1, 180, 180);
        INDArray in = in1.reshape(inshape).castTo(DataType.DOUBLE);
        MultiLayerConfiguration conf1 = new NeuralNetConfiguration.Builder()
                .dataType(DataType.DOUBLE).activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .trainingWorkspaceMode(workspaceMode).inferenceWorkspaceMode(workspaceMode)
                .updater(new Adam()).list()
                .layer(new Bidirectional(mode, new SimpleRnn.Builder()
                        .nIn(10).nOut(10).dataFormat(rnnDataFormat).build())).build();
        MultiLayerNetwork net1 = new MultiLayerNetwork(conf1);
        net1.init();
        MultiLayerConfiguration conf2 = new NeuralNetConfiguration.Builder()
                .dataType(DataType.DOUBLE).activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam()).list().layer(new SimpleRnn.Builder()
                        .nIn(10).nOut(10)
                        .dataFormat(rnnDataFormat).build()).build();
        MultiLayerNetwork net2 = new MultiLayerNetwork(conf2.clone());
        net2.init();
        MultiLayerNetwork net3 = new MultiLayerNetwork(conf2.clone());
        net3.init();
        net2.setParam("0_W", net1.getParam("0_fW"));
        net2.setParam("0_RW", net1.getParam("0_fRW").dup());
        net2.setParam("0_b", net1.getParam("0_fb").dup());
        //net3 has the same params as net1 but but the backwards layer
        net3.setParam("0_W", net1.getParam("0_bW").dup());
        net3.setParam("0_RW", net1.getParam("0_bRW").dup());
        net3.setParam("0_b", net1.getParam("0_bb").dup());
        assertEquals(net1.getParam("0_fW"), net2.getParam("0_W"));
        assertEquals(net1.getParam("0_fRW"), net2.getParam("0_RW"));
        assertEquals(net1.getParam("0_fb"), net2.getParam("0_b"));
        assertEquals(net1.getParam("0_bW"), net3.getParam("0_W"));
        assertEquals(net1.getParam("0_bRW"), net3.getParam("0_RW"));
        assertEquals(net1.getParam("0_bb"), net3.getParam("0_b"));
        INDArray inReverse = TimeSeriesUtils.reverseTimeSeries(in, LayerWorkspaceMgr.noWorkspaces(), ArrayType.INPUT, rnnDataFormat);
        INDArray out2 = net2.output(in);
        INDArray out3Pre = net3.output(inReverse);
        INDArray out3 = TimeSeriesUtils.reverseTimeSeries(out3Pre, LayerWorkspaceMgr.noWorkspaces(), ArrayType.INPUT, rnnDataFormat);


        INDArray outExp;
        switch (mode) {
            case ADD:
                outExp = out2.add(out3);
                break;
            case MUL:
                outExp = out2.mul(out3);
                break;
            case AVERAGE:
                outExp = out2.add(out3).muli(0.5);
                break;
            case CONCAT:
                outExp = Nd4j.concat(1, out2, out3);
                break;
            default:
                throw new RuntimeException();
        }


        INDArray out1 = net1.output(in);


        assertEquals(outExp, out1, mode.toString());
        // Check gradients:
        if (mode == Bidirectional.Mode.ADD || mode == Bidirectional.Mode.CONCAT) {
            INDArray eps = Nd4j.rand(inshape).castTo(DataType.DOUBLE);
            INDArray eps1;
            //in the bidirectional concat case when creating the epsilon array.
            if (mode == Bidirectional.Mode.CONCAT) {
                eps1 = Nd4j.concat(1, eps, eps);
            } else {
                eps1 = eps.dup();
            }
            net1.setInput(in);
            net2.setInput(in);
            net3.setInput(inReverse);
            //propagate input first even if we don't use the results
            net3.feedForward(false, false);
            net2.feedForward(false, false);
            net1.feedForward(false, false);


            INDArray reverseEps = TimeSeriesUtils.reverseTimeSeries(eps, LayerWorkspaceMgr.noWorkspaces(), ArrayType.INPUT, rnnDataFormat);
            Pair<Gradient, INDArray> p3 = net3.backpropGradient(reverseEps, LayerWorkspaceMgr.noWorkspaces());
            Pair<Gradient, INDArray> p2 = net2.backpropGradient(eps, LayerWorkspaceMgr.noWorkspaces());
            Pair<Gradient, INDArray> p1 = net1.backpropGradient(eps1, LayerWorkspaceMgr.noWorkspaces());


            Gradient g1 = p1.getFirst();
            Gradient g2 = p2.getFirst();
            Gradient g3 = p3.getFirst();

            for (boolean updates : new boolean[]{false, true}) {
                if (updates) {
                    net1.getUpdater().update(net1, g1, 0, 0, 3, LayerWorkspaceMgr.noWorkspaces());
                    net2.getUpdater().update(net2, g2, 0, 0, 3, LayerWorkspaceMgr.noWorkspaces());
                    net3.getUpdater().update(net3, g3, 0, 0, 3, LayerWorkspaceMgr.noWorkspaces());
                }

                assertEquals(g2.gradientForVariable().get("0_W"), g1.gradientForVariable().get("0_fW"));
                assertEquals(g2.gradientForVariable().get("0_RW"), g1.gradientForVariable().get("0_fRW"));
                assertEquals(g2.gradientForVariable().get("0_b"), g1.gradientForVariable().get("0_fb"));
                assertEquals(g3.gradientForVariable().get("0_W"), g1.gradientForVariable().get("0_bW"));
                assertEquals(g3.gradientForVariable().get("0_RW"), g1.gradientForVariable().get("0_bRW"));
                assertEquals(g3.gradientForVariable().get("0_b"), g1.gradientForVariable().get("0_bb"));
            }
        }


    }

    @DisplayName("Test Simple Bidirectional Comp Graph")
    @ParameterizedTest
    @MethodSource("params")
    void testSimpleBidirectionalCompGraph(RNNFormat rnnDataFormat,Bidirectional.Mode mode,WorkspaceMode workspaceMode,Nd4jBackend backend) {
        log.info("*** Starting workspace mode: " + workspaceMode);
        Nd4j.getRandom().setSeed(12345);
        long[] inshape = rnnDataFormat == NCW ? new long[]{3, 10, 6} : new long[]{3, 6, 10};
        INDArray in = Nd4j.rand(inshape).castTo(DataType.DOUBLE);
        ComputationGraphConfiguration conf1 = new NeuralNetConfiguration.Builder().dataType(DataType.DOUBLE)
                .activation(Activation.TANH).weightInit(WeightInit.XAVIER).trainingWorkspaceMode(workspaceMode)
                .inferenceWorkspaceMode(workspaceMode).updater(new Adam()).graphBuilder().addInputs("in")
                .layer("0", new Bidirectional(mode, new SimpleRnn.Builder().nIn(10).nOut(10)
                        .dataFormat(rnnDataFormat).build()), "in").setOutputs("0").build();
        ComputationGraph net1 = new ComputationGraph(conf1);
        net1.init();
        ComputationGraphConfiguration conf2 = new NeuralNetConfiguration.Builder().dataType(DataType.DOUBLE)
                .activation(Activation.TANH).weightInit(WeightInit.XAVIER).updater(new Adam())
                .graphBuilder().addInputs("in")
                .layer("0", new SimpleRnn.Builder().nIn(10).nOut(10).dataFormat(rnnDataFormat).build(), "in")
                .setOutputs("0").build();
        ComputationGraph net2 = new ComputationGraph(conf2.clone());
        net2.init();
        ComputationGraph net3 = new ComputationGraph(conf2.clone());
        net3.init();
        net2.setParam("0_W", net1.getParam("0_fW"));
        net2.setParam("0_RW", net1.getParam("0_fRW"));
        net2.setParam("0_b", net1.getParam("0_fb"));
        net3.setParam("0_W", net1.getParam("0_bW"));
        net3.setParam("0_RW", net1.getParam("0_bRW"));
        net3.setParam("0_b", net1.getParam("0_bb"));
        INDArray out1 = net1.outputSingle(in);
        INDArray out2 = net2.outputSingle(in);
        INDArray out3;
        INDArray inReverse;
        if (rnnDataFormat == NWC) {
            inReverse = TimeSeriesUtils.reverseTimeSeries(in.permute(0, 2, 1), LayerWorkspaceMgr.noWorkspaces(), ArrayType.INPUT).permute(0, 2, 1);
            out3 = net3.outputSingle(inReverse);
            out3 = TimeSeriesUtils.reverseTimeSeries(out3.permute(0, 2, 1), LayerWorkspaceMgr.noWorkspaces(), ArrayType.INPUT).permute(0, 2, 1);
        } else {
            inReverse = TimeSeriesUtils.reverseTimeSeries(in, LayerWorkspaceMgr.noWorkspaces(), ArrayType.INPUT);
            out3 = net3.outputSingle(inReverse);
            out3 = TimeSeriesUtils.reverseTimeSeries(out3, LayerWorkspaceMgr.noWorkspaces(), ArrayType.INPUT);
        }
        INDArray outExp;
        switch (mode) {
            case ADD:
                outExp = out2.add(out3);
                break;
            case MUL:
                outExp = out2.mul(out3);
                break;
            case AVERAGE:
                outExp = out2.add(out3).muli(0.5);
                break;
            case CONCAT:
                outExp = Nd4j.concat(1, out2, out3);
                break;
            default:
                throw new RuntimeException();
        }
        assertEquals(outExp, out1, mode.toString());
        // Check gradients:
        if (mode == Bidirectional.Mode.ADD || mode == Bidirectional.Mode.CONCAT) {
            INDArray eps = Nd4j.rand(inshape).castTo(DataType.DOUBLE);
            INDArray eps1;
            if (mode == Bidirectional.Mode.CONCAT) {
                eps1 = Nd4j.concat(1, eps, eps);
            } else {
                eps1 = eps;
            }
            INDArray epsReversed = (rnnDataFormat == NCW) ? TimeSeriesUtils.reverseTimeSeries(eps, LayerWorkspaceMgr.noWorkspaces(), ArrayType.INPUT) : TimeSeriesUtils.reverseTimeSeries(eps.permute(0, 2, 1), LayerWorkspaceMgr.noWorkspaces(), ArrayType.INPUT).permute(0, 2, 1);
            net1.outputSingle(true, false, in);
            net2.outputSingle(true, false, in);
            net3.outputSingle(true, false, inReverse);
            Gradient g1 = net1.backpropGradient(eps1);
            Gradient g2 = net2.backpropGradient(eps);
            Gradient g3 = net3.backpropGradient(epsReversed);
            for (boolean updates : new boolean[]{false, true}) {
                if (updates) {
                    net1.getUpdater().update(g1, 0, 0, 3, LayerWorkspaceMgr.noWorkspaces());
                    net2.getUpdater().update(g2, 0, 0, 3, LayerWorkspaceMgr.noWorkspaces());
                    net3.getUpdater().update(g3, 0, 0, 3, LayerWorkspaceMgr.noWorkspaces());
                }
                assertEquals(g2.gradientForVariable().get("0_W"), g1.gradientForVariable().get("0_fW"));
                assertEquals(g2.gradientForVariable().get("0_RW"), g1.gradientForVariable().get("0_fRW"));
                assertEquals(g2.gradientForVariable().get("0_b"), g1.gradientForVariable().get("0_fb"));
                assertEquals(g3.gradientForVariable().get("0_W"), g1.gradientForVariable().get("0_bW"));
                assertEquals(g3.gradientForVariable().get("0_RW"), g1.gradientForVariable().get("0_bRW"));
                assertEquals(g3.gradientForVariable().get("0_b"), g1.gradientForVariable().get("0_bb"));

            }
        }

    }

}
