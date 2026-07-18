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
package org.eclipse.deeplearning4j.dl4jcore.util;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.EarlyTerminationDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.CrashReportingUtil;
import org.junit.jupiter.api.*;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.reduce.longer.MatchCondition;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import java.io.File;
import static org.junit.jupiter.api.Assertions.*;

import java.nio.file.Path;
import java.util.Map;

@DisplayName("Crash Reporting Util Test")
@NativeTag
@Tag(TagNames.JAVA_ONLY)
@Tag(TagNames.FILE_IO)
@Tag(TagNames.SMOKE)
class CrashReportingUtilTest extends BaseDL4JTest {

    @Override
    public long getTimeoutMilliseconds() {
        return 120000;
    }

    @TempDir
    public Path testDir;

    @Override
    public DataType getDataType() {
        return DataType.FLOAT;
    }

    @AfterEach
    void after() {
        // Reset dir
        CrashReportingUtil.crashDumpOutputDirectory(null);
    }

    private MultiLayerNetwork newTestNetwork(WorkspaceMode workspaceMode) {
        int kernel = 2;
        int stride = 1;
        int padding = 0;
        int inputDepth = 1;
        int height = 28;
        int width = 28;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345L)
                .updater(new Sgd(0.01))
                .weightInit(WeightInit.XAVIER)
                .trainingWorkspaceMode(workspaceMode)
                .inferenceWorkspaceMode(workspaceMode)
                .list()
                .layer(0, new ConvolutionLayer.Builder()
                        .kernelSize(kernel, kernel)
                        .stride(stride, stride)
                        .padding(padding, padding)
                        .nIn(inputDepth)
                        .nOut(3)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(kernel, kernel)
                        .stride(stride, stride)
                        .padding(padding, padding)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nOut(10)
                        .build())
                .setInputType(InputType.convolutionalFlat(height, width, inputDepth))
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.addListeners(new ScoreIterationListener(1));
        return net;
    }

    private DataSetIterator newTrainingIterator() throws Exception {
        return new EarlyTerminationDataSetIterator(new MnistDataSetIterator(32, true, 12345), 5);
    }

    private static void assertNumericallyHealthy(String stage, Map<String, INDArray> arrays) {
        long totalNaN = 0;
        long totalInf = 0;
        StringBuilder details = new StringBuilder();
        for (Map.Entry<String, INDArray> entry : arrays.entrySet()) {
            long nanCount = Nd4j.getExecutioner()
                    .exec(new MatchCondition(entry.getValue(), Conditions.isNan()))
                    .getLong(0);
            long infCount = Nd4j.getExecutioner()
                    .exec(new MatchCondition(entry.getValue(), Conditions.isInfinite()))
                    .getLong(0);
            totalNaN += nanCount;
            totalInf += infCount;
            if (nanCount != 0 || infCount != 0) {
                details.append(entry.getKey())
                        .append("[NaN=").append(nanCount)
                        .append(",Inf=").append(infCount).append("] ");
            }
        }
        assertEquals(0, totalNaN, stage + " contains NaNs: " + details);
        assertEquals(0, totalInf, stage + " contains infinities: " + details);
    }

    private void trainFixtureWithoutCrashDump(WorkspaceMode workspaceMode) throws Exception {
        MultiLayerNetwork net = newTestNetwork(workspaceMode);
        net.fit(newTrainingIterator());
        assertNotNull(net.params());
    }

    @Test
    @DisplayName("First Minibatch Gradients Are Finite Before Update")
    void testFirstMinibatchGradientsAreFiniteBeforeUpdate() throws Exception {
        MultiLayerNetwork net = newTestNetwork(WorkspaceMode.NONE);
        DataSet first = newTrainingIterator().next();
        net.setInput(first.getFeatures());
        net.setLabels(first.getLabels());
        net.computeGradientAndScore();
        assertNumericallyHealthy("first-minibatch gradients", net.gradient().gradientForVariable());
    }

    @Test
    @DisplayName("First Minibatch Parameters Are Finite After Update")
    void testFirstMinibatchParametersAreFiniteAfterUpdate() throws Exception {
        MultiLayerNetwork net = newTestNetwork(WorkspaceMode.NONE);
        DataSet first = newTrainingIterator().next();
        net.fit(first);
        assertNumericallyHealthy("parameters after first update", net.paramTable());
    }

    @Test
    @DisplayName("Training Fixture Without Crash Dump, Workspaces Enabled")
    void testTrainingFixtureWithoutCrashDumpWorkspacesEnabled() throws Exception {
        trainFixtureWithoutCrashDump(WorkspaceMode.ENABLED);
    }

    @Test
    @DisplayName("Training Fixture Without Crash Dump, Workspaces Disabled")
    void testTrainingFixtureWithoutCrashDumpWorkspacesDisabled() throws Exception {
        trainFixtureWithoutCrashDump(WorkspaceMode.NONE);
    }

    @Test
    @DisplayName("Test")
    void test() throws Exception {
        File dir = testDir.toFile();
        CrashReportingUtil.crashDumpOutputDirectory(dir);
        MultiLayerNetwork net = newTestNetwork(WorkspaceMode.ENABLED);
        // Test net that hasn't been trained yet
        Exception e = new Exception();
        CrashReportingUtil.writeMemoryCrashDump(net, e);
        File[] list = dir.listFiles();
        assertNotNull(list);
        assertEquals(1, list.length);
        String str = FileUtils.readFileToString(list[0]);
        // System.out.println(str);
        assertTrue(str.contains("Network Information"));
        assertTrue(str.contains("Network Training Listeners"));
        assertTrue(str.contains("JavaCPP"));
        assertTrue(str.contains("ScoreIterationListener"));
        // Train:
        DataSetIterator iter = newTrainingIterator();
        net.fit(iter);
        dir = testDir.toFile();
        FileUtils.cleanDirectory(dir);
        CrashReportingUtil.crashDumpOutputDirectory(dir);
        CrashReportingUtil.writeMemoryCrashDump(net, e);
        list = dir.listFiles();
        assertNotNull(list);
        assertEquals(1, list.length);
        str = FileUtils.readFileToString(list[0]);
        assertTrue(str.contains("Network Information"));
        assertTrue(str.contains("Network Training Listeners"));
        assertTrue(str.contains("JavaCPP"));
        assertTrue(str.contains("ScoreIterationListener(1)"));
        // System.out.println("///////////////////////////////////////////////////////////");
        // System.out.println(str);
        // System.out.println("///////////////////////////////////////////////////////////");
        // Also test manual memory info
        String mlnMemoryInfo = net.memoryInfo(32, InputType.convolutionalFlat(28, 28, 1));
        // System.out.println("///////////////////////////////////////////////////////////");
        // System.out.println(mlnMemoryInfo);
        // System.out.println("///////////////////////////////////////////////////////////");
        assertTrue(mlnMemoryInfo.contains("Network Information"));
        assertTrue(mlnMemoryInfo.contains("Network Training Listeners"));
        assertTrue(mlnMemoryInfo.contains("JavaCPP"));
        assertTrue(mlnMemoryInfo.contains("ScoreIterationListener(1)"));
        // //////////////////////////////////////
        // Same thing on ComputationGraph:
        dir = testDir.toFile();
        FileUtils.cleanDirectory(dir);
        CrashReportingUtil.crashDumpOutputDirectory(dir);
        ComputationGraph cg = net.toComputationGraph();
        cg.setListeners(new ScoreIterationListener(1));
        // Test net that hasn't been trained yet
        CrashReportingUtil.writeMemoryCrashDump(cg, e);
        list = dir.listFiles();
        assertNotNull(list);
        assertEquals(1, list.length);
        str = FileUtils.readFileToString(list[0]);
        assertTrue(str.contains("Network Information"));
        assertTrue(str.contains("Network Training Listeners"));
        assertTrue(str.contains("JavaCPP"));
        assertTrue(str.contains("ScoreIterationListener(1)"));
        // Train:
        cg.fit(iter);
        dir = testDir.toFile();
        FileUtils.cleanDirectory(dir);
        CrashReportingUtil.crashDumpOutputDirectory(dir);
        CrashReportingUtil.writeMemoryCrashDump(cg, e);
        list = dir.listFiles();
        assertNotNull(list);
        assertEquals(1, list.length);
        str = FileUtils.readFileToString(list[0]);
        assertTrue(str.contains("Network Information"));
        assertTrue(str.contains("Network Training Listeners"));
        assertTrue(str.contains("JavaCPP"));
        assertTrue(str.contains("ScoreIterationListener(1)"));
        // System.out.println("///////////////////////////////////////////////////////////");
        // System.out.println(str);
        // System.out.println("///////////////////////////////////////////////////////////");
        // Also test manual memory info
        String cgMemoryInfo = cg.memoryInfo(32, InputType.convolutionalFlat(28, 28, 1));
        // System.out.println("///////////////////////////////////////////////////////////");
        // System.out.println(cgMemoryInfo);
        // System.out.println("///////////////////////////////////////////////////////////");
        assertTrue(cgMemoryInfo.contains("Network Information"));
        assertTrue(cgMemoryInfo.contains("Network Training Listeners"));
        assertTrue(cgMemoryInfo.contains("JavaCPP"));
        assertTrue(cgMemoryInfo.contains("ScoreIterationListener(1)"));
    }
}
