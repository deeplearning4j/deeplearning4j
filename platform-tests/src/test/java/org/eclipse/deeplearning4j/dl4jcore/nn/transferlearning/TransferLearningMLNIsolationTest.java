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
package org.eclipse.deeplearning4j.dl4jcore.nn.transferlearning;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.api.layers.IOutputLayer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.common.primitives.Pair;
import org.eclipse.deeplearning4j.dl4jcore.TestUtils;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.ResetPolicy;
import org.nd4j.linalg.api.memory.enums.SpillPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

@DisplayName("Transfer Learning MLN Isolation Test")
@Tag(TagNames.SMOKE)
class TransferLearningMLNIsolationTest extends BaseDL4JTest {

    @Test
    @DisplayName("Simple Fine Tune Forward Pass Executes")
    void simpleFineTuneForwardPassExecutes() {
        SimpleFineTuneModels models = simpleFineTuneModels();

        INDArray output = models.modelNow.output(models.randomData.getFeatures(), false);
        assertNotNull(output);
        assertEquals(10, output.size(0));
        assertEquals(3, output.size(1));
    }

    @Test
    @DisplayName("Simple Fine Tune Expected Model Gradient Computation Executes")
    void simpleFineTuneExpectedModelGradientComputationExecutes() {
        SimpleFineTuneModels models = simpleFineTuneModels();

        models.expectedModel.setInput(models.randomData.getFeatures());
        models.expectedModel.setLabels(models.randomData.getLabels());
        models.expectedModel.computeGradientAndScore();

        Gradient gradient = models.expectedModel.gradient();
        assertNotNull(gradient);
        assertNotNull(gradient.gradient());
        assertEquals(models.expectedModel.numParams(), gradient.gradient().length());
    }

    @Test
    @DisplayName("Simple Fresh Model Gradient Computation Executes")
    void simpleFreshModelGradientComputationExecutes() {
        long rng = 12345L;
        Nd4j.getRandom().setSeed(rng);
        DataSet randomData = new DataSet(Nd4j.rand(DataType.FLOAT, 10, 4),
                TestUtils.randomOneHot(DataType.FLOAT, 10, 3));

        MultiLayerNetwork model = freshExpectedModel(rng);
        model.setInput(randomData.getFeatures());
        model.setLabels(randomData.getLabels());
        model.computeGradientAndScore();

        Gradient gradient = model.gradient();
        assertNotNull(gradient);
        assertNotNull(gradient.gradient());
        assertEquals(model.numParams(), gradient.gradient().length());
    }

    @Test
    @DisplayName("Simple Fresh Model Gradient Computation Executes Without Workspaces")
    void simpleFreshModelGradientComputationExecutesWithoutWorkspaces() {
        long rng = 12345L;
        Nd4j.getRandom().setSeed(rng);
        DataSet randomData = new DataSet(Nd4j.rand(DataType.FLOAT, 10, 4),
                TestUtils.randomOneHot(DataType.FLOAT, 10, 3));

        MultiLayerNetwork model = freshExpectedModel(rng, WorkspaceMode.NONE);
        model.setInput(randomData.getFeatures());
        model.setLabels(randomData.getLabels());
        model.computeGradientAndScore();

        Gradient gradient = model.gradient();
        assertNotNull(gradient);
        assertNotNull(gradient.gradient());
        assertEquals(model.numParams(), gradient.gradient().length());
    }

    @Test
    @DisplayName("Output Layer Weight Gradient View GEMM Executes")
    void outputLayerWeightGradientViewGemmExecutes() {
        NeuralNetConfiguration conf = outputLayerConf();
        Map<String, INDArray> gradientViews = gradientViews(conf);

        INDArray input = Nd4j.rand(DataType.FLOAT, 10, 3);
        INDArray delta = Nd4j.rand(DataType.FLOAT, 10, 3);
        INDArray weightGradView = gradientViews.get(DefaultParamInitializer.WEIGHT_KEY);

        Nd4j.gemm(input, delta, weightGradView, true, false, 1.0, 0.0);
        assertEquals(9, weightGradView.length());
    }

    @Test
    @DisplayName("Output Layer Bias Gradient View Sum Executes")
    void outputLayerBiasGradientViewSumExecutes() {
        NeuralNetConfiguration conf = outputLayerConf();
        Map<String, INDArray> gradientViews = gradientViews(conf);

        INDArray delta = Nd4j.rand(DataType.FLOAT, 10, 3);
        INDArray biasGradView = gradientViews.get(DefaultParamInitializer.BIAS_KEY);

        delta.sum(biasGradView, 0);
        assertEquals(3, biasGradView.length());
    }

    @Test
    @DisplayName("Single Output Layer Model Gradient Computation Executes")
    void singleOutputLayerModelGradientComputationExecutes() {
        Nd4j.getRandom().setSeed(12345L);
        DataSet randomData = new DataSet(Nd4j.rand(DataType.FLOAT, 10, 4),
                TestUtils.randomOneHot(DataType.FLOAT, 10, 3));

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345L)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new RmsProp(0.5))
                .list()
                .layer(0, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(4).nOut(3).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setInput(randomData.getFeatures());
        model.setLabels(randomData.getLabels());
        model.computeGradientAndScore();

        Gradient gradient = model.gradient();
        assertNotNull(gradient);
        assertNotNull(gradient.gradient());
        assertEquals(model.numParams(), gradient.gradient().length());
    }

    @Test
    @DisplayName("Dense Layer Backprop Gradient Executes")
    void denseLayerBackpropGradientExecutes() {
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345L)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new RmsProp(0.5))
                .layer(new DenseLayer.Builder().nIn(4).nOut(3).build())
                .build();

        long numParams = conf.getLayer().initializer().numParams(conf);
        INDArray paramsView = Nd4j.create(DataType.FLOAT, numParams);
        Layer layer = conf.getLayer().instantiate(conf, null, 0, paramsView, true, DataType.FLOAT);

        layer.setInput(Nd4j.rand(DataType.FLOAT, 10, 4), LayerWorkspaceMgr.noWorkspaces());
        layer.setBackpropGradientsViewArray(Nd4j.create(DataType.FLOAT, numParams));

        Gradient gradient = layer.backpropGradient(Nd4j.rand(DataType.FLOAT, 10, 3), LayerWorkspaceMgr.noWorkspaces()).getFirst();
        assertNotNull(gradient);
        assertNotNull(gradient.gradientForVariable().get(DefaultParamInitializer.WEIGHT_KEY));
    }

    @Test
    @DisplayName("Manual Output To Dense Backprop Handoff Executes With Workspaces")
    void manualOutputToDenseBackpropHandoffExecutesWithWorkspaces() {
        long rng = 12345L;
        Nd4j.getRandom().setSeed(rng);

        MultiLayerNetwork model = freshExpectedModel(rng, WorkspaceMode.ENABLED);
        model.initGradientsView();
        Layer denseLayer = model.getLayer(0);
        IOutputLayer outputLayer = (IOutputLayer) model.getLayer(1);

        INDArray features = Nd4j.rand(DataType.FLOAT, 10, 4);
        INDArray labels = TestUtils.randomOneHot(DataType.FLOAT, 10, 3);

        WorkspaceConfiguration allActsConfig = allLayersActivationWorkspaceConfig();
        WorkspaceConfiguration workingConfig = layerWorkingWorkspaceConfig();
        WorkspaceConfiguration actGradConfig = layerActivationGradientWorkspaceConfig();

        LayerWorkspaceMgr forwardWorkspaceMgr = LayerWorkspaceMgr.builder()
                .with(ArrayType.INPUT, "WS_TEST_ALL_ACTS", allActsConfig)
                .with(ArrayType.ACTIVATIONS, "WS_TEST_ALL_ACTS", allActsConfig)
                .with(ArrayType.FF_WORKING_MEM, "WS_TEST_WORKING", workingConfig)
                .with(ArrayType.BP_WORKING_MEM, "WS_TEST_WORKING", workingConfig)
                .build();
        LayerWorkspaceMgr oddBackpropMgr = backpropWorkspaceMgr("WS_TEST_ACT_GRAD_2", allActsConfig, actGradConfig, workingConfig);
        LayerWorkspaceMgr evenBackpropMgr = backpropWorkspaceMgr("WS_TEST_ACT_GRAD_1", allActsConfig, actGradConfig, workingConfig);

        try (MemoryWorkspace wsAllActs = Nd4j.getWorkspaceManager().getAndActivateWorkspace(allActsConfig, "WS_TEST_ALL_ACTS")) {
            INDArray denseOutput = denseLayer.activate(features, true, forwardWorkspaceMgr);
            outputLayer.setInput(denseOutput, forwardWorkspaceMgr);
            outputLayer.setLabels(labels);

            Pair<Gradient, INDArray> outputBackprop;
            MemoryWorkspace wsActGrad2 = oddBackpropMgr.notifyScopeEntered(ArrayType.ACTIVATION_GRAD);
            try (MemoryWorkspace wsBpWorking = oddBackpropMgr.notifyScopeEntered(ArrayType.BP_WORKING_MEM)) {
                wsActGrad2.setPreviousWorkspace(wsAllActs);
                wsBpWorking.setPreviousWorkspace(wsAllActs);
                outputBackprop = outputLayer.backpropGradient(null, oddBackpropMgr);
            }

            MemoryWorkspace wsActGrad1 = evenBackpropMgr.notifyScopeEntered(ArrayType.ACTIVATION_GRAD);
            try (MemoryWorkspace wsBpWorking = evenBackpropMgr.notifyScopeEntered(ArrayType.BP_WORKING_MEM)) {
                wsActGrad1.setPreviousWorkspace(wsAllActs);
                wsBpWorking.setPreviousWorkspace(wsAllActs);

                Pair<Gradient, INDArray> denseBackprop = denseLayer.backpropGradient(outputBackprop.getSecond(), evenBackpropMgr);
                assertNotNull(outputBackprop.getSecond());
                assertNotNull(denseBackprop.getFirst());
                assertNotNull(denseBackprop.getFirst().gradientForVariable().get(DefaultParamInitializer.WEIGHT_KEY));
            } finally {
                wsActGrad2.close();
                wsActGrad1.close();
            }
        }
    }

    @Test
    @DisplayName("Manual Dense Backprop Executes With Workspace Managed Input And Epsilon")
    void manualDenseBackpropExecutesWithWorkspaceManagedInputAndEpsilon() {
        long rng = 12345L;
        Nd4j.getRandom().setSeed(rng);

        MultiLayerNetwork model = freshExpectedModel(rng, WorkspaceMode.ENABLED);
        model.initGradientsView();
        Layer denseLayer = model.getLayer(0);

        INDArray features = Nd4j.rand(DataType.FLOAT, 10, 4);

        WorkspaceConfiguration allActsConfig = allLayersActivationWorkspaceConfig();
        WorkspaceConfiguration workingConfig = layerWorkingWorkspaceConfig();
        WorkspaceConfiguration actGradConfig = layerActivationGradientWorkspaceConfig();

        LayerWorkspaceMgr forwardWorkspaceMgr = LayerWorkspaceMgr.builder()
                .with(ArrayType.INPUT, "WS_TEST_ALL_ACTS", allActsConfig)
                .with(ArrayType.ACTIVATIONS, "WS_TEST_ALL_ACTS", allActsConfig)
                .with(ArrayType.FF_WORKING_MEM, "WS_TEST_WORKING", workingConfig)
                .with(ArrayType.BP_WORKING_MEM, "WS_TEST_WORKING", workingConfig)
                .build();
        LayerWorkspaceMgr evenBackpropMgr = backpropWorkspaceMgr("WS_TEST_ACT_GRAD_1", allActsConfig, actGradConfig, workingConfig);

        try (MemoryWorkspace wsAllActs = Nd4j.getWorkspaceManager().getAndActivateWorkspace(allActsConfig, "WS_TEST_ALL_ACTS")) {
            denseLayer.activate(features, true, forwardWorkspaceMgr);

            MemoryWorkspace wsActGrad = evenBackpropMgr.notifyScopeEntered(ArrayType.ACTIVATION_GRAD);
            try (MemoryWorkspace wsBpWorking = evenBackpropMgr.notifyScopeEntered(ArrayType.BP_WORKING_MEM)) {
                wsActGrad.setPreviousWorkspace(wsAllActs);
                wsBpWorking.setPreviousWorkspace(wsAllActs);

                INDArray epsilon = evenBackpropMgr.leverageTo(ArrayType.ACTIVATION_GRAD, Nd4j.rand(DataType.FLOAT, 10, 3));
                Pair<Gradient, INDArray> denseBackprop = denseLayer.backpropGradient(epsilon, evenBackpropMgr);
                assertNotNull(denseBackprop.getFirst());
                assertNotNull(denseBackprop.getFirst().gradientForVariable().get(DefaultParamInitializer.WEIGHT_KEY));
            } finally {
                wsActGrad.close();
            }
        }
    }

    @Test
    @DisplayName("Dense Weight Gradient View GEMM Executes With Workspace Managed Input And Delta")
    void denseWeightGradientViewGemmExecutesWithWorkspaceManagedInputAndDelta() {
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345L)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new RmsProp(0.5))
                .layer(new DenseLayer.Builder().nIn(4).nOut(3).build())
                .build();
        Map<String, INDArray> gradientViews = gradientViews(conf);

        WorkspaceConfiguration allActsConfig = allLayersActivationWorkspaceConfig();
        WorkspaceConfiguration workingConfig = layerWorkingWorkspaceConfig();
        WorkspaceConfiguration actGradConfig = layerActivationGradientWorkspaceConfig();
        LayerWorkspaceMgr forwardWorkspaceMgr = LayerWorkspaceMgr.builder()
                .with(ArrayType.INPUT, "WS_TEST_ALL_ACTS", allActsConfig)
                .with(ArrayType.ACTIVATIONS, "WS_TEST_ALL_ACTS", allActsConfig)
                .with(ArrayType.FF_WORKING_MEM, "WS_TEST_WORKING", workingConfig)
                .with(ArrayType.BP_WORKING_MEM, "WS_TEST_WORKING", workingConfig)
                .build();
        LayerWorkspaceMgr backpropWorkspaceMgr = backpropWorkspaceMgr("WS_TEST_ACT_GRAD_1", allActsConfig, actGradConfig, workingConfig);

        try (MemoryWorkspace wsAllActs = Nd4j.getWorkspaceManager().getAndActivateWorkspace(allActsConfig, "WS_TEST_ALL_ACTS")) {
            INDArray input = forwardWorkspaceMgr.leverageTo(ArrayType.INPUT, Nd4j.rand(DataType.FLOAT, 10, 4));

            MemoryWorkspace wsActGrad = backpropWorkspaceMgr.notifyScopeEntered(ArrayType.ACTIVATION_GRAD);
            try (MemoryWorkspace wsBpWorking = backpropWorkspaceMgr.notifyScopeEntered(ArrayType.BP_WORKING_MEM)) {
                wsActGrad.setPreviousWorkspace(wsAllActs);
                wsBpWorking.setPreviousWorkspace(wsAllActs);

                INDArray delta = backpropWorkspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, Nd4j.rand(DataType.FLOAT, 10, 3));
                INDArray weightGradView = gradientViews.get(DefaultParamInitializer.WEIGHT_KEY);
                Nd4j.gemm(input.castTo(weightGradView.dataType()), delta, weightGradView, true, false, 1.0, 0.0);
                assertEquals(12, weightGradView.length());
            } finally {
                wsActGrad.close();
            }
        }
    }

    @Test
    @DisplayName("Dense Bias Gradient View Sum Executes With Workspace Managed Delta")
    void denseBiasGradientViewSumExecutesWithWorkspaceManagedDelta() {
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345L)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new RmsProp(0.5))
                .layer(new DenseLayer.Builder().nIn(4).nOut(3).build())
                .build();
        Map<String, INDArray> gradientViews = gradientViews(conf);

        WorkspaceConfiguration allActsConfig = allLayersActivationWorkspaceConfig();
        WorkspaceConfiguration workingConfig = layerWorkingWorkspaceConfig();
        WorkspaceConfiguration actGradConfig = layerActivationGradientWorkspaceConfig();
        LayerWorkspaceMgr backpropWorkspaceMgr = backpropWorkspaceMgr("WS_TEST_ACT_GRAD_1", allActsConfig, actGradConfig, workingConfig);

        try (MemoryWorkspace wsAllActs = Nd4j.getWorkspaceManager().getAndActivateWorkspace(allActsConfig, "WS_TEST_ALL_ACTS")) {
            MemoryWorkspace wsActGrad = backpropWorkspaceMgr.notifyScopeEntered(ArrayType.ACTIVATION_GRAD);
            try (MemoryWorkspace wsBpWorking = backpropWorkspaceMgr.notifyScopeEntered(ArrayType.BP_WORKING_MEM)) {
                wsActGrad.setPreviousWorkspace(wsAllActs);
                wsBpWorking.setPreviousWorkspace(wsAllActs);

                INDArray delta = backpropWorkspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, Nd4j.rand(DataType.FLOAT, 10, 3));
                INDArray biasGradView = gradientViews.get(DefaultParamInitializer.BIAS_KEY);
                delta.sum(biasGradView, 0);
                assertEquals(3, biasGradView.length());
            } finally {
                wsActGrad.close();
            }
        }
    }

    @Test
    @DisplayName("Dense PreOutput Executes With Workspace Managed Input")
    void densePreOutputExecutesWithWorkspaceManagedInput() {
        long rng = 12345L;
        Nd4j.getRandom().setSeed(rng);

        MultiLayerNetwork model = freshExpectedModel(rng, WorkspaceMode.ENABLED);
        model.initGradientsView();
        Layer denseLayer = model.getLayer(0);

        INDArray features = Nd4j.rand(DataType.FLOAT, 10, 4);

        WorkspaceConfiguration allActsConfig = allLayersActivationWorkspaceConfig();
        WorkspaceConfiguration workingConfig = layerWorkingWorkspaceConfig();
        WorkspaceConfiguration actGradConfig = layerActivationGradientWorkspaceConfig();

        LayerWorkspaceMgr forwardWorkspaceMgr = LayerWorkspaceMgr.builder()
                .with(ArrayType.INPUT, "WS_TEST_ALL_ACTS", allActsConfig)
                .with(ArrayType.ACTIVATIONS, "WS_TEST_ALL_ACTS", allActsConfig)
                .with(ArrayType.FF_WORKING_MEM, "WS_TEST_WORKING", workingConfig)
                .with(ArrayType.BP_WORKING_MEM, "WS_TEST_WORKING", workingConfig)
                .build();
        LayerWorkspaceMgr backpropWorkspaceMgr = backpropWorkspaceMgr("WS_TEST_ACT_GRAD_1", allActsConfig, actGradConfig, workingConfig);

        try (MemoryWorkspace wsAllActs = Nd4j.getWorkspaceManager().getAndActivateWorkspace(allActsConfig, "WS_TEST_ALL_ACTS")) {
            denseLayer.activate(features, true, forwardWorkspaceMgr);

            MemoryWorkspace wsActGrad = backpropWorkspaceMgr.notifyScopeEntered(ArrayType.ACTIVATION_GRAD);
            try (MemoryWorkspace wsBpWorking = backpropWorkspaceMgr.notifyScopeEntered(ArrayType.BP_WORKING_MEM)) {
                wsActGrad.setPreviousWorkspace(wsAllActs);
                wsBpWorking.setPreviousWorkspace(wsAllActs);

                INDArray input = denseLayer.input().castTo(DataType.FLOAT);
                INDArray weights = denseLayer.getParam(DefaultParamInitializer.WEIGHT_KEY);
                INDArray bias = denseLayer.getParam(DefaultParamInitializer.BIAS_KEY);

                INDArray preOutput = backpropWorkspaceMgr.create(ArrayType.ACTIVATIONS, weights.dataType(),
                        new long[]{input.size(0), weights.size(1)}, 'f');
                input.mmuli(weights, preOutput);
                preOutput.addiRowVector(bias);

                assertEquals(10, preOutput.size(0));
                assertEquals(3, preOutput.size(1));
            } finally {
                wsActGrad.close();
            }
        }
    }

    @Test
    @DisplayName("Dense Activation Backprop Executes With Workspace Managed PreOutput And Epsilon")
    void denseActivationBackpropExecutesWithWorkspaceManagedPreOutputAndEpsilon() {
        long rng = 12345L;
        Nd4j.getRandom().setSeed(rng);

        MultiLayerNetwork model = freshExpectedModel(rng, WorkspaceMode.ENABLED);
        model.initGradientsView();
        Layer denseLayer = model.getLayer(0);

        INDArray features = Nd4j.rand(DataType.FLOAT, 10, 4);

        WorkspaceConfiguration allActsConfig = allLayersActivationWorkspaceConfig();
        WorkspaceConfiguration workingConfig = layerWorkingWorkspaceConfig();
        WorkspaceConfiguration actGradConfig = layerActivationGradientWorkspaceConfig();

        LayerWorkspaceMgr forwardWorkspaceMgr = LayerWorkspaceMgr.builder()
                .with(ArrayType.INPUT, "WS_TEST_ALL_ACTS", allActsConfig)
                .with(ArrayType.ACTIVATIONS, "WS_TEST_ALL_ACTS", allActsConfig)
                .with(ArrayType.FF_WORKING_MEM, "WS_TEST_WORKING", workingConfig)
                .with(ArrayType.BP_WORKING_MEM, "WS_TEST_WORKING", workingConfig)
                .build();
        LayerWorkspaceMgr backpropWorkspaceMgr = backpropWorkspaceMgr("WS_TEST_ACT_GRAD_1", allActsConfig, actGradConfig, workingConfig);

        try (MemoryWorkspace wsAllActs = Nd4j.getWorkspaceManager().getAndActivateWorkspace(allActsConfig, "WS_TEST_ALL_ACTS")) {
            denseLayer.activate(features, true, forwardWorkspaceMgr);

            MemoryWorkspace wsActGrad = backpropWorkspaceMgr.notifyScopeEntered(ArrayType.ACTIVATION_GRAD);
            try (MemoryWorkspace wsBpWorking = backpropWorkspaceMgr.notifyScopeEntered(ArrayType.BP_WORKING_MEM)) {
                wsActGrad.setPreviousWorkspace(wsAllActs);
                wsBpWorking.setPreviousWorkspace(wsAllActs);

                INDArray input = denseLayer.input().castTo(DataType.FLOAT);
                INDArray weights = denseLayer.getParam(DefaultParamInitializer.WEIGHT_KEY);
                INDArray bias = denseLayer.getParam(DefaultParamInitializer.BIAS_KEY);

                INDArray preOutput = backpropWorkspaceMgr.create(ArrayType.ACTIVATIONS, weights.dataType(),
                        new long[]{input.size(0), weights.size(1)}, 'f');
                input.mmuli(weights, preOutput);
                preOutput.addiRowVector(bias);

                INDArray epsilon = backpropWorkspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, Nd4j.rand(DataType.FLOAT, 10, 3));
                INDArray delta = Activation.SIGMOID.getActivationFunction().backprop(preOutput, epsilon).getFirst();

                assertEquals(10, delta.size(0));
                assertEquals(3, delta.size(1));
            } finally {
                wsActGrad.close();
            }
        }
    }

    @Test
    @DisplayName("Simple Fine Tune Transfer Learning Gradient Computation Executes")
    void simpleFineTuneTransferLearningGradientComputationExecutes() {
        SimpleFineTuneModels models = simpleFineTuneModels();

        models.modelNow.setInput(models.randomData.getFeatures());
        models.modelNow.setLabels(models.randomData.getLabels());
        models.modelNow.computeGradientAndScore();

        Gradient gradient = models.modelNow.gradient();
        assertNotNull(gradient);
        assertNotNull(gradient.gradient());
        assertEquals(models.modelNow.numParams(), gradient.gradient().length());
    }

    @Test
    @DisplayName("Simple Fine Tune Updater Initialization Executes")
    void simpleFineTuneUpdaterInitializationExecutes() {
        SimpleFineTuneModels models = simpleFineTuneModels();
        assertNotNull(models.modelNow.getUpdater());
    }

    @Test
    @DisplayName("Simple Fine Tune Fit Executes")
    void simpleFineTuneFitExecutes() {
        SimpleFineTuneModels models = simpleFineTuneModels();

        models.modelNow.fit(models.randomData);
        assertNotNull(models.modelNow.gradient());
    }

    private static SimpleFineTuneModels simpleFineTuneModels() {
        long rng = 12345L;
        Nd4j.getRandom().setSeed(rng);
        DataSet randomData = new DataSet(Nd4j.rand(DataType.FLOAT, 10, 4),
                TestUtils.randomOneHot(DataType.FLOAT, 10, 3));

        NeuralNetConfiguration.Builder confToChange = new NeuralNetConfiguration.Builder()
                .seed(rng)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(0.01, 0.99));
        MultiLayerConfiguration modelToFineTuneConf = confToChange.list()
                .layer(0, new DenseLayer.Builder().nIn(4).nOut(3).build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(3).nOut(3).build())
                .build();

        MultiLayerNetwork modelToFineTune = new MultiLayerNetwork(modelToFineTuneConf);
        modelToFineTune.init();

        MultiLayerNetwork modelNow = new TransferLearning.Builder(modelToFineTune)
                .fineTuneConfiguration(new FineTuneConfiguration.Builder()
                        .seed(rng)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .updater(new RmsProp(0.5))
                        .l2(0.4)
                        .build())
                .build();

        NeuralNetConfiguration.Builder confSet = new NeuralNetConfiguration.Builder()
                .seed(rng)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new RmsProp(0.5))
                .l2(0.4);
        MultiLayerConfiguration expectedConf = confSet.list()
                .layer(0, new DenseLayer.Builder().nIn(4).nOut(3).build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(3).nOut(3).build())
                .build();

        MultiLayerNetwork expectedModel = freshExpectedModel(expectedConf);
        expectedModel.setParams(modelToFineTune.params().dup());

        return new SimpleFineTuneModels(randomData, expectedModel, modelNow);
    }

    private static MultiLayerNetwork freshExpectedModel(long rng) {
        return freshExpectedModel(rng, WorkspaceMode.ENABLED);
    }

    private static MultiLayerNetwork freshExpectedModel(long rng, WorkspaceMode workspaceMode) {
        NeuralNetConfiguration.Builder confSet = new NeuralNetConfiguration.Builder()
                .seed(rng)
                .trainingWorkspaceMode(workspaceMode)
                .inferenceWorkspaceMode(workspaceMode)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new RmsProp(0.5))
                .l2(0.4);
        MultiLayerConfiguration expectedConf = confSet.list()
                .layer(0, new DenseLayer.Builder().nIn(4).nOut(3).build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(3).nOut(3).build())
                .build();
        return freshExpectedModel(expectedConf);
    }

    private static MultiLayerNetwork freshExpectedModel(MultiLayerConfiguration expectedConf) {
        MultiLayerNetwork expectedModel = new MultiLayerNetwork(expectedConf);
        expectedModel.init();
        return expectedModel;
    }

    private static NeuralNetConfiguration outputLayerConf() {
        return new NeuralNetConfiguration.Builder()
                .seed(12345L)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new RmsProp(0.5))
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(3).nOut(3).build())
                .build();
    }

    private static Map<String, INDArray> gradientViews(NeuralNetConfiguration conf) {
        long numParams = conf.getLayer().initializer().numParams(conf);
        INDArray gradientView = Nd4j.create(DataType.FLOAT, numParams);
        return conf.getLayer().initializer().getGradientsFromFlattened(conf, gradientView);
    }

    private static LayerWorkspaceMgr backpropWorkspaceMgr(String activationGradWorkspaceName,
                                                          WorkspaceConfiguration allActsConfig,
                                                          WorkspaceConfiguration actGradConfig,
                                                          WorkspaceConfiguration workingConfig) {
        return LayerWorkspaceMgr.builder()
                .with(ArrayType.INPUT, "WS_TEST_ALL_ACTS", allActsConfig)
                .with(ArrayType.ACTIVATION_GRAD, activationGradWorkspaceName, actGradConfig)
                .with(ArrayType.ACTIVATIONS, "WS_TEST_WORKING", workingConfig)
                .with(ArrayType.FF_WORKING_MEM, "WS_TEST_WORKING", workingConfig)
                .with(ArrayType.BP_WORKING_MEM, "WS_TEST_WORKING", workingConfig)
                .build();
    }

    private static WorkspaceConfiguration allLayersActivationWorkspaceConfig() {
        return WorkspaceConfiguration.builder()
                .initialSize(0)
                .overallocationLimit(0.05)
                .policyLearning(LearningPolicy.FIRST_LOOP)
                .policyReset(ResetPolicy.BLOCK_LEFT)
                .policySpill(SpillPolicy.REALLOCATE)
                .policyAllocation(AllocationPolicy.OVERALLOCATE)
                .build();
    }

    private static WorkspaceConfiguration layerWorkingWorkspaceConfig() {
        return WorkspaceConfiguration.builder()
                .initialSize(0)
                .overallocationLimit(0.02)
                .policyLearning(LearningPolicy.OVER_TIME)
                .cyclesBeforeInitialization(4)
                .policyReset(ResetPolicy.BLOCK_LEFT)
                .policySpill(SpillPolicy.REALLOCATE)
                .policyAllocation(AllocationPolicy.OVERALLOCATE)
                .build();
    }

    private static WorkspaceConfiguration layerActivationGradientWorkspaceConfig() {
        return WorkspaceConfiguration.builder()
                .initialSize(0)
                .overallocationLimit(0.02)
                .policyLearning(LearningPolicy.OVER_TIME)
                .cyclesBeforeInitialization(2)
                .policyReset(ResetPolicy.BLOCK_LEFT)
                .policySpill(SpillPolicy.REALLOCATE)
                .policyAllocation(AllocationPolicy.OVERALLOCATE)
                .build();
    }

    private static final class SimpleFineTuneModels {
        private final DataSet randomData;
        private final MultiLayerNetwork expectedModel;
        private final MultiLayerNetwork modelNow;

        private SimpleFineTuneModels(DataSet randomData, MultiLayerNetwork expectedModel, MultiLayerNetwork modelNow) {
            this.randomData = randomData;
            this.expectedModel = expectedModel;
            this.modelNow = modelNow;
        }
    }
}
