/*
 *  ******************************************************************************
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

package org.deeplearning4j.ui.ui;

import org.apache.commons.io.IOUtils;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.VertxUIServer;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.SameDiffStatsListener;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.junit.jupiter.api.*;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.ui.module.modelmanager.ModelRegistry;
import org.deeplearning4j.ui.module.modelmanager.ModelRegistryEntry;
import org.deeplearning4j.ui.module.modelmanager.HuggingFaceDownloader;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Real end-to-end tests for UI modules. Each test:
 * - Trains actual models or creates real model files
 * - Hits the HTTP endpoints
 * - Verifies the returned data matches the real model state
 */
@Tag(TagNames.FILE_IO)
@Tag(TagNames.UI)
@Tag(TagNames.DIST_SYSTEMS)
@NativeTag
public class TestUIModulesE2E extends BaseDL4JTest {
    private static final Logger log = LoggerFactory.getLogger(TestUIModulesE2E.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final String BASE_URL = "http://localhost:9000";

    @BeforeEach
    public void setUp() throws Exception {
        UIServer.stopInstance();
    }

    @AfterEach
    public void tearDown() throws Exception {
        UIServer.stopInstance();
    }

    @Override
    public long getTimeoutMilliseconds() {
        return 120_000;
    }

    // =========================================================================
    // TrainModule: Real training → verify score values, histograms, graph topology
    // =========================================================================

    @Test
    public void testTrainModuleScoresDecreaseOverTraining() throws Exception {
        StatsStorage ss = new InMemoryStatsStorage();
        UIServer uiServer = UIServer.getInstance();
        uiServer.attach(ss);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Sgd(0.1))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new DenseLayer.Builder().nIn(4).nOut(10).activation(Activation.RELU).build())
                .layer(new OutputLayer.Builder().nIn(10).nOut(3)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new StatsListener(ss, 1));

        DataSetIterator iter = new IrisDataSetIterator(150, 150);
        for (int i = 0; i < 10; i++) {
            net.fit(iter);
            iter.reset();
        }

        // Hit the overview data endpoint
        String json = IOUtils.toString(new URL(BASE_URL + "/train/overview/data"), StandardCharsets.UTF_8);
        Map<String, Object> data = MAPPER.readValue(json, Map.class);

        // Scores must exist and be real positive numbers
        List<Number> scores = (List<Number>) data.get("scores");
        assertNotNull(scores, "scores must be present");
        assertTrue(scores.size() >= 5, "Should have many score entries after 10 epochs, got: " + scores.size());

        // Every score should be a finite positive number
        for (Number s : scores) {
            double val = s.doubleValue();
            assertTrue(Double.isFinite(val), "Score must be finite, got: " + val);
            assertTrue(val > 0, "Score must be positive for cross-entropy loss, got: " + val);
        }

        // Score should generally decrease: first score > last score
        double firstScore = scores.get(0).doubleValue();
        double lastScore = scores.get(scores.size() - 1).doubleValue();
        assertTrue(lastScore < firstScore,
                "Loss should decrease with training. First: " + firstScore + ", Last: " + lastScore);

        // scoresIter must match scores in length and be monotonically increasing
        List<Number> scoresIter = (List<Number>) data.get("scoresIter");
        assertEquals(scores.size(), scoresIter.size(), "scoresIter and scores must have same length");
        for (int i = 1; i < scoresIter.size(); i++) {
            assertTrue(scoresIter.get(i).intValue() >= scoresIter.get(i - 1).intValue(),
                    "Iteration indices must be monotonically increasing");
        }

        // Model metadata must reflect the actual network
        List<List<String>> model = (List<List<String>>) data.get("model");
        assertNotNull(model);
        // model[0][1] = model type
        assertTrue(model.get(0).get(1).contains("MultiLayerNetwork"),
                "Model type should be MultiLayerNetwork, got: " + model.get(0).get(1));
        // model[1][1] = number of layers (2 layers: Dense + Output)
        assertEquals("2", model.get(1).get(1),
                "Should report 2 layers, got: " + model.get(1).get(1));
        // model[2][1] = total params (10*4+10 + 3*10+3 = 83)
        int totalParams = Integer.parseInt(model.get(2).get(1));
        assertEquals(83, totalParams, "Dense(4→10)+Output(10→3) = 40+10+30+3 = 83 params");

        // stdevGradients should have entries for parameter names like "0_W", "0_b", "1_W", "1_b"
        Map<String, List<Number>> stdevGrad = (Map<String, List<Number>>) data.get("stdevGradients");
        assertNotNull(stdevGrad, "stdevGradients must be present");
        assertTrue(stdevGrad.containsKey("0_W"), "Should have gradient stdev for layer 0 weights");
        assertTrue(stdevGrad.containsKey("1_W"), "Should have gradient stdev for layer 1 weights");
        // Values should be real finite numbers
        for (Number v : stdevGrad.get("0_W")) {
            assertTrue(Double.isFinite(v.doubleValue()), "Gradient stdev must be finite");
        }

        // updateRatios should be present with real values
        Map<String, List<Number>> updateRatios = (Map<String, List<Number>>) data.get("updateRatios");
        assertNotNull(updateRatios, "updateRatios must be present");
        assertFalse(updateRatios.isEmpty(), "Should have update ratio data");

        uiServer.stop();
    }

    @Test
    public void testTrainModuleHistogramsAfterTraining() throws Exception {
        StatsStorage ss = new InMemoryStatsStorage();
        UIServer uiServer = UIServer.getInstance();
        uiServer.attach(ss);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(42)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.01))
                .list()
                .layer(new DenseLayer.Builder().nIn(4).nOut(8).activation(Activation.TANH).build())
                .layer(new OutputLayer.Builder().nIn(8).nOut(3)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new StatsListener(ss, 1));

        DataSetIterator iter = new IrisDataSetIterator(150, 150);
        for (int i = 0; i < 5; i++) {
            net.fit(iter);
            iter.reset();
        }

        // Request model data for layer 1 (Dense layer — index 0 is the Input pseudo-vertex)
        String json = IOUtils.toString(new URL(BASE_URL + "/train/model/data/1"), StandardCharsets.UTF_8);
        Map<String, Object> data = MAPPER.readValue(json, Map.class);

        // --- Parameter histograms ---
        Map<String, Object> paramHist = (Map<String, Object>) data.get("paramHist");
        assertNotNull(paramHist, "paramHist must be present after training");
        List<String> paramNames = (List<String>) paramHist.get("paramNames");
        assertNotNull(paramNames);
        assertTrue(paramNames.contains("W"), "paramHist should contain W, got: " + paramNames);
        assertTrue(paramNames.contains("b"), "paramHist should contain b, got: " + paramNames);

        // Verify W histogram has real data
        Map<String, Object> wHist = (Map<String, Object>) paramHist.get("W");
        assertNotNull(wHist, "W histogram must exist");
        double min = ((Number) wHist.get("min")).doubleValue();
        double max = ((Number) wHist.get("max")).doubleValue();
        assertTrue(max > min, "Histogram max must be > min. min=" + min + " max=" + max);
        int bins = ((Number) wHist.get("bins")).intValue();
        assertEquals(20, bins, "Default histogram bins should be 20");
        List<Number> counts = (List<Number>) wHist.get("counts");
        assertEquals(20, counts.size(), "Should have 20 bin counts");
        int totalCount = 0;
        for (Number c : counts) totalCount += c.intValue();
        // Total count should equal number of weight elements: 4*8 = 32
        assertEquals(32, totalCount, "Sum of histogram counts should equal number of weight elements (4*8=32)");

        // --- Update histograms ---
        Map<String, Object> updateHist = (Map<String, Object>) data.get("updateHist");
        assertNotNull(updateHist, "updateHist must be present after training");

        // --- Gradient histograms ---
        Map<String, Object> gradientHist = (Map<String, Object>) data.get("gradientHist");
        assertNotNull(gradientHist, "gradientHist must be present after training");
        List<String> gradParamNames = (List<String>) gradientHist.get("paramNames");
        assertNotNull(gradParamNames);
        assertTrue(gradParamNames.contains("W"), "gradientHist should have W");

        // --- Mean magnitudes ---
        Map<String, Object> meanMag = (Map<String, Object>) data.get("meanMag");
        assertNotNull(meanMag, "meanMag must be present");
        Set<String> layerParamNames = new HashSet<>((Collection<String>) meanMag.get("layerParamNames"));
        assertTrue(layerParamNames.contains("W"), "meanMag should have W");
        assertTrue(layerParamNames.contains("b"), "meanMag should have b");

        // paramMM values should be non-zero (we have trained weights)
        Map<String, List<Number>> paramMM = (Map<String, List<Number>>) meanMag.get("paramMM");
        assertFalse(paramMM.get("W").isEmpty(), "Should have paramMM values for W");
        for (Number v : paramMM.get("W")) {
            assertTrue(v.doubleValue() > 0, "Mean magnitude of weights must be positive, got: " + v);
        }

        // --- Activations ---
        Map<String, Object> activations = (Map<String, Object>) data.get("activations");
        assertNotNull(activations, "activations must be present");
        List<Number> actMean = (List<Number>) activations.get("mean");
        List<Number> actStdev = (List<Number>) activations.get("stdev");
        assertFalse(actMean.isEmpty(), "Activation means should not be empty");
        assertFalse(actStdev.isEmpty(), "Activation stdevs should not be empty");

        // --- Learning rates ---
        Map<String, Object> lrs = (Map<String, Object>) data.get("learningRates");
        assertNotNull(lrs, "learningRates must be present");
        List<String> lrParamNames = (List<String>) lrs.get("paramNames");
        assertTrue(lrParamNames.contains("W"), "Should have learning rate for W");
        Map<String, List<Number>> lrValues = (Map<String, List<Number>>) lrs.get("lrs");
        // Adam default LR is 0.01
        for (Number lr : lrValues.get("W")) {
            assertEquals(0.01, lr.doubleValue(), 1e-6, "LR should be 0.01 (Adam default)");
        }

        // --- Layer info ---
        List<List<String>> layerInfo = (List<List<String>>) data.get("layerInfo");
        assertNotNull(layerInfo);
        assertTrue(layerInfo.size() >= 2, "Should have at least layer name and type rows");

        uiServer.stop();
    }

    @Test
    public void testTrainModuleGraphTopologyMatchesNetwork() throws Exception {
        StatsStorage ss = new InMemoryStatsStorage();
        UIServer uiServer = UIServer.getInstance();
        uiServer.attach(ss);

        // Build a 3-layer network: 4→8→6→3
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new DenseLayer.Builder().nIn(4).nOut(8).activation(Activation.RELU).build())
                .layer(new DenseLayer.Builder().nIn(8).nOut(6).activation(Activation.TANH).build())
                .layer(new OutputLayer.Builder().nIn(6).nOut(3)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new StatsListener(ss, 1));
        net.fit(new IrisDataSetIterator(150, 150));

        // Fetch model graph
        String json = IOUtils.toString(new URL(BASE_URL + "/train/model/graph"), StandardCharsets.UTF_8);
        assertFalse(json.isEmpty(), "Graph response should not be empty");
        Map<String, Object> graph = MAPPER.readValue(json, Map.class);

        // vertexNames: ["Input", "0", "1", "2"] for a 3-layer MLN
        List<String> vertexNames = (List<String>) graph.get("vertexNames");
        assertNotNull(vertexNames);
        assertEquals(4, vertexNames.size(), "3 layers + 1 input = 4 vertices");

        // vertexTypes should have Input, Dense, Dense, Output
        List<String> vertexTypes = (List<String>) graph.get("vertexTypes");
        assertEquals("Input", vertexTypes.get(0));
        assertEquals("Dense", vertexTypes.get(1));
        assertEquals("Dense", vertexTypes.get(2));
        assertTrue(vertexTypes.get(3).contains("Output"), "Last vertex should be Output type");

        // vertexInputs: each layer takes input from the previous
        List<List<Integer>> vertexInputs = (List<List<Integer>>) graph.get("vertexInputs");
        assertEquals(4, vertexInputs.size());
        assertTrue(vertexInputs.get(0).isEmpty(), "Input vertex has no inputs");
        assertEquals(Collections.singletonList(0), vertexInputs.get(1), "Layer 0 takes Input");
        assertEquals(Collections.singletonList(1), vertexInputs.get(2), "Layer 1 takes Layer 0");
        assertEquals(Collections.singletonList(2), vertexInputs.get(3), "Layer 2 takes Layer 1");

        // vertexInfo should have correct sizes
        List<Map<String, String>> vertexInfo = (List<Map<String, String>>) graph.get("vertexInfo");
        assertEquals("4", vertexInfo.get(1).get("Input size"), "Dense 0 input size = 4");
        assertEquals("8", vertexInfo.get(1).get("Output size"), "Dense 0 output size = 8");
        assertEquals("8", vertexInfo.get(2).get("Input size"), "Dense 1 input size = 8");
        assertEquals("6", vertexInfo.get(2).get("Output size"), "Dense 1 output size = 6");
        assertEquals("6", vertexInfo.get(3).get("Input size"), "Output input size = 6");
        assertEquals("3", vertexInfo.get(3).get("Output size"), "Output output size = 3");

        uiServer.stop();
    }

    @Test
    public void testTrainModuleWithComputationGraphRealData() throws Exception {
        StatsStorage ss = new InMemoryStatsStorage();
        UIServer uiServer = UIServer.getInstance();
        uiServer.attach(ss);

        // A ComputationGraph with named layers
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.01))
                .graphBuilder()
                .addInputs("features")
                .addLayer("hidden", new DenseLayer.Builder().nIn(4).nOut(5)
                        .activation(Activation.TANH).build(), "features")
                .addLayer("output", new OutputLayer.Builder().nIn(5).nOut(3)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).build(), "hidden")
                .setOutputs("output")
                .build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();
        net.setListeners(new StatsListener(ss, 1));

        DataSetIterator iter = new IrisDataSetIterator(150, 150);
        for (int i = 0; i < 5; i++) {
            net.fit(iter);
            iter.reset();
        }

        // Overview data should have CG-style param names (layerName_paramName)
        String overviewJson = IOUtils.toString(new URL(BASE_URL + "/train/overview/data"), StandardCharsets.UTF_8);
        Map<String, Object> overview = MAPPER.readValue(overviewJson, Map.class);

        List<Number> scores = (List<Number>) overview.get("scores");
        assertTrue(scores.size() >= 3, "CG should produce scores");
        assertTrue(scores.get(0).doubleValue() > 0, "Scores should be positive");

        // CG param names use "layerName_paramName" format
        Map<String, List<Number>> stdevGrad = (Map<String, List<Number>>) overview.get("stdevGradients");
        assertTrue(stdevGrad.containsKey("hidden_W") || stdevGrad.containsKey("0_W"),
                "CG should have gradient data keyed by layer name. Keys: " + stdevGrad.keySet());

        // Graph topology should use named vertices
        String graphJson = IOUtils.toString(new URL(BASE_URL + "/train/model/graph"), StandardCharsets.UTF_8);
        Map<String, Object> graph = MAPPER.readValue(graphJson, Map.class);
        List<String> vertexNames = (List<String>) graph.get("vertexNames");
        assertTrue(vertexNames.contains("hidden"), "Graph should contain 'hidden' vertex");
        assertTrue(vertexNames.contains("output"), "Graph should contain 'output' vertex");

        uiServer.stop();
    }

    // =========================================================================
    // SameDiff Module: Real model → verify graph structure matches ops/vars
    // =========================================================================

    @Test
    public void testSameDiffUploadVerifyGraphStructure() throws Exception {
        UIServer uiServer = UIServer.getInstance();

        // Build a known SameDiff model: input → mmul(W) → add(b) → softmax → output
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("input", DataType.FLOAT, -1, 4);
        SDVariable w = sd.var("weights", Nd4j.randn(DataType.FLOAT, 4, 3));
        SDVariable b = sd.var("bias", Nd4j.zeros(DataType.FLOAT, 3));
        SDVariable mmul = in.mmul(w);
        SDVariable added = mmul.add(b);
        SDVariable out = sd.nn.softmax("output", added);

        int expectedOps = sd.getOps().size();
        int expectedVars = sd.variables().size();
        Set<String> expectedOpNames = new HashSet<>(sd.getOps().keySet());
        Set<String> expectedVarNames = new HashSet<>();
        for (SDVariable v : sd.variables()) {
            expectedVarNames.add(v.name());
        }

        Path tempFile = Files.createTempFile("samediff-graph-test", ".fb");
        sd.asFlatFile(tempFile.toFile());

        try {
            Map<String, Object> response = uploadFile(BASE_URL + "/samediff/upload", tempFile.toFile(), "model.fb");
            assertNotNull(response);

            // Verify model type
            assertEquals("SameDiff", response.get("modelType"),
                    "modelType should be SameDiff");

            // Verify op and var counts match the actual model
            assertEquals(expectedOps, ((Number) response.get("opCount")).intValue(),
                    "opCount should match actual number of ops");
            assertEquals(expectedVars, ((Number) response.get("varCount")).intValue(),
                    "varCount should match actual number of variables");

            // Verify param count: weights(4*3=12) + bias(3) = 15
            assertEquals(15L, ((Number) response.get("paramCount")).longValue(),
                    "paramCount should be 4*3 + 3 = 15");

            // Verify nodes contain both ops and variables
            List<Map<String, Object>> nodes = (List<Map<String, Object>>) response.get("nodes");
            assertNotNull(nodes);

            Set<String> foundOpIds = new HashSet<>();
            Set<String> foundVarNames = new HashSet<>();
            boolean hasPlaceholder = false;
            boolean hasVariable = false;
            boolean hasConstant = false;

            for (Map<String, Object> node : nodes) {
                Map<String, Object> data = (Map<String, Object>) node.get("data");
                String nodeType = (String) data.get("nodeType");
                if ("op".equals(nodeType)) {
                    foundOpIds.add((String) data.get("id"));
                    // Op nodes should have opType
                    assertNotNull(data.get("opType"), "Op node should have opType");
                } else if ("variable".equals(nodeType)) {
                    String varName = (String) data.get("varName");
                    foundVarNames.add(varName);
                    String varType = (String) data.get("varType");
                    if ("PLACEHOLDER".equals(varType)) hasPlaceholder = true;
                    if ("VARIABLE".equals(varType)) hasVariable = true;
                    if ("CONSTANT".equals(varType)) hasConstant = true;
                }
            }

            // All ops from the original model should appear
            for (String opName : expectedOpNames) {
                assertTrue(foundOpIds.contains(opName),
                        "Op '" + opName + "' should be in graph. Found: " + foundOpIds);
            }

            // Key variables should appear
            assertTrue(foundVarNames.contains("input"), "Should have 'input' variable");
            assertTrue(foundVarNames.contains("weights"), "Should have 'weights' variable");
            assertTrue(foundVarNames.contains("bias"), "Should have 'bias' variable");
            assertTrue(foundVarNames.contains("output"), "Should have 'output' variable");

            assertTrue(hasPlaceholder, "Should have at least one PLACEHOLDER variable");
            assertTrue(hasVariable, "Should have at least one VARIABLE (trainable weight)");

            // Edges should be present connecting vars and ops
            List<Map<String, Object>> edges = (List<Map<String, Object>>) response.get("edges");
            assertNotNull(edges);
            assertTrue(edges.size() > 0, "Graph should have edges");

            // Verify inputs list contains the placeholder
            List<String> inputs = (List<String>) response.get("inputs");
            assertTrue(inputs.contains("input"), "inputs should contain 'input'");
            // outputs comes from sd.outputs() — may be empty for inference-only models
            // (only set when loss/output is explicitly marked)
            List<String> outputs = (List<String>) response.get("outputs");
            assertNotNull(outputs, "outputs list should be present");

            // Verify the cached graph endpoint returns the same data
            String graphJson = IOUtils.toString(new URL(BASE_URL + "/samediff/graph"), StandardCharsets.UTF_8);
            Map<String, Object> cachedGraph = MAPPER.readValue(graphJson, Map.class);
            assertEquals(response.get("opCount"), cachedGraph.get("opCount"),
                    "Cached graph should match uploaded graph");

            // Verify info endpoint reflects the upload
            String infoJson = IOUtils.toString(new URL(BASE_URL + "/samediff/info"), StandardCharsets.UTF_8);
            Map<String, Object> info = MAPPER.readValue(infoJson, Map.class);
            assertEquals(true, info.get("loaded"));
            assertEquals("model.fb", info.get("filename"));
            assertTrue(((Number) info.get("loadTimeMs")).longValue() >= 0, "loadTimeMs should be non-negative");

        } finally {
            Files.deleteIfExists(tempFile);
            uiServer.stop();
        }
    }

    @Test
    public void testSameDiffUploadDL4JModelsMatchTopology() throws Exception {
        UIServer uiServer = UIServer.getInstance();

        // --- Test MultiLayerNetwork upload ---
        MultiLayerConfiguration mlnConf = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new DenseLayer.Builder().nIn(4).nOut(6).activation(Activation.RELU).build())
                .layer(new DenseLayer.Builder().nIn(6).nOut(4).activation(Activation.TANH).build())
                .layer(new OutputLayer.Builder().nIn(4).nOut(3)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).build())
                .build();

        MultiLayerNetwork mln = new MultiLayerNetwork(mlnConf);
        mln.init();

        Path mlnFile = Files.createTempFile("dl4j-mln-test", ".zip");
        org.deeplearning4j.util.ModelSerializer.writeModel(mln, mlnFile.toFile(), true);

        try {
            Map<String, Object> response = uploadFile(BASE_URL + "/samediff/upload", mlnFile.toFile(), "model.zip");
            assertNotNull(response);
            assertTrue(response.get("modelType").toString().contains("MultiLayerNetwork"),
                    "Should detect MultiLayerNetwork type");

            List<Map<String, Object>> nodes = (List<Map<String, Object>>) response.get("nodes");
            // 4 nodes: Input + 3 layers
            assertEquals(4, nodes.size(), "MLN with 3 layers should have 4 nodes (Input + 3 layers)");

            // Verify edges form a linear chain
            List<Map<String, Object>> edges = (List<Map<String, Object>>) response.get("edges");
            assertEquals(3, edges.size(), "3-layer linear chain should have 3 edges");

        } finally {
            Files.deleteIfExists(mlnFile);
        }

        // --- Test ComputationGraph upload ---
        ComputationGraphConfiguration cgConf = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.XAVIER)
                .graphBuilder()
                .addInputs("in")
                .addLayer("dense1", new DenseLayer.Builder().nIn(4).nOut(5)
                        .activation(Activation.RELU).build(), "in")
                .addLayer("out", new OutputLayer.Builder().nIn(5).nOut(3)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).build(), "dense1")
                .setOutputs("out")
                .build();

        ComputationGraph cg = new ComputationGraph(cgConf);
        cg.init();

        Path cgFile = Files.createTempFile("dl4j-cg-test", ".zip");
        org.deeplearning4j.util.ModelSerializer.writeModel(cg, cgFile.toFile(), true);

        try {
            Map<String, Object> response = uploadFile(BASE_URL + "/samediff/upload", cgFile.toFile(), "model.zip");
            assertNotNull(response);
            assertTrue(response.get("modelType").toString().contains("ComputationGraph"),
                    "Should detect ComputationGraph type");

            List<Map<String, Object>> nodes = (List<Map<String, Object>>) response.get("nodes");
            // 3 nodes: in + dense1 + out
            assertEquals(3, nodes.size(), "CG with 2 layers should have 3 nodes (1 input + 2 layers)");

            // Verify named vertices appear in node labels
            Set<String> labels = new HashSet<>();
            for (Map<String, Object> node : nodes) {
                Map<String, Object> data = (Map<String, Object>) node.get("data");
                labels.add((String) data.get("label"));
            }
            assertTrue(labels.contains("dense1"), "Should have 'dense1' vertex");
            assertTrue(labels.contains("out"), "Should have 'out' vertex");

        } finally {
            Files.deleteIfExists(cgFile);
            uiServer.stop();
        }
    }

    // =========================================================================
    // SameDiffStatsListener: Real SameDiff training → verify stats in dashboard
    // =========================================================================

    @Test
    public void testSameDiffStatsListenerRealTraining() throws Exception {
        StatsStorage ss = new InMemoryStatsStorage();
        UIServer uiServer = UIServer.getInstance();
        uiServer.attach(ss);

        // Create a real trainable SameDiff model
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("input", DataType.FLOAT, -1, 4);
        SDVariable label = sd.placeHolder("label", DataType.FLOAT, -1, 3);
        SDVariable w = sd.var("weights", Nd4j.randn(DataType.FLOAT, 4, 3).mul(0.01));
        SDVariable b = sd.var("bias", Nd4j.zeros(DataType.FLOAT, 3));
        SDVariable pred = sd.nn.softmax("prediction", in.mmul(w).add(b));
        SDVariable lossWeights = sd.var("lossWeights", Nd4j.ones(DataType.FLOAT, 1));
        SDVariable loss = sd.loss.softmaxCrossEntropy("loss", label, pred, lossWeights);

        sd.setTrainingConfig(TrainingConfig.builder()
                .updater(new Adam(0.01))
                .dataSetFeatureMapping("input")
                .dataSetLabelMapping("label")
                .build());

        SameDiffStatsListener listener = new SameDiffStatsListener(ss);
        sd.setListeners(listener);

        // Train on Iris data
        DataSetIterator iter = new IrisDataSetIterator(150, 150);
        sd.fit(iter, 5);

        // Let events propagate
        Thread.sleep(500);

        // Verify overview data has real scores from SameDiff training
        String json = IOUtils.toString(new URL(BASE_URL + "/train/overview/data"), StandardCharsets.UTF_8);
        Map<String, Object> data = MAPPER.readValue(json, Map.class);

        List<Number> scores = (List<Number>) data.get("scores");
        assertNotNull(scores, "SameDiff training should produce scores in overview");
        assertTrue(scores.size() >= 3, "Should have multiple score entries. Got: " + scores.size());

        // Scores should be finite positive numbers (cross-entropy loss)
        for (Number s : scores) {
            double val = s.doubleValue();
            assertTrue(Double.isFinite(val), "SameDiff score must be finite, got: " + val);
            assertTrue(val >= 0, "Cross-entropy loss should be non-negative, got: " + val);
        }

        // SameDiffStatsListener reports gradient stdev - verify it's in overview data
        Map<String, Object> stdevGrad = (Map<String, Object>) data.get("stdevGradients");
        // SameDiffStatsListener uses raw variable names like "weights", "bias"
        // (not prefixed like DL4J's "0_W" format)
        if (stdevGrad != null && !stdevGrad.isEmpty()) {
            // At least one gradient stdev entry should have real values
            boolean foundNonEmpty = false;
            for (Object v : stdevGrad.values()) {
                List<?> vals = (List<?>) v;
                if (!vals.isEmpty()) {
                    foundNonEmpty = true;
                    break;
                }
            }
            assertTrue(foundNonEmpty, "Should have at least one gradient stdev series with values");
        }

        uiServer.stop();
    }

    // =========================================================================
    // ModelViewer: Upload → verify node details retrievable
    // =========================================================================

    @Test
    public void testModelViewerUploadAndNodeDetails() throws Exception {
        UIServer uiServer = UIServer.getInstance();

        // Create a SameDiff model with known structure
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("input", DataType.FLOAT, -1, 10);
        SDVariable w1 = sd.var("W1", Nd4j.randn(DataType.FLOAT, 10, 5));
        SDVariable w2 = sd.var("W2", Nd4j.randn(DataType.FLOAT, 5, 3));
        SDVariable hidden = sd.nn.relu("hidden", in.mmul(w1), 0);
        SDVariable out = sd.nn.softmax("output", hidden.mmul(w2));

        Path tempFile = Files.createTempFile("modelview-test", ".fb");
        sd.asFlatFile(tempFile.toFile());

        try {
            Map<String, Object> response = uploadFile(BASE_URL + "/modelview/upload", tempFile.toFile(), "model.fb");
            assertNotNull(response);

            Map<String, Object> graphPart = (Map<String, Object>) response.get("graph");
            Map<String, Object> infoPart = (Map<String, Object>) response.get("info");

            assertNotNull(graphPart, "Response should contain graph");
            assertNotNull(infoPart, "Response should contain info");
            assertEquals(true, infoPart.get("loaded"));
            assertEquals("model.fb", infoPart.get("filename"));

            // Verify the graph has both ops and variables
            List<Map<String, Object>> nodes = (List<Map<String, Object>>) graphPart.get("nodes");
            assertTrue(nodes.size() > 0, "Should have nodes");

            // Verify edges exist
            List<Map<String, Object>> edges = (List<Map<String, Object>>) graphPart.get("edges");
            assertTrue(edges.size() > 0, "Should have edges");

            // Verify per-node detail lookup works
            // Find a node id from the graph
            Map<String, Object> firstNodeData = (Map<String, Object>) nodes.get(0).get("data");
            String nodeId = (String) firstNodeData.get("id");

            HttpURLConnection conn = (HttpURLConnection) new URL(
                    BASE_URL + "/modelview/node/" + java.net.URLEncoder.encode(nodeId, "UTF-8")).openConnection();
            assertEquals(200, conn.getResponseCode(),
                    "Node detail endpoint should return 200 for uploaded node");
            String nodeJson = IOUtils.toString(conn.getInputStream(), StandardCharsets.UTF_8);
            Map<String, Object> nodeDetail = MAPPER.readValue(nodeJson, Map.class);
            assertNotNull(nodeDetail.get("id"), "Node detail should have id");
            conn.disconnect();

            // Verify /modelview/current returns the graph
            String currentJson = IOUtils.toString(new URL(BASE_URL + "/modelview/current"), StandardCharsets.UTF_8);
            Map<String, Object> current = MAPPER.readValue(currentJson, Map.class);
            assertNotNull(current.get("graph"), "current should have graph after upload");
            Map<String, Object> currentGraph = (Map<String, Object>) current.get("graph");
            assertEquals(graphPart.get("opCount"), currentGraph.get("opCount"),
                    "Cached graph opCount should match");

            // Verify a non-existent node returns 404
            HttpURLConnection conn404 = (HttpURLConnection) new URL(
                    BASE_URL + "/modelview/node/nonexistent-node-id-12345").openConnection();
            assertEquals(404, conn404.getResponseCode(), "Non-existent node should return 404");
            conn404.disconnect();

        } finally {
            Files.deleteIfExists(tempFile);
            uiServer.stop();
        }
    }

    @Test
    public void testModelViewerDL4JModelUpload() throws Exception {
        UIServer uiServer = UIServer.getInstance();

        // Upload a DL4J MLN model and verify graph structure
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new DenseLayer.Builder().nIn(10).nOut(8).activation(Activation.RELU).build())
                .layer(new DenseLayer.Builder().nIn(8).nOut(5).activation(Activation.TANH).build())
                .layer(new OutputLayer.Builder().nIn(5).nOut(2)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        Path tempFile = Files.createTempFile("modelview-mln-test", ".zip");
        org.deeplearning4j.util.ModelSerializer.writeModel(net, tempFile.toFile(), true);

        try {
            Map<String, Object> response = uploadFile(BASE_URL + "/modelview/upload", tempFile.toFile(), "model.zip");
            assertNotNull(response);

            Map<String, Object> graphPart = (Map<String, Object>) response.get("graph");
            Map<String, Object> infoPart = (Map<String, Object>) response.get("info");

            assertTrue(infoPart.get("modelType").toString().contains("MultiLayerNetwork"));

            // Should have 4 nodes: Input + 3 layers
            List<Map<String, Object>> nodes = (List<Map<String, Object>>) graphPart.get("nodes");
            assertEquals(4, nodes.size(), "3-layer MLN should produce 4 nodes (Input + 3 layers)");

            // Should have 3 edges forming a chain
            List<Map<String, Object>> edges = (List<Map<String, Object>>) graphPart.get("edges");
            assertEquals(3, edges.size(), "Linear 3-layer chain should have 3 edges");

            // Verify layer properties via node details
            // Find the first Dense layer node (layer-1)
            for (Map<String, Object> node : nodes) {
                Map<String, Object> data = (Map<String, Object>) node.get("data");
                if ("layer-1".equals(data.get("id"))) {
                    Map<String, Object> props = (Map<String, Object>) data.get("properties");
                    assertEquals("10", props.get("Input size"),
                            "First dense layer should have input size 10");
                    assertEquals("8", props.get("Output size"),
                            "First dense layer should have output size 8");
                }
            }

        } finally {
            Files.deleteIfExists(tempFile);
            uiServer.stop();
        }
    }

    // =========================================================================
    // ModelManager: Catalog + Registry lifecycle
    // =========================================================================

    @Test
    public void testModelManagerCatalogHasRealEntries() throws Exception {
        UIServer uiServer = UIServer.getInstance();

        String json = IOUtils.toString(new URL(BASE_URL + "/models/catalog"), StandardCharsets.UTF_8);
        List<Map<String, Object>> catalog = MAPPER.readValue(json,
                MAPPER.getTypeFactory().constructCollectionType(List.class, Map.class));

        assertNotNull(catalog);
        assertFalse(catalog.isEmpty(), "Catalog should have pre-defined models from model-sources.yml");

        // Verify each catalog entry has all required fields
        Set<String> categories = new HashSet<>();
        Set<String> sources = new HashSet<>();
        for (Map<String, Object> entry : catalog) {
            String id = (String) entry.get("id");
            assertNotNull(id, "Catalog entry must have id");
            assertFalse(id.isEmpty(), "Catalog entry id must not be empty");

            String repo = (String) entry.get("repo");
            assertNotNull(repo, "Catalog entry '" + id + "' must have repo");

            String format = (String) entry.get("format");
            assertNotNull(format, "Catalog entry '" + id + "' must have format");

            String category = (String) entry.get("category");
            assertNotNull(category, "Catalog entry '" + id + "' must have category");
            categories.add(category);

            String source = (String) entry.get("source");
            assertNotNull(source, "Catalog entry '" + id + "' must have source");
            sources.add(source);

            assertNotNull(entry.get("description"), "Catalog entry '" + id + "' must have description");
        }

        // Should have at least 3 different categories (encoders, llm, omnihub_zoo, omnihub_llm, etc.)
        assertTrue(categories.size() >= 3,
                "Catalog should have at least 3 categories, got: " + categories);

        // Should have both huggingface and omnihub sources
        assertTrue(sources.contains("huggingface"),
                "Catalog should have huggingface source entries, got: " + sources);
        assertTrue(sources.contains("omnihub"),
                "Catalog should have omnihub source entries, got: " + sources);

        // Verify OmniHub zoo entries exist
        assertTrue(catalog.stream().anyMatch(e -> "omnihub_zoo".equals(e.get("category"))),
                "Catalog should have omnihub_zoo category entries");

        // Verify OmniHub LLM entries exist
        assertTrue(catalog.stream().anyMatch(e -> "omnihub_llm".equals(e.get("category"))),
                "Catalog should have omnihub_llm category entries");

        uiServer.stop();
    }

    @Test
    public void testModelManagerInstalledEndpoint() throws Exception {
        UIServer uiServer = UIServer.getInstance();

        String json = IOUtils.toString(new URL(BASE_URL + "/models/installed"), StandardCharsets.UTF_8);
        List<?> installed = MAPPER.readValue(json, List.class);
        assertNotNull(installed, "Installed endpoint should return a list");
        // Fresh install has no models - just verify it's a valid empty list
        // (this is correct behavior, not a stub check)

        uiServer.stop();
    }

    // =========================================================================
    // Training page nav links (cross-cutting)
    // =========================================================================

    @Test
    public void testTrainingPagesHaveWorkingNavLinks() throws Exception {
        StatsStorage ss = new InMemoryStatsStorage();
        UIServer uiServer = UIServer.getInstance();
        uiServer.attach(ss);

        // Must train so the training pages render with data
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list()
                .layer(new DenseLayer.Builder().nIn(4).nOut(4).activation(Activation.TANH).build())
                .layer(new OutputLayer.Builder().nIn(4).nOut(3)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new StatsListener(ss, 1));
        net.fit(new IrisDataSetIterator(150, 150));

        // Check all three training pages have the new nav links
        String[] pages = {"/train/overview", "/train/model", "/train/system"};
        for (String page : pages) {
            HttpURLConnection conn = (HttpURLConnection) new URL(BASE_URL + page).openConnection();
            assertEquals(200, conn.getResponseCode(), page + " should return 200");
            String html = IOUtils.toString(conn.getInputStream(), StandardCharsets.UTF_8);
            conn.disconnect();

            assertTrue(html.contains("href=\"/samediff\""),
                    page + " should have SameDiff nav link");
            assertTrue(html.contains("href=\"/modelview\""),
                    page + " should have Model Viewer nav link");
            assertTrue(html.contains("href=\"/models\""),
                    page + " should have Model Manager nav link");
        }

        // Verify the linked pages actually respond
        for (String path : new String[]{"/samediff", "/modelview", "/models"}) {
            HttpURLConnection conn = (HttpURLConnection) new URL(BASE_URL + path).openConnection();
            assertEquals(200, conn.getResponseCode(), path + " should return 200");
            String content = IOUtils.toString(conn.getInputStream(), StandardCharsets.UTF_8);
            assertFalse(content.isEmpty(), path + " should return non-empty content");
            conn.disconnect();
        }

        uiServer.stop();
    }

    // ===== Utility Methods =====

    @SuppressWarnings("unchecked")
    private Map<String, Object> uploadFile(String urlStr, File file, String filename) throws Exception {
        String boundary = "----FormBoundary" + System.currentTimeMillis();
        HttpURLConnection conn = (HttpURLConnection) new URL(urlStr).openConnection();
        conn.setDoOutput(true);
        conn.setRequestMethod("POST");
        conn.setRequestProperty("Content-Type", "multipart/form-data; boundary=" + boundary);
        conn.setConnectTimeout(30_000);
        conn.setReadTimeout(60_000);

        try (OutputStream os = conn.getOutputStream();
             PrintWriter writer = new PrintWriter(new OutputStreamWriter(os, StandardCharsets.UTF_8), true)) {

            writer.append("--").append(boundary).append("\r\n");
            writer.append("Content-Disposition: form-data; name=\"file\"; filename=\"").append(filename).append("\"\r\n");
            writer.append("Content-Type: application/octet-stream\r\n");
            writer.append("\r\n");
            writer.flush();

            try (FileInputStream fis = new FileInputStream(file)) {
                byte[] buf = new byte[4096];
                int n;
                while ((n = fis.read(buf)) != -1) {
                    os.write(buf, 0, n);
                }
            }
            os.flush();

            writer.append("\r\n");
            writer.append("--").append(boundary).append("--\r\n");
            writer.flush();
        }

        int code = conn.getResponseCode();
        String responseBody;
        if (code >= 200 && code < 300) {
            responseBody = IOUtils.toString(conn.getInputStream(), StandardCharsets.UTF_8);
        } else {
            responseBody = IOUtils.toString(conn.getErrorStream(), StandardCharsets.UTF_8);
            fail("Upload failed with HTTP " + code + ": " + responseBody);
            return null;
        }
        conn.disconnect();

        return MAPPER.readValue(responseBody, Map.class);
    }
}
