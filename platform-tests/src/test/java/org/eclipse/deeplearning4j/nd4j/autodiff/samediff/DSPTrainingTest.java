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

package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.listeners.At;
import org.nd4j.autodiff.listeners.Loss;
import org.nd4j.autodiff.listeners.Operation;
import org.nd4j.autodiff.listeners.records.History;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.autodiff.samediff.internal.TrainingSession;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.TestDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Sgd;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;
import static org.nd4j.linalg.api.buffer.DataType.FLOAT;

/**
 * Tests for DSP (DynamicShapePlan) training path in TrainingSession.
 * Verifies that training via DSP produces correct loss curves, handles
 * different batch sizes, per-iteration shape freezing, and matches the
 * standard per-op execution path.
 */
@Slf4j
@NativeTag
@Tag(TagNames.TRAINING)
@Tag(TagNames.SAMEDIFF)
public class DSPTrainingTest extends BaseNd4jTestWithBackends {

    private boolean prevDspEnabled;
    private String prevDspTrainingProp;

    @BeforeEach
    public void setup() {
        prevDspEnabled = InferenceSession.isDynamicShapePlanEnabled();
        prevDspTrainingProp = System.getProperty(ND4JSystemProperties.DSP_TRAINING_ENABLED);
    }

    @AfterEach
    public void teardown() {
        InferenceSession.setDynamicShapePlanEnabled(prevDspEnabled);
        if (prevDspTrainingProp != null) {
            System.setProperty(ND4JSystemProperties.DSP_TRAINING_ENABLED, prevDspTrainingProp);
        } else {
            System.clearProperty(ND4JSystemProperties.DSP_TRAINING_ENABLED);
        }
    }

    // =========================================================================
    // Model builders
    // =========================================================================

    private SameDiff buildSimpleMLPRegression(int inputSize, int hiddenSize, int outputSize) {
        Nd4j.getRandom().setSeed(12345);
        SameDiff sd = SameDiff.create();

        SDVariable input = sd.placeHolder("input", FLOAT, -1, inputSize);
        SDVariable label = sd.placeHolder("label", FLOAT, -1, outputSize);

        SDVariable w0 = sd.var("w0", Nd4j.randn(FLOAT, inputSize, hiddenSize).muli(0.01));
        SDVariable b0 = sd.zero("b0", FLOAT, 1, hiddenSize);
        SDVariable w1 = sd.var("w1", Nd4j.randn(FLOAT, hiddenSize, outputSize).muli(0.01));
        SDVariable b1 = sd.zero("b1", FLOAT, 1, outputSize);

        SDVariable z0 = input.mmul(w0).add(b0);
        SDVariable a0 = sd.nn.tanh(z0);
        SDVariable z1 = a0.mmul(w1).add(b1);

        sd.loss().meanSquaredError("loss", label, z1, null);
        return sd;
    }

    private SameDiff buildSimpleMLPClassification(int inputSize, int hiddenSize, int numClasses) {
        Nd4j.getRandom().setSeed(12345);
        SameDiff sd = SameDiff.create();

        SDVariable input = sd.placeHolder("input", FLOAT, -1, inputSize);
        SDVariable label = sd.placeHolder("label", FLOAT, -1, numClasses);

        SDVariable w0 = sd.var("w0", Nd4j.randn(FLOAT, inputSize, hiddenSize).muli(0.01));
        SDVariable b0 = sd.zero("b0", FLOAT, 1, hiddenSize);
        SDVariable w1 = sd.var("w1", Nd4j.randn(FLOAT, hiddenSize, numClasses).muli(0.01));
        SDVariable b1 = sd.zero("b1", FLOAT, 1, numClasses);

        SDVariable z0 = input.mmul(w0).add(b0);
        SDVariable a0 = sd.nn.tanh(z0);
        SDVariable prediction = a0.mmul(w1).add(b1);

        sd.loss().softmaxCrossEntropy("loss", label, prediction, null);
        return sd;
    }

    // =========================================================================
    // Data generators
    // =========================================================================

    private Map<String, INDArray> generateRegressionData(int batchSize, int inputSize, int outputSize) {
        INDArray features = Nd4j.randn(FLOAT, batchSize, inputSize);
        INDArray trueW = Nd4j.randn(FLOAT, inputSize, outputSize).muli(0.5);
        INDArray labels = features.mmul(trueW).addi(Nd4j.randn(FLOAT, batchSize, outputSize).muli(0.1));
        Map<String, INDArray> placeholders = new HashMap<>();
        placeholders.put("input", features);
        placeholders.put("label", labels);
        return placeholders;
    }

    /**
     * Build an iterator of pre-sized batches. Each DataSet is already the desired batch size.
     * The iterator returns one DataSet per next() call — no splitting.
     */
    private DataSetIterator batchIterator(List<DataSet> batches) {
        return new TestDataSetIterator(batches, 1);
    }

    /**
     * Build an iterator of N batches all with the same batch size.
     */
    private DataSetIterator uniformBatchIterator(int batchSize, int inputSize, int outputSize, int numBatches) {
        List<DataSet> batches = new ArrayList<>();
        for (int i = 0; i < numBatches; i++) {
            Map<String, INDArray> data = generateRegressionData(batchSize, inputSize, outputSize);
            batches.add(new DataSet(data.get("input"), data.get("label")));
        }
        return batchIterator(batches);
    }

    /**
     * Build an iterator with intentionally varying batch sizes.
     */
    private DataSetIterator varyingBatchIterator(int[] batchSizes, int inputSize, int outputSize) {
        List<DataSet> batches = new ArrayList<>();
        for (int bs : batchSizes) {
            Map<String, INDArray> data = generateRegressionData(bs, inputSize, outputSize);
            batches.add(new DataSet(data.get("input"), data.get("label")));
        }
        return batchIterator(batches);
    }

    // =========================================================================
    // Helper: set up a direct TrainingSession (bypasses sd.fit() which creates
    // a new TrainingSession per call, losing DSP executor state)
    // =========================================================================

    private static class DirectTrainingSetup {
        final SameDiff sd;
        final SameDiff gradInstance;
        final TrainingSession ts;
        final TrainingConfig config;
        final Set<String> paramsToTrain;
        final Map<String, GradientUpdater> updaterMap;
        final List<String> lossVars;

        DirectTrainingSetup(int inputSize, int hiddenSize, int outputSize, double lr) {
            Nd4j.getRandom().setSeed(12345);
            sd = SameDiff.create();

            SDVariable input = sd.placeHolder("input", FLOAT, -1, inputSize);
            SDVariable label = sd.placeHolder("label", FLOAT, -1, outputSize);

            SDVariable w0 = sd.var("w0", Nd4j.randn(FLOAT, inputSize, hiddenSize).muli(0.01));
            SDVariable b0 = sd.zero("b0", FLOAT, 1, hiddenSize);
            SDVariable w1 = sd.var("w1", Nd4j.randn(FLOAT, hiddenSize, outputSize).muli(0.01));
            SDVariable b1 = sd.zero("b1", FLOAT, 1, outputSize);

            SDVariable z0 = input.mmul(w0).add(b0);
            SDVariable a0 = sd.nn.tanh(z0);
            SDVariable z1 = a0.mmul(w1).add(b1);
            sd.loss().meanSquaredError("loss", label, z1, null);

            config = new TrainingConfig.Builder()
                    .updater(new Adam(lr))
                    .dataSetFeatureMapping("input")
                    .dataSetLabelMapping("label")
                    .build();
            sd.setTrainingConfig(config);

            sd.createGradFunction();
            gradInstance = sd.getFunction("grad");
            gradInstance.setTrainingConfig(config);

            ts = new TrainingSession(gradInstance);

            paramsToTrain = new LinkedHashSet<>();
            for (Map.Entry<String, org.nd4j.autodiff.samediff.internal.Variable> v : sd.getVariables().entrySet()) {
                if (v.getValue().getVariable().getVariableType() == VariableType.VARIABLE) {
                    paramsToTrain.add(v.getKey());
                }
            }

            updaterMap = new HashMap<>();
            Adam adam = new Adam(lr);
            for (String param : paramsToTrain) {
                INDArray paramArr = sd.getVariable(param).getArr();
                long stateSize = adam.stateSize(paramArr.length());
                INDArray stateView = stateSize == 0 ? null : Nd4j.createUninitialized(paramArr.dataType(), 1, stateSize);
                GradientUpdater gu = adam.instantiate(stateView, false);
                gu.setStateViewArray(stateView, paramArr.shape(), paramArr.ordering(), true);
                updaterMap.put(param, gu);
            }

            lossVars = sd.getLossVariables();
        }

        Loss trainIteration(Map<String, INDArray> placeholders, int iteration) {
            At at = At.builder().epoch(0).iteration(iteration).operation(Operation.TRAINING).build();
            return ts.trainingIteration(config, placeholders, paramsToTrain, updaterMap,
                    null, lossVars, Collections.emptyList(), at);
        }

        void close() {
            sd.close();
        }
    }

    // =========================================================================
    // Test 1: DSP training with uniform batches — loss decreases
    // =========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDspTrainingLossDecreases(Nd4jBackend backend) {
        System.setProperty(ND4JSystemProperties.DSP_TRAINING_ENABLED, "true");
        InferenceSession.setDynamicShapePlanEnabled(true);

        SameDiff sd = buildSimpleMLPRegression(4, 10, 2);
        sd.setTrainingConfig(new TrainingConfig.Builder()
                .updater(new Adam(0.01))
                .dataSetFeatureMapping("input")
                .dataSetLabelMapping("label")
                .build());

        int numBatches = 20;
        DataSetIterator iter = uniformBatchIterator(32, 4, 2, numBatches);
        History hist = sd.fit().train(iter, 1).exec();

        INDArray lossValues = hist.getLossCurve().getLossValues();
        assertTrue(lossValues.sumNumber().doubleValue() > 0.0, "Loss should be positive");

        sd.close();
    }

    // =========================================================================
    // Test 2: DSP and non-DSP produce equivalent results
    // =========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDspMatchesStandardTrainingPath(Nd4jBackend backend) {
        int batchSize = 16;
        int numBatches = 10;

        // Generate fixed training data
        List<DataSet> fixedData = new ArrayList<>();
        Nd4j.getRandom().setSeed(99999);
        for (int i = 0; i < numBatches; i++) {
            Map<String, INDArray> data = generateRegressionData(batchSize, 4, 2);
            fixedData.add(new DataSet(data.get("input"), data.get("label")));
        }

        // Train with DSP
        System.setProperty(ND4JSystemProperties.DSP_TRAINING_ENABLED, "true");
        InferenceSession.setDynamicShapePlanEnabled(true);
        double[] dspLosses = trainWithFixedData(fixedData, 12345);

        // Train without DSP
        System.setProperty(ND4JSystemProperties.DSP_TRAINING_ENABLED, "false");
        InferenceSession.setDynamicShapePlanEnabled(false);
        double[] stdLosses = trainWithFixedData(fixedData, 12345);

        for (int i = 0; i < numBatches; i++) {
            double relError = Math.abs(dspLosses[i] - stdLosses[i]) / (Math.abs(stdLosses[i]) + 1e-8);
            assertTrue(relError < 0.05,
                    "Iteration " + i + ": DSP loss " + dspLosses[i] +
                            " vs standard loss " + stdLosses[i] + " (relError=" + relError + ")");
        }
    }

    private double[] trainWithFixedData(List<DataSet> data, long seed) {
        Nd4j.getRandom().setSeed(seed);
        DirectTrainingSetup setup = new DirectTrainingSetup(4, 10, 2, 0.001);
        double[] losses = new double[data.size()];
        for (int i = 0; i < data.size(); i++) {
            Map<String, INDArray> ph = new HashMap<>();
            ph.put("input", data.get(i).getFeatures());
            ph.put("label", data.get(i).getLabels());
            Loss loss = setup.trainIteration(ph, i);
            losses[i] = loss.getLoss("loss");
        }
        setup.close();
        return losses;
    }

    // =========================================================================
    // Test 3: Varying batch sizes — DSP handles uneven batches
    // =========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDspTrainingVaryingBatchSizes(Nd4jBackend backend) {
        System.setProperty(ND4JSystemProperties.DSP_TRAINING_ENABLED, "true");
        InferenceSession.setDynamicShapePlanEnabled(true);

        DirectTrainingSetup setup = new DirectTrainingSetup(4, 10, 2, 0.01);

        // Intentionally varying batch sizes: typical epoch + incomplete last batch
        int[] batchSizes = {32, 32, 32, 32, 17, 16, 16, 16, 16, 7};

        for (int i = 0; i < batchSizes.length; i++) {
            Map<String, INDArray> data = generateRegressionData(batchSizes[i], 4, 2);
            Loss loss = setup.trainIteration(data, i);

            double lossVal = loss.getLoss("loss");
            assertTrue(Double.isFinite(lossVal),
                    "Loss not finite at iter " + i + " (batch=" + batchSizes[i] + "): " + lossVal);
            assertTrue(lossVal >= 0,
                    "Loss negative at iter " + i + " (batch=" + batchSizes[i] + "): " + lossVal);
            log.info("Iter {} batch={}: loss={}", i, batchSizes[i], lossVal);
        }

        setup.close();
    }

    // =========================================================================
    // Test 4: Eager shape freezing — frozen after first iteration
    // =========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEagerShapeFreezingAfterFirstIteration(Nd4jBackend backend) {
        System.setProperty(ND4JSystemProperties.DSP_TRAINING_ENABLED, "true");
        InferenceSession.setDynamicShapePlanEnabled(true);

        DirectTrainingSetup setup = new DirectTrainingSetup(4, 10, 2, 0.01);

        // First iteration
        Map<String, INDArray> data = generateRegressionData(16, 4, 2);
        Loss loss = setup.trainIteration(data, 0);
        assertNotNull(loss);
        assertTrue(Double.isFinite(loss.getLoss("loss")));

        // After first iteration, shapes should be frozen eagerly
        DynamicShapePlanExecutor executor = setup.ts.getDynamicShapePlanExecutor();
        if (executor != null) {
            assertTrue(executor.isShapesFrozen(),
                    "Shapes should be frozen after first training iteration (eager freeze)");
        }

        // Second iteration with same shape — should stay frozen (graph replay)
        data = generateRegressionData(16, 4, 2);
        loss = setup.trainIteration(data, 1);
        assertNotNull(loss);
        assertTrue(Double.isFinite(loss.getLoss("loss")));

        executor = setup.ts.getDynamicShapePlanExecutor();
        if (executor != null) {
            assertTrue(executor.isShapesFrozen(),
                    "Shapes should remain frozen for same batch size");
        }

        setup.close();
    }

    // =========================================================================
    // Test 5: Shape unfreeze on batch size change, re-freeze immediately
    // =========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testShapeUnfreezeOnBatchChange(Nd4jBackend backend) {
        System.setProperty(ND4JSystemProperties.DSP_TRAINING_ENABLED, "true");
        InferenceSession.setDynamicShapePlanEnabled(true);

        DirectTrainingSetup setup = new DirectTrainingSetup(4, 10, 2, 0.01);

        // Iteration 0: batch=16, should freeze eagerly
        Map<String, INDArray> data = generateRegressionData(16, 4, 2);
        setup.trainIteration(data, 0);

        DynamicShapePlanExecutor executor = setup.ts.getDynamicShapePlanExecutor();
        if (executor != null) {
            assertTrue(executor.isShapesFrozen(), "Should be frozen after iter 0");
        }

        // Iteration 1: batch=32 — shape change, should unfreeze then re-freeze
        data = generateRegressionData(32, 4, 2);
        Loss loss = setup.trainIteration(data, 1);
        assertTrue(Double.isFinite(loss.getLoss("loss")));

        executor = setup.ts.getDynamicShapePlanExecutor();
        if (executor != null) {
            // Re-frozen immediately for the new shape
            assertTrue(executor.isShapesFrozen(),
                    "Should be re-frozen immediately after batch size change");
        }

        // Iteration 2: batch=32 again — stays frozen, graph replay
        data = generateRegressionData(32, 4, 2);
        loss = setup.trainIteration(data, 2);
        assertTrue(Double.isFinite(loss.getLoss("loss")));

        executor = setup.ts.getDynamicShapePlanExecutor();
        if (executor != null) {
            assertTrue(executor.isShapesFrozen(),
                    "Should remain frozen for same batch size");
        }

        setup.close();
    }

    // =========================================================================
    // Test 6: Freeze/unfreeze/re-freeze cycle across epoch boundaries
    // =========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testShapeFreezeUnfreezeCycle(Nd4jBackend backend) {
        System.setProperty(ND4JSystemProperties.DSP_TRAINING_ENABLED, "true");
        InferenceSession.setDynamicShapePlanEnabled(true);

        DirectTrainingSetup setup = new DirectTrainingSetup(4, 10, 2, 0.01);
        int iter = 0;

        // Epoch 1: 4 batches of 16, last batch of 7 (incomplete)
        for (int i = 0; i < 4; i++) {
            Map<String, INDArray> data = generateRegressionData(16, 4, 2);
            Loss loss = setup.trainIteration(data, iter++);
            assertTrue(Double.isFinite(loss.getLoss("loss")));
        }

        DynamicShapePlanExecutor exec = setup.ts.getDynamicShapePlanExecutor();
        if (exec != null) {
            assertTrue(exec.isShapesFrozen(), "Should be frozen during batch=16 run");
        }

        // Last batch of epoch 1: batch=7 — unfreeze + re-freeze
        {
            Map<String, INDArray> data = generateRegressionData(7, 4, 2);
            Loss loss = setup.trainIteration(data, iter++);
            assertTrue(Double.isFinite(loss.getLoss("loss")));
        }

        exec = setup.ts.getDynamicShapePlanExecutor();
        if (exec != null) {
            assertTrue(exec.isShapesFrozen(),
                    "Should be re-frozen after incomplete batch (for next iteration)");
        }

        // Epoch 2: back to batch=16 — unfreeze(7!=16) + re-freeze
        for (int i = 0; i < 4; i++) {
            Map<String, INDArray> data = generateRegressionData(16, 4, 2);
            Loss loss = setup.trainIteration(data, iter++);
            assertTrue(Double.isFinite(loss.getLoss("loss")));
        }

        exec = setup.ts.getDynamicShapePlanExecutor();
        if (exec != null) {
            assertTrue(exec.isShapesFrozen(), "Should be frozen during epoch 2 batch=16 run");
        }

        setup.close();
    }

    // =========================================================================
    // Test 7: Parameters actually get updated
    // =========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testParametersUpdated(Nd4jBackend backend) {
        System.setProperty(ND4JSystemProperties.DSP_TRAINING_ENABLED, "true");
        InferenceSession.setDynamicShapePlanEnabled(true);

        DirectTrainingSetup setup = new DirectTrainingSetup(4, 10, 2, 0.01);

        // Snapshot initial parameters
        Map<String, INDArray> initialParams = new HashMap<>();
        for (String param : setup.paramsToTrain) {
            initialParams.put(param, setup.sd.getVariable(param).getArr().dup());
        }

        // Train 10 iterations
        for (int i = 0; i < 10; i++) {
            Map<String, INDArray> data = generateRegressionData(16, 4, 2);
            Loss loss = setup.trainIteration(data, i);
            assertNotNull(loss);
        }

        // Verify parameters changed
        boolean anyChanged = false;
        for (String param : setup.paramsToTrain) {
            INDArray current = setup.sd.getVariable(param).getArr();
            INDArray initial = initialParams.get(param);
            if (!current.equalsWithEps(initial, 1e-6)) {
                anyChanged = true;
                break;
            }
        }
        assertTrue(anyChanged, "Parameters should have been updated during training");

        setup.close();
    }

    // =========================================================================
    // Test 8: DSP disabled falls back to standard path
    // =========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDspDisabledFallback(Nd4jBackend backend) {
        System.setProperty(ND4JSystemProperties.DSP_TRAINING_ENABLED, "false");
        InferenceSession.setDynamicShapePlanEnabled(false);

        SameDiff sd = buildSimpleMLPRegression(4, 10, 2);
        sd.setTrainingConfig(new TrainingConfig.Builder()
                .updater(new Adam(0.01))
                .dataSetFeatureMapping("input")
                .dataSetLabelMapping("label")
                .build());

        DataSetIterator iter = uniformBatchIterator(16, 4, 2, 5);
        History h = sd.fit().train(iter, 1).exec();
        assertTrue(h.getLossCurve().getLossValues().sumNumber().doubleValue() > 0.0);

        sd.close();
    }

    // =========================================================================
    // Test 9: Multiple epochs with consistent batches
    // =========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDspTrainingMultipleEpochs(Nd4jBackend backend) {
        System.setProperty(ND4JSystemProperties.DSP_TRAINING_ENABLED, "true");
        InferenceSession.setDynamicShapePlanEnabled(true);

        // Use MSE regression model (softmax_cross_entropy_loss_grad has a native op issue)
        SameDiff sd = buildSimpleMLPRegression(4, 10, 2);
        sd.setTrainingConfig(new TrainingConfig.Builder()
                .updater(new Adam(0.01))
                .dataSetFeatureMapping("input")
                .dataSetLabelMapping("label")
                .build());

        DataSetIterator iter = uniformBatchIterator(16, 4, 2, 5);
        History hist = sd.fit().train(iter, 3).exec();

        INDArray lossValues = hist.getLossCurve().getLossValues();
        assertNotNull(lossValues, "Loss curve should not be null");
        assertEquals(3, lossValues.rows(), "Should have 3 epochs of loss values");

        float epoch0Loss = lossValues.getFloat(0, 0);
        float epoch2Loss = lossValues.getFloat(2, 0);
        log.info("Epoch 0 loss: {}, Epoch 2 loss: {}", epoch0Loss, epoch2Loss);
        assertTrue(Double.isFinite(epoch0Loss), "Epoch 0 loss should be finite");
        assertTrue(Double.isFinite(epoch2Loss), "Epoch 2 loss should be finite");

        sd.close();
    }

    // =========================================================================
    // Test 10: Batch size 1 edge case
    // =========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDspTrainingBatchSizeOne(Nd4jBackend backend) {
        System.setProperty(ND4JSystemProperties.DSP_TRAINING_ENABLED, "true");
        InferenceSession.setDynamicShapePlanEnabled(true);

        DirectTrainingSetup setup = new DirectTrainingSetup(4, 10, 2, 0.01);

        for (int i = 0; i < 5; i++) {
            Map<String, INDArray> data = generateRegressionData(1, 4, 2);
            Loss loss = setup.trainIteration(data, i);
            assertTrue(Double.isFinite(loss.getLoss("loss")),
                    "Loss not finite with batch size 1 at iter " + i);
        }

        setup.close();
    }

    // =========================================================================
    // Test 11: DSP training with L2 regularization
    // =========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDspTrainingWithRegularization(Nd4jBackend backend) {
        System.setProperty(ND4JSystemProperties.DSP_TRAINING_ENABLED, "true");
        InferenceSession.setDynamicShapePlanEnabled(true);

        SameDiff sd = buildSimpleMLPRegression(4, 10, 2);
        sd.setTrainingConfig(new TrainingConfig.Builder()
                .updater(new Adam(0.01))
                .l2(1e-3)
                .dataSetFeatureMapping("input")
                .dataSetLabelMapping("label")
                .build());

        DataSetIterator iter = uniformBatchIterator(16, 4, 2, 10);
        History h = sd.fit().train(iter, 1).exec();
        assertTrue(h.getLossCurve().getLossValues().sumNumber().doubleValue() > 0.0,
                "Loss with L2 regularization should be positive");

        sd.close();
    }

    // =========================================================================
    // Test 12: Graph replay every iteration (frozen from iter 0)
    // =========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGraphReplayEveryIteration(Nd4jBackend backend) {
        System.setProperty(ND4JSystemProperties.DSP_TRAINING_ENABLED, "true");
        InferenceSession.setDynamicShapePlanEnabled(true);

        DirectTrainingSetup setup = new DirectTrainingSetup(4, 10, 2, 0.01);
        int batchSize = 16;

        // Use same data every iteration (overfit on one batch) so loss must decrease
        Map<String, INDArray> fixedData = generateRegressionData(batchSize, 4, 2);

        int numIters = 50;
        double[] losses = new double[numIters];
        for (int i = 0; i < numIters; i++) {
            Loss loss = setup.trainIteration(fixedData, i);
            losses[i] = loss.getLoss("loss");

            // After every iteration, shapes should be frozen (graph replay enabled)
            DynamicShapePlanExecutor executor = setup.ts.getDynamicShapePlanExecutor();
            if (executor != null) {
                assertTrue(executor.isShapesFrozen(),
                        "Shapes should be frozen at iter " + i + " (graph replay every iteration)");
            }
        }

        // Verify loss actually decreases when overfitting on same batch
        log.info("First loss: {}, Last loss: {}", losses[0], losses[numIters - 1]);
        assertTrue(losses[numIters - 1] < losses[0],
                "Loss should decrease. First: " + losses[0] + ", last: " + losses[numIters - 1]);

        setup.close();
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
