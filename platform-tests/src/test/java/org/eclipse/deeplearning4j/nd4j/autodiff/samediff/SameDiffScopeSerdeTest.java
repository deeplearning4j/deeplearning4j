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

import static org.deeplearning4j.datasets.iterator.RandomDataSetIterator.Values.ONE_HOT;
import static org.deeplearning4j.datasets.iterator.RandomDataSetIterator.Values.ZEROS;
import static org.junit.jupiter.api.Assertions.*;
import static org.nd4j.linalg.api.buffer.DataType.FLOAT;

import java.io.File;
import java.util.*;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.datasets.iterator.RandomDataSetIterator;
import org.deeplearning4j.datasets.iterator.ReconstructionDataSetIterator;
import org.junit.jupiter.api.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.*;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.enums.WeightsFormat;
import org.nd4j.evaluation.IEvaluation;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.classification.EvaluationBinary;
import org.nd4j.evaluation.classification.EvaluationCalibration;
import org.nd4j.evaluation.classification.ROC;
import org.nd4j.evaluation.classification.ROCBinary;
import org.nd4j.evaluation.classification.ROCMultiClass;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.PaddingMode;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.adapter.SingletonDataSetIterator;
import org.nd4j.linalg.dataset.adapter.SingletonMultiDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.weightinit.impl.OneInitScheme;
import org.nd4j.weightinit.impl.UniformInitScheme;

@Slf4j
@NativeTag
@Tag(TagNames.SAMEDIFF)
public class SameDiffScopeSerdeTest extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    @BeforeEach
    public void before() {
        Nd4j.create(1);
        Nd4j.getRandom().setSeed(123);
    }

    @AfterEach
    public void after() {
        Nd4j.getNativeOps().enableDebugMode(false);
        Nd4j.getNativeOps().enableVerboseMode(false);
    }

    public Map<String, INDArray> variablesForInput() {
        INDArray inputs = Nd4j.create(new double[][]{
                {0.52, 1.12, 0.77},
                {0.88, -1.08, 0.15},
                {0.52, 0.06, -1.30},
                {0.74, -2.49, 1.39}
        });

        INDArray labels = Nd4j.create(new double[]{1, 1, 0, 1}).reshape(4, 1);
        INDArray weights = Nd4j.zeros(3, 1).castTo(labels.dataType());

        Map<String, INDArray> inputMap = new HashMap<>();
        inputMap.put("x", inputs);
        inputMap.put("w", weights);
        inputMap.put("y", labels);
        return inputMap;
    }

    private INDArray testLinearLayers(boolean relu, int batchSize, int modelDim, DataSet dataInput) {
        SameDiff sd = SameDiff.create();
        DataSetIterator data = new SingletonDataSetIterator(dataInput);
        SDVariable features = sd.placeHolder("features", FLOAT, batchSize, modelDim);
        SDVariable labels = sd.placeHolder("labels", FLOAT, batchSize, modelDim);
        SDVariable weights = sd.var("weights", new OneInitScheme('c'), FLOAT, modelDim, modelDim);
        SDVariable bias = sd.zero("bias", FLOAT, modelDim);
        SDVariable predictions = relu ? sd.nn.reluLayer("predictions", features, weights, bias) : sd.nn.linear("predictions", features, weights, bias);
        sd.loss.meanSquaredError("loss", labels, predictions, null);

        TrainingConfig config = new TrainingConfig.Builder()
                .updater(new Adam(0.1))
                .dataSetFeatureMapping("features")
                .dataSetLabelMapping("labels")
                .build();
        sd.setTrainingConfig(config);
        sd.fit(data, 10);

        Evaluation evaluation = new Evaluation();
        sd.evaluate(data, "predictions", evaluation);

        return sd.getVariable("predictions").eval(Collections.singletonMap("features", dataInput.getFeatures()));
    }

    private INDArray testLinearLayersManual(boolean manual, int batchSize, int modelDim, DataSet dataInput) {
        SameDiff sd = SameDiff.create();
        DataSetIterator data = new SingletonDataSetIterator(dataInput);
        SDVariable features = sd.placeHolder("features", FLOAT, batchSize, modelDim);
        SDVariable labels = sd.placeHolder("labels", FLOAT, batchSize, modelDim);
        SDVariable weights = sd.var("weights", new OneInitScheme('c'), FLOAT, modelDim, modelDim);
        SDVariable bias = sd.zero("bias", FLOAT, modelDim);
        SDVariable predictions = manual ? features.mmul(weights).add("predictions", bias) : sd.nn.linear("predictions", features, weights, bias);
        sd.loss.meanSquaredError("loss", labels, predictions, null);

        TrainingConfig config = new TrainingConfig.Builder()
                .updater(new Adam(0.1))
                .dataSetFeatureMapping("features")
                .dataSetLabelMapping("labels")
                .build();
        sd.setTrainingConfig(config);
        sd.fit(data, 10);

        Evaluation evaluation = new Evaluation();
        sd.evaluate(data, "predictions", evaluation);

        return sd.getVariable("predictions").eval(Collections.singletonMap("features", dataInput.getFeatures()));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLinearEquivalency(Nd4jBackend backend) {
        int batchSize = 32;
        int modelDim = 10;

        DataSetIterator iterator = new ReconstructionDataSetIterator(new RandomDataSetIterator(100, new long[]{batchSize, modelDim}, new long[]{}, ONE_HOT, ZEROS));
        DataSet next = iterator.next();
        assertEquals(testLinearLayers(true, batchSize, modelDim, next), testLinearLayers(false, batchSize, modelDim, next));
        assertEquals(testLinearLayersManual(true, batchSize, modelDim, next), testLinearLayersManual(false, batchSize, modelDim, next));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLabelInputPlaceHolderSgd(Nd4jBackend backend) {

        SameDiff sd = SameDiff.create();

        int nIn = 3;
        int nOut = 4;
        int minibatch = 3;
        SDVariable input = sd.var("in", new long[]{-1, nIn});
        SDVariable label = sd.var("label", new long[]{-1, nOut});
        assertTrue(input.isPlaceHolder());
        assertTrue(label.isPlaceHolder());
        SDVariable weights = sd.var("W", new long[]{nIn, nOut});
        SDVariable bias = sd.var("b", new long[]{1, nOut});

        SDVariable mmul = sd.mmul("mmul", input, weights);
        SDVariable z = mmul.add("z", bias);
        SDVariable out = sd.math().tanh(z);

        SDVariable diff = out.sub(label);
        SDVariable sqDiff = diff.mul(diff);
        SDVariable msePerEx = sd.mean("msePerEx", sqDiff, 1);
        SDVariable avgMSE = sd.mean("loss", msePerEx, 0);

        INDArray inputArr = Nd4j.rand(minibatch, nIn);
        INDArray labelArr = Nd4j.rand(minibatch, nOut);
        INDArray weightsArr = Nd4j.rand(nIn, nOut);
        INDArray biasArr = Nd4j.rand(1, nOut);

        sd.associateArrayWithVariable(inputArr, input);
        sd.associateArrayWithVariable(labelArr, label);
        sd.associateArrayWithVariable(weightsArr, weights);
        sd.associateArrayWithVariable(biasArr, bias);

        INDArray result = avgMSE.eval();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSequenceAdd(Nd4jBackend backend) throws Exception {
        assertThrows(NullPointerException.class, () -> {
            SameDiff sd = SameDiff.create();
            sd.addItemToSequence("dummy", null, 0);
        });

        assertThrows(IllegalStateException.class, () -> {
            SameDiff sd = SameDiff.create();
            sd.addItemToSequence("dummy", Nd4j.ones(1), 0);
        });

        SameDiff sd = SameDiff.create();
        sd.createSequence("x", new INDArray[]{Nd4j.ones(1)});
        assertTrue(sd.hasVariable("x"));
        assertEquals(VariableType.SEQUENCE, sd.getVariable("x").getVariableType());
        assertEquals(Nd4j.ones(1), sd.itemForSequence("x", 0));
        sd.setItemForSequenceAtIndex("x", Nd4j.ones(2), 0);
        assertEquals(Nd4j.ones(2), sd.itemForSequence("x", 0));
        assertEquals(1, sd.sequenceLength("x"));
        sd.removeItemFromSequence("x", 0);
        assertFalse(sd.hasVariable("x"));
        assertThrows(IllegalStateException.class, () -> {
            SameDiff sd2 = SameDiff.create();
            sd2.itemForSequence("x", 1);
        });
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSequenceNegativeIndex(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        INDArray[] sequence = {Nd4j.ones(1), Nd4j.ones(2)};
        sd.createSequence("x", sequence);
        sd.addItemToSequence("x", Nd4j.ones(3), -1);
        assertEquals(Nd4j.ones(3), sd.itemForSequence("x", -1));
        sd.removeItemFromSequence("x", -1);
        assertEquals(Nd4j.ones(2), sd.itemForSequence("x", -1));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReductionShapes1(Nd4jBackend backend) {

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", new long[]{10, 9, 8});
        SDVariable mean1 = sd.mean(in, 2);      //[10,9] out
        SDVariable mean2 = sd.mean(mean1, 1);   //[10] out
        Map<String, INDArray> m = sd.output((Map<String, INDArray>) null, mean1.name(), mean2.name());

        INDArray m1 = m.get(mean1.name());
        INDArray m2 = m.get(mean2.name());

        assertArrayEquals(new long[]{10, 9}, m1.shape());
        assertArrayEquals(new long[]{10}, m2.shape());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReductionShapes2(Nd4jBackend backend) {

        SameDiff sd2 = SameDiff.create();
        SDVariable in2 = sd2.var("in", new long[]{10, 9, 8});
        SDVariable meanA = sd2.mean(in2, 0);      //[9,8] out
        Map<String, INDArray> out = sd2.outputAll(null);
        assertArrayEquals(new long[]{9, 8}, out.get(meanA.name()).shape());

        SDVariable meanB = sd2.mean(meanA, 0);   //[8] out
        Map<String, INDArray> m = sd2.outputAll(null);
        assertArrayEquals(new long[]{8}, m.get(meanB.name()).shape());

        assertArrayEquals(new long[]{9, 8}, m.get(meanA.name()).shape());
        assertArrayEquals(new long[]{8}, m.get(meanB.name()).shape());

        m = sd2.outputAll(null);

        INDArray mA = m.get(meanA.name());
        INDArray mB = m.get(meanB.name());

        assertArrayEquals(new long[]{9, 8}, mA.shape());
        assertArrayEquals(new long[]{8}, mB.shape());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRunLogisticRegression(Nd4jBackend backend) {
        Map<String, INDArray> vars = this.variablesForInput();
        SameDiff outside = SameDiff.create();
        outside.defineFunction("activate", (sameDiff, inputs, variableInputs) -> {
            sameDiff.enableDebugMode();
            SDVariable x = sameDiff.var("x", inputs.get("x"));
            SDVariable w = sameDiff.var("w", inputs.get("w"));
            SDVariable y = sameDiff.var("y", inputs.get("y"));
            SDVariable activation = sameDiff.nn().sigmoid("activation", sameDiff.mmul("mmul", x, w));
            SDVariable oneMinusY = y.rsub("oneminusy", 1.0);
            SDVariable oneMinusPredictions = activation.rsub("oneminusactivations", 1.0);
            SDVariable outputTimesY = y.mul("output * y", activation);
            SDVariable yHat = oneMinusPredictions.mul("yhat", oneMinusY);
            SDVariable probs = outputTimesY.add("probs", yHat);
            SDVariable logProbs = sameDiff.math().log("logprob", probs);
            SDVariable ret = sameDiff.sum("totalsum", logProbs, Integer.MAX_VALUE);
            SDVariable ret2 = sameDiff.math().neg("negtotalsum", ret);
            return new SDVariable[]{ret2};
        }, vars);

        SameDiff activation = outside.getFunction("activate");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDenseLayerForwardPass(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        SameDiff sd = SameDiff.create();

        INDArray iInput = Nd4j.rand(3, 4);
        INDArray iWeights = Nd4j.rand(4, 5);
        INDArray iBias = Nd4j.rand(1, 5);

        SDVariable input = sd.var("input", iInput);
        SDVariable weights = sd.var("weights", iWeights);
        SDVariable bias = sd.var("bias", iBias);

        SDVariable mmul = sd.mmul("mmul", input, weights);
        SDVariable z = mmul.add("z", bias);
        SDVariable out = sd.nn().sigmoid("out", z);

        INDArray expMmul = iInput.mmul(iWeights);
        INDArray expZ = expMmul.addRowVector(iBias);
        INDArray expOut = Transforms.sigmoid(expZ, true);

        Map<String, INDArray> m = sd.outputAll(Collections.emptyMap());

        assertEquals(expMmul, m.get(mmul.name()));
        assertEquals(expZ, m.get(z.name()));
        assertEquals(expOut, m.get(out.name()));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConvertToConstant(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("in", DataType.FLOAT, 1, 3);
        SDVariable w = sd.var("w", Nd4j.rand(DataType.FLOAT, 3, 4));
        SDVariable b = sd.var("b", Nd4j.rand(DataType.FLOAT, 1, 4));
        SDVariable mmul = in.mmul(w);
        SDVariable add = mmul.add(b);
        SDVariable tanh = sd.math().tanh(add);
        SDVariable loss = sd.variance(tanh, true);
        loss.markAsLoss();
        INDArray inArr = Nd4j.rand(DataType.FLOAT, 1, 3);
        in.setArray(inArr);

        TrainingConfig c = TrainingConfig.builder()
                .updater(new Adam(0.1))
                .weightDecay(0.01, true)
                .dataSetFeatureMapping("in")
                .skipBuilderValidation(true)
                .build();
        sd.setTrainingConfig(c);

        sd.fit(new SingletonMultiDataSetIterator(new DataSet(inArr, null).toMultiDataSet()), 1);

        INDArray out = tanh.eval();

        w.convertToConstant();

        INDArray out2 = tanh.eval();

        assertEquals(out, out2);
        assertEquals(VariableType.CONSTANT, w.getVariableType());
        assertEquals(VariableType.VARIABLE, b.getVariableType());
        assertEquals(VariableType.ARRAY, add.getVariableType());
        assertEquals(VariableType.ARRAY, tanh.getVariableType());

        sd.fit(new SingletonMultiDataSetIterator(new DataSet(inArr, null).toMultiDataSet()), 1);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPlaceholderToConstant(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("in", DataType.FLOAT, 1, 3);
        SDVariable in2 = sd.placeHolder("in2", DataType.FLOAT, 3, 4);
        SDVariable b = sd.var("b", Nd4j.rand(DataType.FLOAT, 1, 4));
        SDVariable mmul = in.mmul(in2);
        SDVariable add = mmul.add(b);
        SDVariable tanh = sd.math().tanh(add);
        SDVariable loss = sd.variance(tanh, true);

        INDArray inArr = Nd4j.rand(DataType.FLOAT, 1, 3);
        in.setArray(inArr);
        INDArray inArr2 = Nd4j.rand(DataType.FLOAT, 3, 4);
        in2.setArray(inArr2);
        loss.markAsLoss();
        TrainingConfig c = TrainingConfig.builder()
                .updater(new Adam(0.1))
                .weightDecay(0.01, true)
                .dataSetFeatureMapping("in", "in2")
                .skipBuilderValidation(true)
                .build();
        sd.setTrainingConfig(c);

        sd.fit(new SingletonMultiDataSetIterator(new MultiDataSet(new INDArray[]{inArr, inArr2}, null)), 1);

        INDArray out = tanh.eval();

        in.convertToConstant();

        INDArray out2 = tanh.eval();

        assertEquals(out, out2);
        assertEquals(VariableType.CONSTANT, in.getVariableType());
        assertEquals(inArr, in.getArr());

        sd.fit(new SingletonMultiDataSetIterator(new MultiDataSet(new INDArray[]{inArr2}, null)), 1);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConvertToVariable(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("in", DataType.FLOAT, 1, 3);
        INDArray const1 = Nd4j.rand(DataType.FLOAT, 3, 4);
        SDVariable w = sd.constant("w", const1);
        SDVariable b = sd.var("b", Nd4j.rand(DataType.FLOAT, 1, 4));
        SDVariable mmul = in.mmul(w);
        SDVariable add = mmul.add(b);
        SDVariable tanh = sd.math().tanh(add);
        SDVariable loss = sd.variance(tanh, true);
        loss.markAsLoss();
        INDArray inArr = Nd4j.rand(DataType.FLOAT, 1, 3);
        in.setArray(inArr);

        TrainingConfig c = TrainingConfig.builder()
                .updater(new Adam(0.1))
                .weightDecay(0.01, true)
                .dataSetFeatureMapping("in")
                .skipBuilderValidation(true)
                .build();
        sd.setTrainingConfig(c);

        INDArray out = tanh.eval();
        sd.fit(new SingletonMultiDataSetIterator(new DataSet(inArr, null).toMultiDataSet()), 1);
        w.convertToVariable();

        INDArray out2 = tanh.eval();

        assertNotEquals(out, out2);
        assertEquals(VariableType.VARIABLE, w.getVariableType());
        assertEquals(VariableType.VARIABLE, b.getVariableType());
        assertEquals(VariableType.ARRAY, add.getVariableType());
        assertEquals(VariableType.ARRAY, tanh.getVariableType());

        sd.fit(new SingletonMultiDataSetIterator(new DataSet(inArr, null).toMultiDataSet()), 1);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPlaceholderShapeValidation(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable scalar = sd.scalar("scalar", 0.0f);
        SDVariable ph1 = sd.placeHolder("ph1", DataType.FLOAT, 3, 4);
        SDVariable ph2 = sd.placeHolder("ph2", DataType.FLOAT, -1, 4);
        SDVariable ph3 = sd.placeHolder("ph3", DataType.FLOAT, 3, -1);
        SDVariable ph4 = sd.placeHolder("ph4", DataType.FLOAT, -1, -1);

        INDArray correctShape = Nd4j.create(DataType.FLOAT, 3, 4);
        INDArray wrongShape = Nd4j.create(DataType.FLOAT, 2, 3);
        INDArray wrongRank1 = Nd4j.create(DataType.FLOAT, 1);
        INDArray wrongRank2 = Nd4j.create(DataType.FLOAT, 3, 4, 5);
        for (SDVariable v : new SDVariable[]{ph1, ph2, ph3, ph4}) {
            v.setArray(correctShape);

            if (v != ph4) {
                try {
                    v.setArray(wrongShape);
                    fail("Expected exception");
                } catch (Exception t) {
                    String msg = t.getMessage();
                    assertTrue(msg.contains("shape") && msg.contains("[2, 3]") && msg
                            .contains(Arrays.toString(v.placeholderShape())), msg);
                }
            }

            try {
                v.setArray(wrongRank1);
                fail("Expected exception");
            } catch (Exception t) {
                String msg = t.getMessage();
                assertTrue(msg.contains("shape") && msg.contains("[1]") && msg
                        .contains(Arrays.toString(v.placeholderShape())), msg);
            }

            try {
                v.setArray(wrongRank2);
                fail("Expected exception");
            } catch (Exception t) {
                String msg = t.getMessage();
                assertTrue(msg.contains("shape") && msg.contains("[3, 4, 5]") && msg
                        .contains(Arrays.toString(v.placeholderShape())), msg);
            }
        }

        SDVariable sum = sd.math.mergeAdd(new SDVariable[]{ph1, ph2, ph3, ph4});
        SDVariable mean = sum.add(scalar).mean();
        mean.markAsLoss();
        MultiDataSet mds = new MultiDataSet(new INDArray[]{wrongShape, wrongShape, wrongShape, wrongShape}, null);

        sd.setTrainingConfig(TrainingConfig.builder()
                .dataSetFeatureMapping("ph1", "ph2", "ph3", "ph4")
                .markLabelsUnused()
                .updater(new Adam(1e-3)).build());

        Exception fitEx = assertThrows(Exception.class, () -> sd.fit(mds));
        String fitMsg = fitEx.getMessage();
        assertTrue(fitMsg.contains("shape") && fitMsg.contains("[2, 3]"), fitMsg);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testInferenceWithoutLabel(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();

        int nIn = 4;
        int minibatch = 3;
        SDVariable input = sd.placeHolder("in", DataType.FLOAT, -1, 4);
        SDVariable label = sd.placeHolder("label", DataType.FLOAT, -1, 3);

        SDVariable w = sd.var("w", Nd4j.rand(DataType.FLOAT, 4, 3));
        SDVariable b = sd.var("b", Nd4j.rand(DataType.FLOAT, 1, 3));

        SDVariable mmul = input.mmul(w).add(b);
        SDVariable softmax = sd.nn().softmax("softmax", mmul);
        SDVariable loss = sd.loss().logLoss("loss", label, softmax);

        INDArray inputArr = Nd4j.rand(DataType.FLOAT, minibatch, nIn);

        Map<String, INDArray> m = sd.output(Collections.singletonMap("in", inputArr), "softmax");
        assertEquals(1, m.size());
        assertTrue(m.containsKey("softmax"));

        INDArray out = m.get("softmax");

        INDArray labelUnused = Nd4j.rand(DataType.FLOAT, minibatch, 3);
        Map<String, INDArray> allPh = new HashMap<>();
        allPh.put("in", inputArr);
        allPh.put("label", labelUnused);
        m = sd.output(allPh, "softmax");
        assertEquals(1, m.size());
        assertTrue(m.containsKey("softmax"));
        INDArray out2 = m.get("softmax");
        assertEquals(out, out2);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testInferenceWithoutUnnecessaryPlaceholders(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();

        int nIn = 4;
        int minibatch = 3;
        SDVariable input = sd.placeHolder("in", DataType.FLOAT, -1, 4);
        SDVariable label = sd.placeHolder("label", DataType.FLOAT, -1, 3);

        SDVariable input2 = sd.placeHolder("in2", DataType.FLOAT);

        SDVariable w = sd.var("w", Nd4j.rand(DataType.FLOAT, 4, 3));
        SDVariable b = sd.var("b", Nd4j.rand(DataType.FLOAT, 1, 3));

        SDVariable mmul = input.mmul(w).add(b);
        SDVariable softmax = sd.nn().softmax("softmax", mmul);
        SDVariable loss = sd.loss().logLoss("loss", label, softmax);
        SDVariable loss2 = softmax.mul(input2);

        INDArray inputArr = Nd4j.rand(DataType.FLOAT, minibatch, nIn);

        Map<String, INDArray> m = sd.output(Collections.singletonMap("in", inputArr), "softmax");
        assertEquals(1, m.size());
        assertTrue(m.containsKey("softmax"));

        INDArray out = m.get("softmax");

        INDArray labelUnused = Nd4j.rand(DataType.FLOAT, minibatch, 3);
        Map<String, INDArray> allPh = new HashMap<>();
        allPh.put("in", inputArr);
        allPh.put("label", labelUnused);
        allPh.put("in2", Nd4j.scalar(1.0f));
        m = sd.output(allPh, "softmax");
        assertEquals(1, m.size());
        assertTrue(m.containsKey("softmax"));
        INDArray out2 = m.get("softmax");
        assertEquals(out, out2);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConvertDTypes1(Nd4jBackend backend) {

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.var("x", Nd4j.rand(DataType.FLOAT, 3, 4));
        SDVariable y = sd.var("y", Nd4j.rand(DataType.FLOAT, 4, 2));
        SDVariable z = x.mmul("z", y);
        SDVariable tanh = sd.math().tanh("tanh", z);
        SDVariable stdev = tanh.std("stdev", true);

        assertEquals(DataType.FLOAT, x.dataType());
        assertEquals(DataType.FLOAT, y.dataType());
        assertEquals(DataType.FLOAT, z.dataType());
        assertEquals(DataType.FLOAT, tanh.dataType());
        assertEquals(DataType.FLOAT, stdev.dataType());

        Map<String, INDArray> out = sd.output((Map<String, INDArray>) null, "x", "y", "z", "tanh", "stdev");
        for (Map.Entry<String, INDArray> e : out.entrySet()) {
            assertEquals(DataType.FLOAT, e.getValue().dataType(), e.getKey());
        }

        assertEquals(DataType.FLOAT, x.getArr().dataType());
        assertEquals(DataType.FLOAT, y.getArr().dataType());

        Map<String, DataType> toConvert = new HashMap<>();
        toConvert.put("x", DataType.DOUBLE);
        toConvert.put("y", DataType.DOUBLE);
        sd.convertDataTypes(toConvert);

        assertEquals(DataType.DOUBLE, x.dataType());
        assertEquals(DataType.DOUBLE, y.dataType());
        assertEquals(DataType.DOUBLE, z.dataType());
        assertEquals(DataType.DOUBLE, tanh.dataType());
        assertEquals(DataType.DOUBLE, stdev.dataType());

        out = sd.output((Map<String, INDArray>) null, "x", "y", "z", "tanh", "stdev");
        for (Map.Entry<String, INDArray> e : out.entrySet()) {
            assertEquals(DataType.DOUBLE, e.getValue().dataType(), e.getKey());
        }

        assertEquals(DataType.DOUBLE, x.getArr().dataType());
        assertEquals(DataType.DOUBLE, y.getArr().dataType());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConvertDTypes2(Nd4jBackend backend) {

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
        SDVariable y = sd.var("y", Nd4j.rand(DataType.FLOAT, 1, 4));
        SDVariable xD = x.castTo("xD", DataType.DOUBLE);
        SDVariable yD = y.castTo("yD", DataType.DOUBLE);
        SDVariable add = xD.add("a", yD);
        SDVariable relu = sd.nn().relu("r", add, 1);

        assertEquals(DataType.FLOAT, x.dataType());
        assertEquals(DataType.FLOAT, y.dataType());
        assertEquals(DataType.DOUBLE, xD.dataType());
        assertEquals(DataType.DOUBLE, yD.dataType());
        assertEquals(DataType.DOUBLE, add.dataType());
        assertEquals(DataType.DOUBLE, relu.dataType());

        Map<String, INDArray> ph = Collections.singletonMap("x", Nd4j.rand(DataType.FLOAT, 3, 4));

        Map<String, INDArray> out = sd.output(ph, "x", "y", "xD", "yD", "a", "r");
        for (Map.Entry<String, INDArray> e : out.entrySet()) {
            if (e.getKey().equals("x") || e.getKey().equals("y")) {
                assertEquals(DataType.FLOAT, e.getValue().dataType(), e.getKey());
            } else {
                assertEquals(DataType.DOUBLE, e.getValue().dataType(), e.getKey());
            }
        }

        assertEquals(DataType.FLOAT, y.getArr().dataType());

        Map<String, DataType> toConvert = new HashMap<>();
        toConvert.put("x", DataType.DOUBLE);
        toConvert.put("y", DataType.DOUBLE);
        sd.convertDataTypes(toConvert);

        assertEquals(DataType.DOUBLE, x.dataType());
        assertEquals(DataType.DOUBLE, y.dataType());
        assertEquals(DataType.DOUBLE, xD.dataType());
        assertEquals(DataType.DOUBLE, yD.dataType());
        assertEquals(DataType.DOUBLE, add.dataType());
        assertEquals(DataType.DOUBLE, relu.dataType());

        // After convertDataTypes, x is now DOUBLE — supply a DOUBLE placeholder
        Map<String, INDArray> phDouble = Collections.singletonMap("x", Nd4j.rand(DataType.DOUBLE, 3, 4));
        out = sd.output(phDouble, "x", "y", "xD", "yD", "a", "r");
        for (Map.Entry<String, INDArray> e : out.entrySet()) {
            assertEquals(DataType.DOUBLE, e.getValue().dataType(), e.getKey());
        }

        assertEquals(DataType.DOUBLE, y.getArr().dataType());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCastLongToIntWithModelSaveLoad(Nd4jBackend backend) throws Exception {
        SameDiff sd = SameDiff.create();
        SDVariable inputIds = sd.placeHolder("input_ids", DataType.LONG, -1, -1);
        SDVariable inputIdsInt32 = inputIds.castTo("input_ids_int32", DataType.INT);

        File tempFile = File.createTempFile("cast_test_model", ".fb");
        tempFile.deleteOnExit();
        sd.asFlatFile(tempFile);

        SameDiff loaded = SameDiff.fromFlatFile(tempFile);
        assertNotNull(loaded);

        INDArray input = Nd4j.zeros(DataType.LONG, 1, 512);
        input.putScalar(0, 0, 101);
        input.putScalar(0, 1, 2023);
        input.putScalar(0, 2, 2003);

        Map<String, INDArray> placeholders = Collections.singletonMap("input_ids", input);
        Map<String, INDArray> outputs = loaded.output(placeholders, "input_ids_int32");

        INDArray result = outputs.get("input_ids_int32");
        assertNotNull(result);
        assertEquals(DataType.INT, result.dataType());
        assertArrayEquals(new long[]{1, 512}, result.shape());
        assertEquals(101, result.getInt(0, 0));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSameDiffSeedReproducibilityVarInit(Nd4jBackend backend) {

        SameDiff sd0 = SameDiff.create();
        SameDiff sd1 = SameDiff.create();
        Nd4j.getRandom().setSeed(12345);
        SDVariable rand0 = sd0.var("random", new UniformInitScheme('c', 3), DataType.FLOAT, 3, 1);

        Nd4j.getRandom().setSeed(12345);
        SDVariable rand1 = sd1.var("random", new UniformInitScheme('c', 3), DataType.FLOAT, 3, 1);

        INDArray a0 = rand0.eval();
        Nd4j.getRandom().setSeed(0);
        INDArray a1 = rand1.eval();
        assertEquals(a0, a1);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTrainingConfigJson(Nd4jBackend backend) {
        for (IEvaluation e : new IEvaluation[]{new Evaluation(), new RegressionEvaluation(), new EvaluationBinary(), new ROC(),
                new ROCMultiClass(), new ROCBinary(), new EvaluationCalibration()}) {
            TrainingConfig config = TrainingConfig.builder()
                    .l2(1e-4)
                    .updater(new Adam(0.1))
                    .dataSetFeatureMapping("out").dataSetLabelMapping("label")
                    .trainEvaluation("out", 0, e)
                    .build();
            String json = config.toJson();
            TrainingConfig fromJson = TrainingConfig.fromJson(json);
            assertEquals(config, fromJson);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRngSanityCheck(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);
        for (DataType dt : new DataType[]{DataType.FLOAT, DataType.DOUBLE, DataType.BFLOAT16}) {
            if (!dt.isNumerical())
                continue;
            SameDiff sameDiff = SameDiff.create();
            INDArray indaShape = Nd4j.createFromArray(3, 10);
            SDVariable sdShape = sameDiff.constant(indaShape);
            SDVariable random = sameDiff.random().uniform("data", 0.0, 10.0, dt, 3, 10);
            INDArray out = random.eval();
            String s = out.toString();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMissingPlaceholderError(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();

        int nOut = 4;
        int minibatch = 10;
        SDVariable predictions = sd.var("in", DataType.DOUBLE, minibatch, nOut);
        SDVariable labels = sd.placeHolder("labels", DataType.DOUBLE, -1, nOut);

        org.nd4j.autodiff.loss.LossReduce reduction = org.nd4j.autodiff.loss.LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT;

        SDVariable loss = sd.loss().absoluteDifference("loss", labels, predictions, null, reduction);

        try {
            loss.eval();
            fail("Exception should have been thrown");
        } catch (Exception e) {
            String msg = e.getMessage();
            assertTrue(msg.contains("labels"), msg);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEquals1(Nd4jBackend backend) {

        SameDiff sd1 = SameDiff.create();
        SameDiff sd2 = SameDiff.create();

        assertEquals(sd1, sd2);

        SDVariable p1 = sd1.placeHolder("ph", DataType.FLOAT, -1, 10);
        SDVariable p2 = sd2.placeHolder("ph", DataType.FLOAT, -1, 10);

        assertEquals(sd1, sd2);

        SDVariable w1 = sd1.constant("c1", 1.0f);
        SDVariable w2 = sd2.constant("c1", 1.0f);

        assertEquals(sd1, sd2);

        SDVariable a1 = p1.add("add", w1);
        SDVariable a2 = p2.add("add", w2);

        assertEquals(sd1, sd2);

        SDVariable w1a = sd1.constant("c2", 2.0f);
        SDVariable w2a = sd2.constant("cX", 2.0f);

        assertNotEquals(sd1, sd2);
        w2a.rename("c2");

        assertEquals(sd1, sd2);

        sd2.createGradFunction("ph");

        assertEquals(sd1, sd2);

        w2a.getArr().assign(3.0f);

        assertNotEquals(sd1, sd2);

        w1a.getArr().assign(3.0f);
        assertEquals(sd1, sd2);

        SDVariable s1 = p1.sub("op", w1);
        SDVariable s2 = p2.add("op", w1);
        assertNotEquals(sd1, sd2);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConv2DWeightsFormat(Nd4jBackend backend) {
        int bS = 2, iH = 4, iW = 3, iC = 4, oC = 3, kH = 3, kW = 2, sH = 1, sW = 1, pH = 0, pW = 0, dH = 1, dW = 1;
        int oH = 2, oW = 2;
        SameDiff sd = SameDiff.create();

        WeightsFormat format = WeightsFormat.OIYX;

        INDArray inArr = Nd4j.linspace(DataType.FLOAT, 25, -0.5, 96).reshape(new long[]{bS, iC, iH, iW});
        INDArray weights = Nd4j.createFromArray(new float[]{
                        -3.f, -1.8f, -0.6f, 0.6f, 1.8f, 3.f, -2.7f, -1.5f, -0.3f, 0.9f, 2.1f, 3.3f, -2.4f, -1.2f, 0.f, 1.2f, 2.4f, 3.6f, -2.1f, -0.9f, 0.3f, 1.5f,
                        2.7f, 3.9f, -2.9f, -1.7f, -0.5f, 0.7f, 1.9f, 3.1f, -2.6f, -1.4f, -0.2f, 1.f, 2.2f, 3.4f, -2.3f, -1.1f, 0.1f, 1.3f, 2.5f, 3.7f, -2.f, -0.8f, 0.4f, 1.6f,
                        2.8f, 4.f, -2.8f, -1.6f, -0.4f, 0.8f, 2.f, 3.2f, -2.5f, -1.3f, -0.1f, 1.1f, 2.3f, 3.5f, -2.2f, -1.f, 0.2f, 1.4f, 2.6f, 3.8f, -1.9f, -0.7f, 0.5f, 1.7f, 2.9f, 4.1f}).
                reshape(new long[]{oC, iC, kH, kW});

        INDArray bias = Nd4j.createFromArray(new float[]{-1, 2, 0.5f});

        SDVariable sdInput = sd.var("in", inArr);
        SDVariable sdWeights = sd.var("dW", weights);
        SDVariable sdBias = sd.var("b", bias);

        Conv2DConfig c = Conv2DConfig.builder()
                .kH(kH).kW(kW)
                .pH(pH).pW(pW)
                .sH(sH).sW(sW)
                .dH(dH).dW(dW)
                .paddingMode(PaddingMode.VALID)
                .weightsFormat(format)
                .build();

        SDVariable out = sd.cnn().conv2d(sdInput, sdWeights, sdBias, c);

        assertArrayEquals(new long[]{bS, oC, oH, oW}, out.eval().shape());
    }
}
