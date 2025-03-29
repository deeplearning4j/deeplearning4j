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

import static org.deeplearning4j.datasets.iterator.RandomDataSetIterator.Values.*;
import static org.deeplearning4j.datasets.iterator.RandomDataSetIterator.Values.INTEGER_0_10;
import static org.junit.jupiter.api.Assertions.*;
import static org.nd4j.linalg.api.buffer.DataType.FLOAT;
import static org.nd4j.linalg.indexing.NDArrayIndex.all;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.util.*;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.writable.IntWritable;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.RandomDataSetIterator;
import org.deeplearning4j.datasets.iterator.ReconstructionDataSetIterator;
import org.junit.jupiter.api.*;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.autodiff.samediff.*;
import org.nd4j.autodiff.samediff.api.OutAndGrad;
import org.nd4j.autodiff.util.SameDiffUtils;
import org.nd4j.autodiff.validation.OpValidation;
import org.nd4j.autodiff.validation.TestCase;
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
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.ExternalErrorsFunction;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.LocalResponseNormalizationConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.PaddingMode;
import org.nd4j.linalg.api.ops.impl.reduce3.ManhattanDistance;
import org.nd4j.linalg.api.ops.impl.shape.CreateView;
import org.nd4j.linalg.api.ops.impl.shape.tensorops.TensorArray;
import org.nd4j.linalg.api.ops.impl.transforms.any.IsMax;
import org.nd4j.linalg.api.ops.impl.transforms.custom.GreaterThanOrEqual;
import org.nd4j.linalg.api.ops.impl.transforms.custom.IsNonDecreasing;
import org.nd4j.linalg.api.ops.impl.transforms.custom.IsNumericTensor;
import org.nd4j.linalg.api.ops.impl.transforms.custom.IsStrictlyIncreasing;
import org.nd4j.linalg.api.ops.impl.transforms.custom.LessThanOrEqual;
import org.nd4j.linalg.api.ops.impl.transforms.custom.Max;
import org.nd4j.linalg.api.ops.impl.transforms.custom.Min;
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.checkutil.NDArrayCreationUtil;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.adapter.SingletonDataSetIterator;
import org.nd4j.linalg.dataset.adapter.SingletonMultiDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.common.primitives.Pair;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.shade.guava.collect.Maps;
import org.nd4j.weightinit.impl.OneInitScheme;
import org.nd4j.weightinit.impl.UniformInitScheme;
import org.nd4j.weightinit.impl.XavierInitScheme;
import org.nd4j.weightinit.impl.ZeroInitScheme;

@Slf4j
@NativeTag
@Tag(TagNames.SAMEDIFF)
public class SameDiffTests extends BaseNd4jTestWithBackends {



    @Override
    public char ordering() {
        return 'c';
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Disabled("Can mess with global tests. Should only be run in isolation.")
    public void testOpExecTrace(Nd4jBackend backend) {
        Nd4j.toggleTrace(true);
        final INDArray input = Nd4j.linspace(1,4,4).reshape(2,2);

        SameDiff sd = SameDiff.create();
        SDVariable input2 = sd.var("input", input);


        SDVariable t = sd.nn.softmax(input2,1);

        Nd4j.getExecutioner().enableDebugMode(true);
        Nd4j.getExecutioner().enableVerboseMode(true);
        sd.calculateGradients(Collections.emptyMap(), Collections.singleton("input"));
        SameDiff traced  = SameDiff.collectTrace();
        assertTrue(traced.ops().length > 0);
        System.out.println(traced.summary());
        Nd4j.purgeTrace();
        assertTrue(NativeOpsHolder.getInstance().getDeviceNativeOps().listOpTraces() == null);
        Nd4j.toggleTrace(false);
    }




    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCtc(Nd4jBackend backend) {
        Nd4j.getExecutioner().enableVerboseMode(true);
        Nd4j.getExecutioner().enableDebugMode(true);
        SameDiff sd = SameDiff.create();
        INDArray labelsND = Nd4j.createFromArray(new int[][] {{1917, 3468, 1024, 2744, 4092, 2613,  112,  922, 4785, 1675},
                {119, 16, 202, 2352, 2945, 3468, 2744, 112, 0, 0}});
        INDArray labels_len_ND =  Nd4j.createFromArray(new int[] {10, 8});
        INDArray logits_len_ND =  Nd4j.createFromArray(new int[] {155, 155});
        SDVariable labels = sd.constant(labelsND);
        SDVariable logits = sd.random.normal(0, 1, DataType.FLOAT, new long[] {2, 155, 5530});
        SDVariable labels_len = sd.constant(labels_len_ND);
        SDVariable logits_len = sd.constant(logits_len_ND);
        SDVariable ctc = sd.loss.ctcLoss("ctcLoss", labels, logits, labels_len, logits_len);
        //
        System.out.println(ctc.eval());
    }



    @Override
    public long getTimeoutMilliseconds() {
        return 999999999L;
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

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLinearEquivalency(Nd4jBackend backend) {
        int batchSize = 32;
        int modelDim = 10;

        DataSetIterator iterator = new ReconstructionDataSetIterator(new RandomDataSetIterator(100, new long[]{batchSize, modelDim}, new long[]{}, ONE_HOT, ZEROS));
        DataSet next = iterator.next();
        assertEquals(testLinearLayers(true,batchSize,modelDim,next),testLinearLayers(false,batchSize,modelDim,next));
        assertEquals(testLinearLayersManual(true,batchSize,modelDim,next),testLinearLayersManual(false,batchSize,modelDim,next));

    }

    private INDArray testLinearLayers(boolean relu, int batchSize, int modelDim, DataSet dataInput) {
        SameDiff sd = SameDiff.create();
        DataSetIterator data = new SingletonDataSetIterator(dataInput);
        SDVariable features = sd.placeHolder("features", FLOAT, batchSize, modelDim);
        SDVariable labels = sd.placeHolder("labels", FLOAT, batchSize, modelDim);
        SDVariable weights = sd.var("weights", new OneInitScheme('c'), FLOAT, modelDim, modelDim);
        SDVariable bias = sd.zero("bias", FLOAT,modelDim);
        SDVariable predictions = relu?  sd.nn.reluLayer("predictions", features, weights, bias) : sd.nn.linear("predictions", features, weights, bias);       // <<< variant 2 (doesn't work)
        sd.loss.meanSquaredError("loss", labels, predictions, null);

        TrainingConfig config = new TrainingConfig.Builder()
                .updater(new Adam(0.1))
                .dataSetFeatureMapping("features")
                .dataSetLabelMapping("labels")
                .build();
        sd.setTrainingConfig(config);

// the task is to reconstruct the one-hot encoded input

        sd.fit(data, 10);

        Evaluation evaluation = new Evaluation();
        sd.evaluate(data, "predictions", evaluation);

        return sd.getVariable("predictions").eval(Collections.singletonMap("features",dataInput.getFeatures()));
    }


    private INDArray testLinearLayersManual(boolean manual, int batchSize, int modelDim, DataSet dataInput) {
        SameDiff sd = SameDiff.create();
        DataSetIterator data = new SingletonDataSetIterator(dataInput);
        SDVariable features = sd.placeHolder("features", FLOAT, batchSize, modelDim);
        SDVariable labels = sd.placeHolder("labels", FLOAT, batchSize, modelDim);
        SDVariable weights = sd.var("weights", new OneInitScheme('c'), FLOAT, modelDim, modelDim);
        SDVariable bias = sd.zero("bias", FLOAT,modelDim);
        SDVariable predictions = manual?  features.mmul(weights).add("predictions", bias) : sd.nn.linear("predictions", features, weights, bias);       // <<< variant 2 (doesn't work)
        sd.loss.meanSquaredError("loss", labels, predictions, null);

        TrainingConfig config = new TrainingConfig.Builder()
                .updater(new Adam(0.1))
                .dataSetFeatureMapping("features")
                .dataSetLabelMapping("labels")
                .build();
        sd.setTrainingConfig(config);

// the task is to reconstruct the one-hot encoded input

        sd.fit(data, 10);

        Evaluation evaluation = new Evaluation();
        sd.evaluate(data, "predictions", evaluation);

        return sd.getVariable("predictions").eval(Collections.singletonMap("features",dataInput.getFeatures()));
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSameDiffShapeNonNumerical() {
        SameDiff sd = SameDiff.create();
        SDVariable var = sd.create(null, sd.constant(8), DataType.BOOL);
        assertEquals(8,var.shape().eval().getLong(0)); // throws exception    }
        sd.setShape(var,var.shape())[0].eval();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSameDiffCreate() {
        SameDiff sd = SameDiff.create();
        SDVariable var = sd.create(null, sd.constant(8), DataType.INT32);
        assertEquals(DataType.INT, var.eval().dataType());
        assertEquals(DataType.INT,var.dataType());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testVariableNaming_1(Nd4jBackend backend) {
        val sd = SameDiff.create();

        val input = sd.var("inp", new long[]{2, 3});

        val nodeA = sd.math().square(input);
        val nodeB = sd.math().square(nodeA);

        sd.associateArrayWithVariable(Nd4j.create(new double[]{1, 2, 3, 4, 5, 6}, new long[]{2, 3}).castTo(input.dataType()), input);

        sd.outputAll(null);

        nodeA.isPlaceHolder();
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAddArgsAndOutput(Nd4jBackend backend) {
        SameDiff sameDiff = SameDiff.create();
        val varOne = sameDiff.var("one", Nd4j.ones(2));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMseBackwards(Nd4jBackend backend) {

        SameDiff sd = SameDiff.create();

        int nOut = 4;
        int minibatch = 3;
        SDVariable input = sd.var("in", DataType.FLOAT, new long[]{minibatch, nOut});
        SDVariable label = sd.var("label", DataType.FLOAT, new long[]{minibatch, nOut});

        SDVariable diff = input.sub(label);
        SDVariable sqDiff = diff.mul(diff);
        SDVariable msePerEx = sd.mean("msePerEx", sqDiff, 1);
        SDVariable avgMSE = sd.mean("loss", msePerEx, 0);

        INDArray inputArr = Nd4j.rand(DataType.FLOAT, minibatch, nOut);
        INDArray labelArr = Nd4j.rand(DataType.FLOAT, minibatch, nOut);

        sd.associateArrayWithVariable(inputArr, input);
        sd.associateArrayWithVariable(labelArr, label);

        INDArray result = avgMSE.eval();
        assertEquals(1, result.length());

        sd.calculateGradients(Collections.emptyMap(), sd.getVariables().keySet());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEvalVariable(Nd4jBackend backend) {
        SameDiff sameDiff = SameDiff.create();
        INDArray ones = Nd4j.ones(4);
        INDArray twos = ones.add(ones);
        SDVariable inputOne = sameDiff.var("inputone", ones);
        SDVariable inputResult = inputOne.add("extravarname", inputOne);
        assertEquals(twos, inputResult.eval());
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSum(Nd4jBackend backend) {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1, 4, 4, DataType.FLOAT)).reshape(1, 4);
        SDVariable x = sameDiff.var("x", arr);
        SDVariable result = sameDiff.sum(x, 1); //[1,4].sum(1) == [1]

        INDArray exp = Nd4j.scalar(arr.sumNumber().floatValue()).reshape(1);
        INDArray resultArr = result.eval();
        assertEquals(exp, resultArr);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAddEval(Nd4jBackend backend) {
        SameDiff sameDiff = SameDiff.create();
        INDArray x = Nd4j.scalar(1.0);
        INDArray y = Nd4j.scalar(2.0);
        SDVariable xVar = sameDiff.placeHolder("x", DataType.DOUBLE, 1, 1);
        SDVariable yVar = sameDiff.placeHolder("y", DataType.DOUBLE, 1, 1);
        SDVariable output = xVar.add(yVar);
        Map<String, INDArray> m = new HashMap<>();
        m.put("x", x);
        m.put("y", y);
        INDArray out = sameDiff.output(m, Collections.singletonList(output.name())).get(output.name());
        INDArray outputAssertion = x.add(y);
        assertEquals(outputAssertion, out);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMseForward(Nd4jBackend backend) {

        SameDiff sd = SameDiff.create();

        int nOut = 4;
        int minibatch = 3;
        SDVariable input = sd.var("in", new long[]{-1, nOut});
        SDVariable label = sd.var("label", new long[]{-1, nOut});

        SDVariable diff = input.sub(label);
        SDVariable sqDiff = diff.mul(diff);
        SDVariable msePerEx = sd.mean("msePerEx", sqDiff, 1);
        SDVariable score = sd.mean("score", msePerEx);

        INDArray inputArr = Nd4j.rand(minibatch, nOut);
        INDArray labelArr = Nd4j.rand(minibatch, nOut);

        sd.associateArrayWithVariable(inputArr, input);
        sd.associateArrayWithVariable(labelArr, label);

        INDArray result = score.eval();
        assertNotNull(result);                          //*** Fails Here - Null output ***
        assertEquals(1, result.length());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDistance(Nd4jBackend backend) {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1, 4, 4)).reshape(2, 2);
        SDVariable x = sameDiff.var("x", arr);
        SDVariable y = sameDiff.var("y", arr);
        SDVariable result = sameDiff.math().cosineSimilarity(x, y, 1);
        SDVariable addResult = result.add(result);
        SDVariable finalReshape = sameDiff.reshape(addResult, 1, 2);
        Map<String,INDArray> out = sameDiff.output(Collections.emptyMap(), finalReshape.name());
        assertArrayEquals(new long[]{1, 2}, out.get(finalReshape.name()).shape());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTensorGradMmul(Nd4jBackend backend) {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1, 4, 4)).reshape(2, 2);
        SDVariable x = sameDiff.var("x", arr);
        SDVariable y = sameDiff.var("y", arr);
        SDVariable result = sameDiff.mmul(x, y);
        SDVariable otherResult = result.add(result);
        Map<String,INDArray> m = sameDiff.outputAll(null);
        assertArrayEquals(new long[]{2, 2}, m.get(result.name()).shape());
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEval(Nd4jBackend backend) {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Nd4j.linspace(1, 4, 4);
        SDVariable x = sameDiff.var("x", arr);
        SDVariable sigmoid = sameDiff.nn().sigmoid("s", x);
        INDArray assertion = Transforms.sigmoid(arr);
        INDArray eval = sameDiff.output(Collections.singletonMap("x", arr), Collections.singletonList("s")).get("s");
        assertEquals(assertion, eval);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testFunctionInputsAndArgs(Nd4jBackend backend) {
        SameDiff sameDiff = SameDiff.create();
        SDVariable var = sameDiff.var("one", Nd4j.scalar(1.0));
        SDVariable variable2 = sameDiff.var("two", Nd4j.scalar(1.0));
        val sum = var.add(variable2);
        INDArray out = sum.eval();
        assertArrayEquals(new long[0], out.shape());
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCrossSameDiffVariableInitWithAlloc(Nd4jBackend backend) {
        SameDiff first = SameDiff.create();
        SameDiff second = SameDiff.create();

        SDVariable firstVar = first.var("one", new long[]{2, 2});
        SDVariable secondVar = second.var(firstVar);
        assertEquals(firstVar.getArr(), secondVar.getArr());
        assertEquals(firstVar.name(), secondVar.name());
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCrossSameDiffVariableInitWithPlaceHolder(Nd4jBackend backend) {
        SameDiff first = SameDiff.create();
        SameDiff second = SameDiff.create();

        SDVariable firstVar = first.var("one", new long[]{2, 2});
        SDVariable secondVar = second.var(firstVar);
        assertNotNull(firstVar.getArr());

        assertEquals(firstVar.getArr(), secondVar.getArr());
        assertEquals(firstVar.name(), secondVar.name());
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testVariableArrayReference(Nd4jBackend backend) {
        SameDiff sameDiff = SameDiff.create();
        SDVariable arr = sameDiff.var("one", new long[]{2, 2});
        assertArrayEquals(new long[]{2, 2}, arr.getShape());
        assertNotNull(arr.getArr());
        assertArrayEquals(new long[]{2, 2}, arr.getArr().shape());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEvalAddSelf(Nd4jBackend backend) {
        /**
         * Note this test fails yet due to needing
         * to validate simple cases like x * x
         * matching number of inputs.
         */
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Nd4j.linspace(1, 4, 4);
        SDVariable x = sameDiff.var("x", arr);
        SDVariable s = x.mul("s", x);
        INDArray assertion = arr.mul(arr);
        INDArray eval = sameDiff.output(Collections.singletonMap("x", arr), Collections.singletonList("s")).get("s");
        assertEquals(assertion, eval);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEvalAdd(Nd4jBackend backend) {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Nd4j.linspace(1, 4, 4);
        INDArray yArr = arr.dup();
        SDVariable x = sameDiff.var("x", arr);
        SDVariable y = sameDiff.var("y", yArr);

        SDVariable sigmoid = x.mul(y);
        INDArray assertion = arr.mul(arr);
        Map<String, INDArray> vars = new HashMap<>();
        vars.put("x", arr);
        vars.put("y", yArr);
        INDArray eval = sameDiff.output(vars, Collections.singletonList(sigmoid.name())).get(sigmoid.name());
        assertEquals(assertion, eval);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDup(Nd4jBackend backend) {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1, 8, 8)).reshape(2, 2, 2);
        SDVariable x = sameDiff.var("x", arr);
        SDVariable y = sameDiff.var("y", arr);
        SameDiff tg2 = sameDiff.dup();
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testElementWiseDivAndRDiv(Nd4jBackend backend) {
        SameDiff sameDiff = SameDiff.create();
        INDArray ones = Nd4j.ones(4);
        INDArray toDivBy = Nd4j.valueArrayOf(4, 0.25);
        Map<String, INDArray> xAndY = new HashMap<>();
        xAndY.put("x", ones);
        xAndY.put("y", toDivBy);
        sameDiff.defineFunction("div", (sameDiff1, inputs, variableInputs) -> {
            SDVariable x = sameDiff1.var("x", inputs.get("x"));
            SDVariable y = sameDiff1.var("y", inputs.get("y"));
            return new SDVariable[]{x.div("out", y)};
        }, xAndY);

        sameDiff.defineFunction("rdiv", (sameDiff12, inputs, variableInputs) -> {
            SDVariable x = sameDiff12.var("x", inputs.get("x"));
            SDVariable y = sameDiff12.var("y", inputs.get("y"));
            return new SDVariable[]{x.rdiv("out", y)};
        }, xAndY);

        INDArray assertionForDiv = Nd4j.valueArrayOf(4, 4.0);
        INDArray assertionForRDiv = Nd4j.valueArrayOf(4, 0.25);
        assertEquals(assertionForDiv, sameDiff.getFunction("div").outputSingle(null, "out"));
        assertEquals(assertionForRDiv, sameDiff.getFunction("rdiv").outputSingle(null, "out"));

    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNegativeGradient(Nd4jBackend backend) {
        SameDiff sameDiff = SameDiff.create();
        INDArray ones = Nd4j.ones(4);
        Map<String, INDArray> xAndY = new HashMap<>();
        xAndY.put("x", ones);
        sameDiff.defineFunction("neg", (sameDiff1, inputs, variableInputs) -> {
            SDVariable x = sameDiff1.var("x", inputs.get("x"));
            return new SDVariable[]{sameDiff1.math().neg("out", x)};
        }, xAndY);

        INDArray assertionForDiv = Nd4j.valueArrayOf(4, -1);
        assertEquals(assertionForDiv, sameDiff.getFunction("neg").outputSingle(null, "out"));

    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSumOp(Nd4jBackend backend) {
        SameDiff sameDiff = SameDiff.create();
        INDArray sumInput = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put("x", sumInput);
        sameDiff.defineFunction("sum", (sameDiff1, inputs1, variableInputs) -> {
            SDVariable input = sameDiff1.var("x", inputs1.get("x"));
            SDVariable sum = sameDiff1.sum("sum", input, 1);
            return new SDVariable[]{sum};
        }, inputs);

        INDArray assertion = sumInput.sum(1);
        INDArray out = sameDiff.getFunction("sum").output(Collections.emptyMap(), Collections.singletonList("sum"))
                .get("sum");
        assertEquals(assertion, out);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testVariableReferenceNoFunction(Nd4jBackend backend) {
        /**
         * Creating a variable should not create a differential function.
         */
        SameDiff sameDiff = SameDiff.create();
        SDVariable sdVariable = sameDiff.var("one", Nd4j.scalar(1.0));
        assertNotNull(sameDiff.getVariable(sdVariable.name()));
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testVariableWithFunction(Nd4jBackend backend) {
        /**
         * A variable's function should be null
         * when just a variable but
         * have a function result
         * when the variable itself is the result of a function.
         *
         */
        SameDiff sameDiff = SameDiff.create();
        SDVariable sdVariable = sameDiff.var("one", Nd4j.scalar(1.0));
        SDVariable add = sdVariable.add(1.0);
        assertEquals(sameDiff.getVariable(add.name()), add);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testUpdateVariable(Nd4jBackend backend) {
        SameDiff sameDiff = SameDiff.create();
        SDVariable one = sameDiff.one("one", new long[]{1, 1});
        one.rename("one-diff");
        assertEquals(one.eval(), sameDiff.getVariable("one-diff").eval());
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDefineFunctionArrayExistence(Nd4jBackend backend) {
        SameDiff sameDiff = SameDiff.create();
        String testFunctionName = "testfunction";
        SDVariable[] inputVars = new SDVariable[]{
                sameDiff.var("one", new long[]{1, 1}),
                sameDiff.var("two", new long[]{1, 1}),

        };

        SameDiff functionDef = sameDiff.defineFunction(testFunctionName, (sameDiff1, inputs, variableInputs) -> new SDVariable[]{variableInputs[0].add(variableInputs[1])}, inputVars);

        //1 input plus 2 outputs
        assertEquals(3, functionDef.variables().size());


    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAutoBroadcastAddMatrixVector(Nd4jBackend backend) {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray row = Nd4j.ones(2);
        INDArray assertion = arr.add(1.0);
        SDVariable left = sameDiff.var("arr", arr);
        SDVariable right = sameDiff.var("row", row);
        SDVariable test = left.add(right);
        assertEquals(assertion, test.eval());
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNegativeOneShape(Nd4jBackend backend) {
        val sd = SameDiff.create();
        SDVariable var = sd.placeHolder("test", DataType.FLOAT, -1, 3);
        assertTrue(var.isPlaceHolder());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testShapeResolutionMinus1(Nd4jBackend backend) {
        int nIn = 3;
        int nOut = 4;

        int minibatch = 3;

        for (boolean useMinus1 : new boolean[]{false, true}) {
            log.info("Starting: {}", (useMinus1 ? "minibatch -1" : "minibatch 3"));

            long[] inShape;
            if (useMinus1) {
                inShape = new long[]{-1, nIn};
            } else {
                inShape = new long[]{minibatch, nIn};
            }
            val wShape = new long[]{nIn, nOut};
            val bShape = new long[]{1, nOut};

            SameDiff sd = SameDiff.create();
            SDVariable layerInput = sd.var("in", inShape);
            SDVariable weights = sd.var("W", wShape);
            SDVariable bias = sd.var("b", bShape);

            SDVariable mmul = sd.mmul("mmul", layerInput, weights);
            SDVariable z = mmul.add("z", bias);
            SDVariable out = sd.nn().sigmoid("out", z);

            Map<String, INDArray> m = new HashMap<>();
            INDArray in = Nd4j.rand(new long[]{minibatch, nIn});
            INDArray w = Nd4j.rand(wShape);
            INDArray b = Nd4j.rand(bShape);

            sd.associateArrayWithVariable(in, sd.getVariable("in"));
            assertNotNull(sd.getArrForVarName("in"));
            sd.associateArrayWithVariable(w, sd.getVariable("W"));
            sd.associateArrayWithVariable(b, sd.getVariable("b"));

            INDArray outArr = out.eval();

            assertArrayEquals(new long[]{minibatch, nOut}, outArr.shape());
        }
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
    public void testSequenceAdd(Nd4jBackend backend) throws IOException {
        assertThrows(NullPointerException.class,() -> {
            SameDiff sd = SameDiff.create();
            sd.addItemToSequence("dummy",null,0);
        });

        assertThrows(IllegalStateException.class,() -> {
            SameDiff sd = SameDiff.create();
            sd.addItemToSequence("dummy",Nd4j.ones(1),0);
        });


        SameDiff sd = SameDiff.create();
        sd.createSequence("x",new INDArray[]{Nd4j.ones(1)});
        assertTrue(sd.hasVariable("x"));
        assertEquals(VariableType.SEQUENCE,sd.getVariable("x").getVariableType());
        assertEquals(Nd4j.ones(1),sd.itemForSequence("x",0));
        sd.setItemForSequenceAtIndex("x",Nd4j.ones(2),0);
        assertEquals(Nd4j.ones(2),sd.itemForSequence("x",0));
        assertEquals(1,sd.sequenceLength("x"));
        sd.removeItemFromSequence("x",0);
        assertFalse(sd.hasVariable("x"));
        assertThrows(IllegalStateException.class,() -> {
            SameDiff sd2 = SameDiff.create();
            sd2.itemForSequence("x",1);
        });
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSequenceNegativeIndex(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        INDArray[] sequence = {Nd4j.ones(1),Nd4j.ones(2)};
        sd.createSequence("x",sequence);
        //adds the item at the last index
        sd.addItemToSequence("x",Nd4j.ones(3),-1);
        assertEquals(Nd4j.ones(3),sd.itemForSequence("x",-1));
        sd.removeItemFromSequence("x",-1);
        assertEquals(Nd4j.ones(2),sd.itemForSequence("x",-1));
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSequentialMeansPlaceholder(Nd4jBackend backend) {
        for (int dim0 : new int[]{10, -1}) {
            String msg = "Dimension 0 = " + dim0;
            System.out.println(msg);
            SameDiff sd = SameDiff.create();
            SDVariable in = sd.var("in", new long[]{dim0, 9, 8});
            SDVariable mean1 = sd.mean(in, 2);                  //[10,9,8] -> [10,9]
            SDVariable mean2 = sd.mean(mean1, 1);               //[10,9] -> [10]

            INDArray inArr = Nd4j.create(10, 9, 8);
            sd.associateArrayWithVariable(inArr, in);

            INDArray out = mean2.eval();

            long[] shape = out.shape();
            assertArrayEquals(new long[]{10}, shape,msg);
        }
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReductionShapes1(Nd4jBackend backend) {

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", new long[]{10, 9, 8});
        SDVariable mean1 = sd.mean(in, 2);      //[10,9] out
        SDVariable mean2 = sd.mean(mean1, 1);   //[10] out
        Map<String,INDArray> m = sd.output((Map<String,INDArray>)null, mean1.name(), mean2.name());

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
        Map<String,INDArray> out = sd2.outputAll(null);
        assertArrayEquals(new long[]{9, 8}, out.get(meanA.name()).shape());

        SDVariable meanB = sd2.mean(meanA, 0);   //[8] out
        Map<String,INDArray> m = sd2.outputAll(null);
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
    public void testNames(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable in1 = sd.var("in", new long[]{3, 2});
        SDVariable in2 = sd.var("in2", new long[]{3, 3});

        val m = in1.add(1.0);
        val f = m.add(2.0);
        val s = in2.add(5.0);

        Map<String,INDArray> map = sd.outputAll(null);
//        log.info("Result M: {}", map.get(m.name()));
//        log.info("Result F: {}", map.get(f.name()));
//        log.info("Result S: {}", map.get(s.name()));
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
        int epochsToRun = 5;
        double lr = 0.1;
     /*   for(int i = 0; i < epochsToRun; i++) {
            activation.execBackwards();
            INDArray wGrad = activation.grad("w").getArr().reshape(vars.get("w").shape());
            vars.get("w").subi(wGrad.mul(lr));
            System.out.println("Score: " + activation.getVariable("negtotalsum").getArr());
        }*/

    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTransposeWithVector(Nd4jBackend backend) {
        val sd = SameDiff.create();
        val matrix = Nd4j.linspace(1, 12, 12).reshape(4, 3);
        val vector = Nd4j.linspace(1, 4, 4).reshape(4, 1);
        val input1 = sd.var("input", matrix);
        val input2 = sd.var("input2", vector);
        val output = sd.mmul("output", input1, input2, true, false, false);
        INDArray out = output.eval();
        assertArrayEquals(new long[]{3, 1}, out.shape());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSimpleDefineFunction(Nd4jBackend backend) {
        SameDiff sameDiffOuter = SameDiff.create();
        Map<String, INDArray> inputs = variablesForInput();
        inputs.remove("y");
        String logisticForward = "logisticPredictions";
        sameDiffOuter.defineFunction(logisticForward, (sameDiff, inputs1, variableInputs) -> {

            SDVariable input = sameDiff.var("x", inputs1.get("x"));
            SDVariable w = sameDiff.var("w", inputs1.get("w"));
            SDVariable preOutput = sameDiff.mmul(input, w);
            SDVariable sigmoid = sameDiff.nn().sigmoid(preOutput);
            return new SDVariable[]{sigmoid};
        }, inputs);

        assertEquals(1, sameDiffOuter.definedFunctionNames().size());

        //note here that we don't add the duplicate ops with define function anymore
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSumGradient(Nd4jBackend backend) {
        SameDiff sameDiff = SameDiff.create();
        SDVariable twoByTwo = sameDiff.var("initial", Nd4j.linspace(1, 4, 4, DataType.FLOAT).reshape(2, 2));
        SDVariable sum = sameDiff.sum(twoByTwo, Integer.MAX_VALUE);
        Map<String,INDArray> grads = sameDiff.calculateGradients(Collections.emptyMap(), sameDiff.getVariables().keySet());
        assertEquals(Nd4j.ones(DataType.FLOAT, 2, 2), grads.get(twoByTwo.name()));
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRsubScalar(Nd4jBackend backend) {
        SameDiff sameDiff = SameDiff.create();
        Map<String, INDArray> params = new HashMap<>();
        INDArray var = Nd4j.valueArrayOf(4, 2);
        params.put("x", var);
        sameDiff.defineFunction("rsubop", (sameDiff1, inputs, variableInputs) -> {
            SDVariable input = sameDiff1.var("x", inputs.get("x"));
            SDVariable ret = input.rsub("rsub", 1.0);
            return new SDVariable[]{ret};
        }, params);

        SameDiff logisticGraph = sameDiff.getFunction("rsubop");
        INDArray output = logisticGraph.output(params, Collections.singletonList("rsub")).get("rsub");
        assertEquals(Nd4j.ones(4).muli(-1), output);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testFunctionScalarResultPropagation(Nd4jBackend backend) {
        SameDiff sameDiffOuter = SameDiff.create();
        Map<String, INDArray> inputs = variablesForInput();

        sameDiffOuter.defineFunction("logisticPredictions", (sameDiff, inputs12, variableInputs) -> {
            SDVariable input = sameDiff.var("x", inputs12.get("x"));
            SDVariable w = sameDiff.var("w", inputs12.get("w"));
            SDVariable preOutput = sameDiff.mmul(input, w);
            SDVariable sigmoid = sameDiff.nn().sigmoid(preOutput);
            return new SDVariable[]{sigmoid};
        }, inputs);

        sameDiffOuter.defineFunction("oneminuspredictions", (sameDiff, inputs1, variableInputs) -> {
            SDVariable y = sameDiff.var("y", inputs1.get("y"));
            SDVariable oneMinusPredictions = y.rsub("rsub", 1.0);
            return new SDVariable[]{oneMinusPredictions};
        }, inputs);

        SameDiff logisticGraph = sameDiffOuter.getFunction("oneminuspredictions");
        Map<String, INDArray> inputsSubset = new HashMap<>();
        inputsSubset.put("y", inputs.get("y"));
        INDArray output = logisticGraph.output(inputsSubset, Collections.singletonList("rsub")).get("rsub");
        INDArray assertion = Nd4j.create(new double[]{0, 0, 1, 0}, new int[]{4, 1});
        assertEquals(assertion, output);

    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMmul(Nd4jBackend backend) {
        SameDiff sameDiffOuter = SameDiff.create();
        Map<String, INDArray> inputs = variablesForInput();
        SDVariable x = sameDiffOuter.var("x", inputs.get("x"));
        SDVariable w = sameDiffOuter.var("w", inputs.get("w"));
        SDVariable output = sameDiffOuter.mmul(x, w);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGraphBuilding(Nd4jBackend backend) {
        final SameDiff sameDiffOuter = SameDiff.create();
        Map<String, INDArray> inputs = variablesForInput();

        sameDiffOuter.defineFunction("logisticPredictions", (sameDiff, inputs1, variableInputs) -> {
            SDVariable input = sameDiff.var("x", inputs1.get("x"));
            SDVariable w = sameDiff.var("w", inputs1.get("w"));
            SDVariable y = sameDiff.var("y", inputs1.get("y"));
            SDVariable preOutput = sameDiff.mmul(input, w);
            SDVariable sigmoid = sameDiff.nn().sigmoid(preOutput);

            return new SDVariable[]{sigmoid};
        }, inputs);

        sameDiffOuter.defineFunction("loss", (sameDiff, inputs12, variableInputs) -> {
            SDVariable outputs = sameDiffOuter.invokeFunctionOn("logisticPredictions", sameDiff);
            SDVariable y = sameDiff.getVariable("y");
            SDVariable outputTimesY = outputs.mul(y);
            return new SDVariable[]{outputTimesY};

        }, inputs);

        SameDiff logisticPrediction = sameDiffOuter.getFunction("logisticPredictions");
        List<String> logisticOpNameAssertions = Arrays.asList("mmul", "sigmoid");


    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScalarAdd(Nd4jBackend backend) {
        SameDiff sameDiff = SameDiff.create();
        SDVariable twoByTwo = sameDiff.var("first", Nd4j.linspace(1, 4, 4).reshape('c', 2, 2));
        SDVariable add = twoByTwo.add(1.0);
        INDArray test = add.eval();
        INDArray assertion = Nd4j.linspace(1, 4, 4).reshape('c', 2, 2).add(1.0);
        assertEquals(assertion, test);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSums(Nd4jBackend backend) {
        SameDiff sameDiff = SameDiff.create();
        INDArray ones = Nd4j.ones(7, 4);
        SDVariable sdVariable = sameDiff.var("ones", ones);
        SDVariable result = sdVariable.add(1.0);
        SDVariable total = sameDiff.sum(result, Integer.MAX_VALUE);
        INDArray out = total.eval();
        assertEquals(56, out.getDouble(0), 1e-1);
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

        Map<String,INDArray> m = sd.outputAll(Collections.emptyMap());

        assertEquals(expMmul, m.get(mmul.name()));
        assertEquals(expZ, m.get(z.name()));
        assertEquals(expOut, m.get(out.name()));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testActivationBackprop(Nd4jBackend backend) {

        Activation[] afns = new Activation[]{
                Activation.TANH,
                Activation.SIGMOID,
                Activation.ELU,
                Activation.SOFTPLUS,
                Activation.SOFTSIGN,
                Activation.HARDTANH,
                Activation.CUBE,
                //WRONG output - see issue https://github.com/deeplearning4j/nd4j/issues/2426
                Activation.RELU,            //JVM crash
                Activation.LEAKYRELU        //JVM crash
        };

        for (Activation a : afns) {

            SameDiff sd = SameDiff.create();
            INDArray inArr = Nd4j.linspace(-3, 3, 7);
            INDArray labelArr = Nd4j.linspace(-3, 3, 7).muli(0.5);
            SDVariable in = sd.var("in", inArr.dup());

//            System.out.println("inArr: " + inArr);

            INDArray outExp;
            SDVariable out;
            switch (a) {
                case ELU:
                    out = sd.nn().elu("out", in);
                    outExp = Transforms.elu(inArr, true);
                    break;
                case HARDTANH:
                    out = sd.nn().hardTanh("out", in);
                    outExp = Transforms.hardTanh(inArr, true);
                    break;
                case LEAKYRELU:
                    out = sd.nn().leakyRelu("out", in, 0.01);
                    outExp = Transforms.leakyRelu(inArr, true);
                    break;
                case RELU:
                    out = sd.nn().relu("out", in, 0.0);
                    outExp = Transforms.relu(inArr, true);
                    break;
                case SIGMOID:
                    out = sd.nn().sigmoid("out", in);
                    outExp = Transforms.sigmoid(inArr, true);
                    break;
                case SOFTPLUS:
                    out = sd.nn().softplus("out", in);
                    outExp = Transforms.softPlus(inArr, true);
                    break;
                case SOFTSIGN:
                    out = sd.nn().softsign("out", in);
                    outExp = Transforms.softsign(inArr, true);
                    break;
                case TANH:
                    out = sd.math().tanh("out", in);
                    outExp = Transforms.tanh(inArr, true);
                    break;
                case CUBE:
                    out = sd.math().cube("out", in);
                    outExp = Transforms.pow(inArr, 3, true);
                    break;
                default:
                    throw new RuntimeException(a.toString());
            }

            //Sum squared error loss:
            SDVariable label = sd.var("label", labelArr.dup());
            SDVariable diff = label.sub("diff", out);
            SDVariable sqDiff = diff.mul("sqDiff", diff);
            SDVariable totSum = sd.sum("totSum", sqDiff, Integer.MAX_VALUE);    //Loss function...
            sd.setLossVariables(totSum);
            Map<String,INDArray> m = sd.output(Collections.emptyMap(), "out");
            INDArray outAct = m.get("out");
            assertEquals(outExp, outAct,a.toString());

            // L = sum_i (label - out)^2
            //dL/dOut = 2(out - label)
            INDArray dLdOutExp = outExp.sub(labelArr).mul(2);
            INDArray dLdInExp = a.getActivationFunction().backprop(inArr.dup(), dLdOutExp.dup()).getFirst();

            Map<String,INDArray> grads = sd.calculateGradients(null, "out", "in");
//            sd.execBackwards(Collections.emptyMap());
//            SameDiff gradFn = sd.getFunction("grad");
            INDArray dLdOutAct = grads.get("out");
            INDArray dLdInAct = grads.get("in");

            assertEquals(dLdOutExp, dLdOutAct,a.toString());
            assertEquals(dLdInExp, dLdInAct,a.toString());
        }
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPlaceholderReduceSimple(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable v = sd.var("in", new long[]{-1, 10});
        SDVariable vSum = sd.sum(v, 1);                             //Exception here
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSequentialMeans(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", new long[]{10, 10, 10});
        SDVariable mean1 = sd.mean(in, 2);      //[10,10] out
        SDVariable mean2 = sd.mean(mean1, 1);   //[10,1] out - ***exception here***
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBatchNormTest(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();

        INDArray input = Nd4j.rand(1, 10);
        INDArray mean = Nd4j.rand(1, 10).reshape(10);
        INDArray var = Nd4j.rand(1, 10).reshape(10);
        INDArray gamma = Nd4j.rand(1, 10).reshape(10);
        INDArray beta = Nd4j.rand(1, 10).reshape(10);

        SDVariable sdInput = sd.var("input", input);
        SDVariable sdMean = sd.var("mean", mean);
        SDVariable sdVar = sd.var("var", var);
        SDVariable sdGamma = sd.var("gamma", gamma);
        SDVariable sdBeta = sd.var("beta", beta);

        SDVariable out = sd.nn().batchNorm(sdInput, sdMean, sdVar, sdGamma, sdBeta,
                0.0, 1);
        out = sd.math().tanh(out);

        INDArray outArr = out.eval();
        assertArrayEquals(new long[]{1, 10}, outArr.shape());

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLrn(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();

        INDArray input = Nd4j.create(new float[]{4, 4, 4, 4}, new long[]{1, 4, 1, 1});

        SDVariable sdInput = sd.var("input", input);

        LocalResponseNormalizationConfig lrn = LocalResponseNormalizationConfig.builder()
                .alpha(1.0)
                .beta(.5)
                .bias(0.0)
                .depth(1).build();

        SDVariable out = sd.cnn().localResponseNormalization(sdInput, lrn);
        SDVariable sdOut = sd.math().tanh("out", out);

        Map<String,INDArray> map = sd.output(Collections.emptyMap(), "out", out.name());

        for (int i = 0; i < 4; i++) {
            assertEquals(1, map.get(out.name()).get(all(), NDArrayIndex.point(i), all(), all()).getInt(0));
        }

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMoments(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();

        INDArray input = Nd4j.create(new float[]{1, 2, 3, 4}, new long[]{2, 2});

        SDVariable sdInput = sd.var("input", input);

        SDVariable[] moments = sd.math().moments(sdInput, new long[]{0, 1},false);
        SDVariable mean = moments[0];
        SDVariable variance = moments[1];

        SDVariable sum = mean.add(variance);
        SDVariable out = sd.math().tanh("out", sum);

        Map<String,INDArray> m = sd.outputAll(null);

        INDArray meanArray = m.get(mean.name());
        INDArray varArray = m.get(variance.name());

        assertEquals(meanArray.getDouble(0), 2.5, 1e-5);
        assertEquals(varArray.getDouble(0), 1.25, 1e-5);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNormalizeMoments(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();

        INDArray counts = Nd4j.create(new float[]{2}, new long[]{1, 1});
        INDArray means = Nd4j.create(new float[]{2, 4}, new long[]{1, 2});
        INDArray vars = Nd4j.create(new float[]{6, 8}, new long[]{1, 2});

        SDVariable sdCounts = sd.var("counts", counts);
        SDVariable sdMeans = sd.var("means", means);
        SDVariable sdVars = sd.var("vars", vars);
        double shift = 0.0;

        SDVariable[] moments = sd.math().normalizeMoments(sdCounts, sdMeans, sdVars, shift);
        SDVariable normMean = moments[0];
        SDVariable normVariance = moments[1];

        SDVariable sum = normMean.add(normVariance);
        SDVariable out = sd.math().tanh("out", sum);

        Map<String,INDArray> m = sd.outputAll(null);

        INDArray meanArray = m.get(normMean.name());
        INDArray varArray = m.get(normVariance.name());

        assertEquals(meanArray.getDouble(0, 0), 1, 1e-5);
        assertEquals(meanArray.getDouble(0, 1), 2, 1e-5);
        assertArrayEquals(meanArray.shape(), varArray.shape());
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDepthWiseConv2dBasic(Nd4jBackend backend) {
        int nIn = 3;
        int depthWise = 4;
        int kH = 2;
        int kW = 2;

        int mb = 3;
        int imgH = 28;
        int imgW = 28;

        SameDiff sd = SameDiff.create();
        INDArray depthWeightArr = Nd4j.create(kH, kW, nIn, depthWise);

        INDArray bArr = Nd4j.create(1, depthWise * nIn);
        INDArray inArr = Nd4j.create(mb, nIn, imgH, imgW);

        SDVariable in = sd.var("in", inArr);
        SDVariable dW = sd.var("dW", depthWeightArr);
        SDVariable b = sd.var("b", bArr);

        Conv2DConfig c = Conv2DConfig.builder()
                .kH(kH).kW(kW)
                .pH(0).pW(0)
                .sH(1).sW(1)
                .dH(1).dW(1)
                .paddingMode(PaddingMode.VALID)
                .build();

        SDVariable out = sd.cnn().depthWiseConv2d(in, dW, b, c);
        out = sd.math().tanh("out", out);

        INDArray outArr = out.eval();
        //Expected output size: out = (in - k + 2*p)/s + 1 = (28-2+0)/1+1 = 27
        val outShape = outArr.shape();
        assertArrayEquals(new long[]{mb, depthWise * nIn, 27, 27}, outShape);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void validateMeanDiff(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        INDArray arr = Nd4j.rand(3, 4);

        SameDiff sd = SameDiff.create();
        SDVariable v = sd.var("in", arr);
        SDVariable mean = sd.mean("mean", v);

        INDArray out = mean.eval();
        assertEquals(out, arr.mean(Integer.MAX_VALUE));

        Map<String,INDArray> m = sd.calculateGradients(Collections.emptyMap(), sd.getVariables().keySet());
        INDArray dLdIn = m.get("in");

        //If L = mean(in)
        //then dL/dIn = 1/N

        assertEquals(Nd4j.valueArrayOf(arr.shape(), 1.0 / arr.length()), dLdIn);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void validateSumDiff(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        INDArray arr = Nd4j.rand(3, 4);

        SameDiff sd = SameDiff.create();
        SDVariable v = sd.var("in", arr);
        SDVariable mean = sd.sum("sum", v);

        INDArray out = mean.eval();
        assertEquals(out, arr.sum(Integer.MAX_VALUE));

        Map<String,INDArray> m = sd.calculateGradients(Collections.emptyMap(), sd.getVariables().keySet());
        INDArray dLdIn = m.get("in");

        //If L = sum(in)
        //then dL/dIn = 1

        assertEquals(Nd4j.ones(arr.shape()), dLdIn);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void validateStdevDiff(Nd4jBackend backend) {
        for (boolean biasCorrected : new boolean[]{true, false}) {
            Nd4j.getRandom().setSeed(12345);

            INDArray arr = Nd4j.rand(3, 4);

            SameDiff sd = SameDiff.create();
            SDVariable v = sd.var("in", arr);
            SDVariable stdev = sd.standardDeviation("stdev", v, biasCorrected);

            INDArray out = stdev.eval();
            assertEquals(out, arr.std(biasCorrected, Integer.MAX_VALUE));

            Map<String,INDArray> g = sd.calculateGradients(Collections.emptyMap(), sd.getVariables().keySet());
            INDArray dLdIn = sd.grad("in").getArr();

            //If L = stdev(in)
            //then dL/dIn = (in-mean) / (s*(N-1))
            // or /N for non-bias corrected

            double m = arr.meanNumber().doubleValue();
            double s = arr.stdNumber(biasCorrected).doubleValue();
            INDArray exp = arr.sub(m).div(s);
            exp.divi(biasCorrected ? arr.length() - 1 : arr.length());

            assertEquals(exp, dLdIn);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void validateVarDiff(Nd4jBackend backend) {
        for (boolean biasCorrected : new boolean[]{true,false}) {
            Nd4j.getRandom().setSeed(12345);

            INDArray arr = Nd4j.rand(3, 4);

            SameDiff sd = SameDiff.create();
            SDVariable v = sd.var("in", arr);
            SDVariable var = sd.variance("var", v, biasCorrected);

            INDArray out = var.eval();
            assertEquals(out, arr.var(biasCorrected, Integer.MAX_VALUE));

            Map<String,INDArray> g = sd.calculateGradients(Collections.emptyMap(), sd.getVariables().keySet());
            INDArray dLdIn = g.get("in");

            //If L = var(in)
            //then dL/dIn = 2/(N-1) * (in-mean)
            // or /N for non-bias corrected

            double m = arr.meanNumber().doubleValue();
            INDArray exp = arr.sub(m).mul(2);
            exp.divi(biasCorrected ? arr.length() - 1 : arr.length());
            //non bias corrected gradients are less precise
            double eps = biasCorrected ? Nd4j.EPS_THRESHOLD : 1e-2;
            assertTrue(exp.equalsWithEps(dLdIn,eps));
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void validateMinDiff(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        INDArray arr = Nd4j.rand(3, 4);

        SameDiff sd = SameDiff.create();
        SDVariable v = sd.var("in", arr);
        SDVariable min = sd.min("min", v);

        INDArray out = min.eval();
        assertEquals(out, arr.min(Integer.MAX_VALUE));

        Map<String,INDArray> g = sd.calculateGradients(Collections.emptyMap(), sd.getVariables().keySet());
        INDArray dLdIn = sd.grad("in").getArr();

        //If L = min(in)
        //then dL/dIn = 1 if in_i == min(in) or 0 otherwise

        //Note that we don't have an "IsMin" op, so use IsMax(neg(in)) which is equivalent
        INDArray exp = Nd4j.exec(new IsMax(arr.neg()))[0].castTo(Nd4j.defaultFloatingPointType());

        assertEquals(exp, dLdIn);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void validateMaxDiff(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        INDArray arr = Nd4j.rand(DataType.DOUBLE, 3, 4);

        SameDiff sd = SameDiff.create();
        SDVariable v = sd.var("in", arr);
        SDVariable min = sd.max("max", v);

        INDArray out = min.eval();
        assertEquals(out, arr.max(Integer.MAX_VALUE));

        sd.calculateGradients(Collections.emptyMap(), sd.getVariables().keySet());
        INDArray dLdIn = sd.grad("in").getArr();

        //If L = max(in)
        //then dL/dIn = 1 if in_i == max(in) or 0 otherwise

        INDArray exp = Nd4j.exec(new IsMax(arr.dup()))[0].castTo(DataType.DOUBLE);

        assertEquals(exp, dLdIn);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void validateProdDiff(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        INDArray arr = Nd4j.rand(3, 4);

        SameDiff sd = SameDiff.create();
        SDVariable v = sd.var("in", arr);
        SDVariable prod = sd.prod("prod", v);

        double p = arr.prodNumber().doubleValue();
        INDArray out = prod.eval();
        assertEquals(out, arr.prod(Integer.MAX_VALUE));

        Map<String,INDArray> g = sd.calculateGradients(Collections.emptyMap(), sd.getVariables().keySet());
        INDArray dLdIn = sd.grad("in").getArr();

        //If L = prod(in)
        //then dL/dIn = prod(in) / in       i.e., product of input *excluding* in_i as (d/dx(xyzabc) = yzabc

        INDArray exp = arr.rdiv(p);
        assertEquals(exp, dLdIn);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSquare(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        int mb = 5;
        int nOut = 4;

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", Nd4j.rand(mb, nOut));
        SDVariable label = sd.var("label", Nd4j.rand(mb, nOut));
        SDVariable diff = in.sub(label);
        SDVariable sqDiff = sd.math().square(diff);

        INDArray expOut = in.getArr().sub(label.getArr());
        expOut.muli(expOut);

        INDArray out = sqDiff.eval();

        assertEquals(out, expOut);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testExpandDims(Nd4jBackend backend) {
        for (int i = 0; i <= 2; i++) {
            SameDiff sd = SameDiff.create();
            SDVariable in = sd.var("in", Nd4j.create(2, 3));
            SDVariable expanded = sd.expandDims(in, i);

            INDArray out = expanded.eval();
            switch (i) {
                case 0:
                    assertArrayEquals(new long[]{1, 2, 3}, out.shape());
                    break;
                case 1:
                    assertArrayEquals(new long[]{2, 1, 3}, out.shape());
                    break;
                case 2:
                    assertArrayEquals(new long[]{2, 3, 1}, out.shape());
                    break;
                default:
                    throw new RuntimeException();
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testZerosLike(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable var0 = sd.var("in", DataType.DOUBLE, new long[]{3, 4});
        SDVariable out = sd.zerosLike("out", var0);

        INDArray out1 = out.eval();
        assertEquals(Nd4j.zeros(3, 4), out1);

        sd.associateArrayWithVariable(Nd4j.create(3, 4), var0);
        INDArray out2 = out.eval();
        assertEquals(Nd4j.zeros(DataType.DOUBLE, 3, 4), out2);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testOnesLike(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable var0 = sd.var("in", new long[]{3, 4});
        SDVariable out = sd.onesLike("out", var0);

        INDArray out1 = out.eval();
        assertEquals(Nd4j.ones(3, 4), out1);

        sd.associateArrayWithVariable(Nd4j.create(3, 4), var0);
        INDArray out2 = out.eval();
        assertEquals(Nd4j.ones(3, 4), out2);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testOnesLikeBackprop(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable var0 = sd.var("in", new long[]{3, 4});
        SDVariable ones = sd.onesLike("ones", var0);
        SDVariable out = sd.sum("oun", ones);

        INDArray outArr = out.eval();
        assertEquals(Nd4j.scalar(12.0), outArr);

        Map<String,INDArray> m = sd.calculateGradients(Collections.emptyMap(), sd.getVariables().keySet());

        assertEquals(Nd4j.create(3, 4), m.get("in"));
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testManhattanAlongDim0(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        INDArray a = Nd4j.rand(new long[]{3, 4, 5});
        INDArray b = Nd4j.rand(new long[]{3, 4, 5});

        INDArray expOut = Nd4j.exec(new ManhattanDistance(a, b, 0));

        val expShape = new long[]{4, 5};

        assertArrayEquals(expShape, expOut.shape());
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testJaccardDistance(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        INDArray a = Nd4j.rand(new long[]{3, 4}).addi(0.1);
        INDArray b = Nd4j.rand(new long[]{3, 4}).addi(0.1);


        SameDiff sd = SameDiff.create();
        SDVariable in1 = sd.var("in1", a);
        SDVariable in2 = sd.var("in2", b);

        SDVariable jaccard = sd.math().jaccardDistance("out", in1, in2);

        INDArray min = Transforms.min(a, b);
        INDArray max = Transforms.max(a, b);

        double minSum = min.sumNumber().doubleValue();
        double maxSum = max.sumNumber().doubleValue();
        double jd = 1.0 - minSum / maxSum;

        INDArray out = jaccard.eval();
        assertEquals(1, out.length());

        assertEquals(jd, out.getDouble(0), 1e-6);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPairwiseBooleanTransforms(Nd4jBackend backend) {
        /*
        eq, neq, gt, lt, gte, lte, or, and, xor
         */
        //Test transforms (pairwise)
        Nd4j.getRandom().setSeed(12345);

        for (int i = 0; i < 11; i++) {
            SameDiff sd = SameDiff.create();

            int nOut = 4;
            int minibatch = 5;

            INDArray ia = Nd4j.randn(minibatch, nOut);
            INDArray ib = Nd4j.randn(minibatch, nOut);

            SDVariable in1 = sd.var("in1", ia);
            SDVariable in2 = sd.var("in2", ib);

            SDVariable t;
            INDArray expOut;
            switch (i) {
                case 0:
                    t = sd.eq(in1, in2);
                    expOut = ia.eq(ib);
                    break;
                case 1:
                    t = sd.neq(in1, in2);
                    expOut = ia.neq(ib);
                    break;
                case 2:
                    t = sd.gt(in1, in2);
                    expOut = ia.gt(ib);
                    break;
                case 3:
                    t = sd.lt(in1, in2);
                    expOut = ia.lt(ib);
                    break;
                case 4:
                    t = sd.gte(in1, in2);
                    expOut = Nd4j.create(DataType.BOOL, ia.shape());
                    Nd4j.exec(new GreaterThanOrEqual(new INDArray[]{ia, ib}, new INDArray[]{expOut}));
                    break;
                case 5:
                    t = sd.lte(in1, in2);
                    expOut = Nd4j.create(DataType.BOOL, ia.shape());
                    Nd4j.exec(new LessThanOrEqual(new INDArray[]{ia, ib}, new INDArray[]{expOut}));
                    break;
                case 6:
                    ia = Nd4j.exec(new BernoulliDistribution(ia, 0.5));
                    ib = Nd4j.exec(new BernoulliDistribution(ib, 0.5));
                    t = sd.math().or(in1.castTo(DataType.BOOL), in2.castTo(DataType.BOOL));
                    expOut = Transforms.or(ia, ib);
                    break;
                case 7:
                    t = sd.max(in1, in2);
                    expOut = Nd4j.exec(new Max(ia, ib, ia.dup()))[0];
                    break;
                case 8:
                    t = sd.min(in1, in2);
                    expOut = Nd4j.exec(new Min(ia, ib, ia.dup()))[0];
                    break;
                case 9:
                    ia = Nd4j.exec(new BernoulliDistribution(ia, 0.5));
                    ib = Nd4j.exec(new BernoulliDistribution(ib, 0.5));
                    t = sd.math().and(in1.castTo(DataType.BOOL), in2.castTo(DataType.BOOL));
                    expOut = Transforms.and(ia, ib);
                    break;
                case 10:
                    ia = Nd4j.exec(new BernoulliDistribution(ia, 0.5));
                    ib = Nd4j.exec(new BernoulliDistribution(ib, 0.5));
                    t = sd.math().xor(in1.castTo(DataType.BOOL), in2.castTo(DataType.BOOL));
                    expOut = Transforms.xor(ia, ib);
                    break;
                default:
                    throw new RuntimeException();
            }

            log.info("Executing: " + i);
            INDArray out = t.eval();

            assertEquals(expOut, out);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBooleanChecks(Nd4jBackend backend) {
        /*
        isNonDecreasing,
         */
        Nd4j.getRandom().setSeed(12345);

        for (int i = 0; i < 3; i++) {
            SameDiff sd = SameDiff.create();

            int nOut = 4;
            int minibatch = 5;

            INDArray ia = Nd4j.randn(minibatch, nOut);

            SDVariable in1 = sd.var("in1", ia);
            INDArray expOut = Nd4j.scalar(true);
            SDVariable t;

            switch (i) {
                case 0:
                    t = sd.math().isNonDecreasing(in1);
                    Nd4j.exec(new IsNonDecreasing(ia, expOut));
                    break;
                case 1:
                    t = sd.math().isStrictlyIncreasing(in1);
                    Nd4j.exec(new IsStrictlyIncreasing(ia, expOut));
                    break;
                case 2:
                    t = sd.isNumericTensor(in1);
                    Nd4j.exec(new IsNumericTensor(new INDArray[]{ia}, new INDArray[]{expOut}));
                    break;
                default:
                    throw new RuntimeException();
            }

            log.info("Executing: " + i);
            INDArray out = t.eval();

            assertEquals(expOut, out);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testIsStrictlyIncShape(Nd4jBackend backend) {
        int nOut = 0;
        int minibatch = 0;

        INDArray ia = Nd4j.randn(minibatch, nOut);
        INDArray expOut = Nd4j.create(DataType.BOOL, ia.shape());

        Nd4j.exec(new IsStrictlyIncreasing(ia, expOut));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testExpandDims2d(Nd4jBackend backend) {
        val origShape = new long[]{3, 4};

        for (int i = 0; i < 3; i++) {
            for (Pair<INDArray, String> p : NDArrayCreationUtil
                    .getAllTestMatricesWithShape(origShape[0], origShape[1], 12345, DataType.FLOAT)) {
                INDArray inArr = p.getFirst().muli(100);

                SameDiff sd = SameDiff.create();
                SDVariable in = sd.var("in", inArr);
                SDVariable expand = sd.expandDims(in, i);

                INDArray out = expand.eval();

                INDArray expOut;
                switch (i) {
                    case 0:
                        expOut = inArr.dup('c').reshape('c', 1, origShape[0], origShape[1]);
                        break;
                    case 1:
                        expOut = inArr.dup('c').reshape('c', origShape[0], 1, origShape[1]);
                        break;
                    case 2:
                        expOut = inArr.dup('c').reshape('c', origShape[0], origShape[1], 1);
                        break;
                    default:
                        throw new RuntimeException();
                }

                String msg = "expandDim=" + i + ", source=" + p.getSecond();

                assertEquals(out, expOut,msg);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSqueezeDims(Nd4jBackend backend) {
        val origShape = new long[]{3, 4, 5};

        for (int i = 0; i < 3; i++) {

            val shape = origShape.clone();
            shape[i] = 1;

            for (Pair<INDArray, String> p : NDArrayCreationUtil
                    .getAll3dTestArraysWithShape(12345, shape, DataType.FLOAT)) {
                INDArray inArr = p.getFirst().muli(100);

                SameDiff sd = SameDiff.create();
                SDVariable in = sd.var("in", inArr);
                SDVariable squeeze = sd.squeeze(in, i);

                INDArray out = squeeze.eval();

                INDArray expOut;
                switch (i) {
                    case 0:
                        expOut = inArr.dup('c').reshape('c', origShape[1], origShape[2]);
                        break;
                    case 1:
                        expOut = inArr.dup('c').reshape('c', origShape[0], origShape[2]);
                        break;
                    case 2:
                        expOut = inArr.dup('c').reshape('c', origShape[0], origShape[1]);
                        break;
                    default:
                        throw new RuntimeException();
                }

                String msg = "squeezeDim=" + i + ", source=" + p.getSecond();

                assertEquals(out, expOut,msg);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testExpandSqueezeChain(Nd4jBackend backend) {

        val origShape = new long[]{3, 4};

        for (int i = 0; i < 3; i++) {
            for (Pair<INDArray, String> p : NDArrayCreationUtil
                    .getAllTestMatricesWithShape(origShape[0], origShape[1], 12345, DataType.FLOAT)) {
                INDArray inArr = p.getFirst().muli(100);

                SameDiff sd = SameDiff.create();
                SDVariable in = sd.var("in", inArr);
                SDVariable expand = sd.expandDims(in, i);
                SDVariable squeeze = sd.squeeze(expand, i);

                INDArray out = squeeze.eval();

                String msg = "expand/Squeeze=" + i + ", source=" + p.getSecond();

                assertEquals(out, inArr,msg);  //expand -> squeeze: should be opposite ops
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSqueezeExpandChain(Nd4jBackend backend) {

        val origShape = new long[]{3, 4, 5};

        for (int i = 0; i < 3; i++) {

            val shape = origShape.clone();
            shape[i] = 1;

            for (Pair<INDArray, String> p : NDArrayCreationUtil
                    .getAll3dTestArraysWithShape(12345, shape, DataType.FLOAT)) {
                INDArray inArr = p.getFirst().muli(100);

                SameDiff sd = SameDiff.create();
                SDVariable in = sd.var("in", inArr);
                SDVariable squeeze = sd.squeeze(in, i);
                SDVariable expand = sd.expandDims(squeeze, i);

                INDArray out = expand.eval();

                String msg = "expand/Squeeze=" + i + ", source=" + p.getSecond();

                assertEquals(out, inArr,msg);  //squeeze -> expand: should be opposite ops
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConfusionMatrix(Nd4jBackend backend) {
        INDArray labels = Nd4j.createFromArray(1, 2, 4);
        INDArray pred = Nd4j.createFromArray(2, 2, 4);
        INDArray weights = Nd4j.createFromArray(10, 100, 1000);
        Integer numClasses = 5;
        SameDiff sd = SameDiff.create();
        SDVariable labelsVar = sd.constant("labels", labels);
        SDVariable predictionsVar = sd.constant("predictions", pred);
        SDVariable weightsVar = sd.constant("weights", weights);
        SDVariable cm = sd.math().confusionMatrix("cm", labelsVar, predictionsVar, weightsVar, numClasses);
        INDArray out = cm.eval();

        INDArray exp = Nd4j.create(new float[][]{{0, 0, 0, 0, 0}, {0, 0, 10, 0, 0}, {0, 0, 100, 0, 0},
                {0, 0, 0, 0, 0}, {0, 0, 0, 0, 1000}}).castTo(DataType.INT);

        assertEquals(exp, out);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testArgMax(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        for (val dim : new long[][]{{0}, {1}, {Integer.MAX_VALUE}, {0, 1}, {}}) {
            INDArray inArr = Nd4j.rand(3, 4);
            SameDiff sd = SameDiff.create();

            SDVariable in = sd.var("in", inArr);
            SDVariable argmax = sd.argmax("argmax", in, dim);

            INDArray out = argmax.eval();

            INDArray exp = Nd4j.argMax(inArr, dim);

            assertEquals(exp, out);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testArgMin(Nd4jBackend backend) {

        Nd4j.getRandom().setSeed(12345);

        for (val dim : new long[][]{{0}, {1}, {Integer.MAX_VALUE}, {0, 1}, {}}) {
            INDArray inArr = Nd4j.rand(3, 4);
            SameDiff sd = SameDiff.create();

            SDVariable in = sd.var("in", inArr);
            SDVariable argmin = sd.argmin("argmin", in, dim);

            INDArray out = argmin.eval();

            INDArray exp = Nd4j.argMax(inArr.neg(), dim);   //argmin(x) == argmax(-x)

            assertEquals(exp, out);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScatterAdd(Nd4jBackend backend) {
        INDArray arr1 = Nd4j.zeros(3, 3);
        INDArray arr2 = Nd4j.createFromArray(0, 1);
        INDArray arr3 = Nd4j.ones(2, 3);
        INDArray expected = Nd4j.create(new float[]{1, 1, 1,
                        1, 1, 1,
                        0, 0, 0},
                new long[]{3, 3}).castTo(Nd4j.defaultFloatingPointType());

        SameDiff sd = SameDiff.create();
        SDVariable refs = sd.var("refs", arr1);
        SDVariable idxs = sd.constant("idxs", arr2);
        SDVariable upds = sd.placeHolder("upds", arr3.dataType(), arr3.shape());
        upds.setArray(arr3);

        SDVariable result = sd.scatterAdd(refs, idxs, upds);
        assertArrayEquals(new long[]{3, 3}, result.eval().shape());
        assertEquals(expected, result.eval());

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScatterMul(Nd4jBackend backend) {
        INDArray arr1 = Nd4j.ones(3, 3);
        INDArray arr2 = Nd4j.createFromArray(0, 1);
        INDArray arr3 = Nd4j.zeros(2, 3);
        INDArray expected = Nd4j.create(new float[]{0, 0, 0,
                        0, 0, 0,
                        1, 1, 1},
                new long[]{3, 3}).castTo(Nd4j.defaultFloatingPointType());

        SameDiff sd = SameDiff.create();
        SDVariable refs = sd.var("refs", arr1);
        SDVariable idxs = sd.constant("idxs", arr2);
        SDVariable upds = sd.placeHolder("upds", arr3.dataType(), arr3.shape());
        upds.setArray(arr3);

        SDVariable result = sd.scatterMul(refs, idxs, upds);
        assertArrayEquals(new long[]{3, 3}, result.eval().shape());
        assertEquals(expected, result.eval());

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScatterSub(Nd4jBackend backend) {
        INDArray arr1 = Nd4j.ones(3, 3);
        INDArray arr2 = Nd4j.createFromArray(0, 1);
        INDArray arr3 = Nd4j.ones(2, 3);
        INDArray expected = Nd4j.create(new float[]{0, 0, 0,
                        0, 0, 0,
                        1, 1, 1},
                new long[]{3, 3}).castTo(Nd4j.defaultFloatingPointType());

        SameDiff sd = SameDiff.create();
        SDVariable refs = sd.var("refs", arr1);
        SDVariable idxs = sd.constant("idxs", arr2);
        SDVariable upds = sd.placeHolder("upds", arr3.dataType(), arr3.shape());
        upds.setArray(arr3);

        SDVariable result = sd.scatterSub(refs, idxs, upds);
        assertArrayEquals(new long[]{3, 3}, result.eval().shape());
        assertEquals(expected, result.eval());

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScatterDiv(Nd4jBackend backend) {
        INDArray arr1 = Nd4j.ones(3, 3);
        INDArray arr2 = Nd4j.createFromArray(0, 1);
        INDArray arr3 = Nd4j.ones(2, 3).assign(2);
        INDArray expected = Nd4j.create(new float[]{0.5f, 0.5f, 0.5f,
                        0.5f, 0.5f, 0.5f,
                        1.0f, 1.0f, 1.0f},
                new long[]{3, 3}).castTo(Nd4j.defaultFloatingPointType());

        SameDiff sd = SameDiff.create();
        SDVariable refs = sd.var("refs", arr1);
        SDVariable idxs = sd.constant("idxs", arr2);
        SDVariable upds = sd.placeHolder("upds", arr3.dataType(), arr3.shape());
        upds.setArray(arr3);

        SDVariable result = sd.scatterDiv(refs, idxs, upds);
        assertArrayEquals(new long[]{3, 3}, result.eval().shape());
        assertEquals(expected, result.eval());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScatterMax(Nd4jBackend backend) {
        INDArray arr1 = Nd4j.ones(3, 3);
        INDArray arr2 = Nd4j.createFromArray(0, 1);
        INDArray arr3 = Nd4j.ones(2, 3).assign(2);
        INDArray expected = Nd4j.create(new float[]{2.0f, 2.0f, 2.0f,
                        2.0f, 2.0f, 2.0f,
                        1.0f, 1.0f, 1.0f},
                new long[]{3, 3}).castTo(Nd4j.defaultFloatingPointType());

        SameDiff sd = SameDiff.create();
        SDVariable refs = sd.var("refs", arr1);
        SDVariable idxs = sd.constant("idxs", arr2);
        SDVariable upds = sd.placeHolder("upds", arr3.dataType(), arr3.shape());
        upds.setArray(arr3);

        SDVariable result = sd.scatterMax(refs, idxs, upds);
        assertArrayEquals(new long[]{3, 3}, result.eval().shape());
        assertEquals(expected, result.eval());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScatterMin(Nd4jBackend backend) {
        INDArray arr1 = Nd4j.ones(3, 3);
        INDArray arr2 = Nd4j.createFromArray(1, 2);
        INDArray arr3 = Nd4j.ones(2, 3).assign(-2.0f);
        INDArray expected = Nd4j.create(new float[]{1.0f, 1.0f, 1.0f,
                        -2.0f, -2.0f, -2.0f,
                        -2.0f, -2.0f, -2.0f},
                new long[]{3, 3}).castTo(Nd4j.defaultFloatingPointType());

        SameDiff sd = SameDiff.create();
        SDVariable refs = sd.var("refs", arr1);
        SDVariable idxs = sd.constant("idxs", arr2);
        SDVariable upds = sd.placeHolder("upds", arr3.dataType(), arr3.shape());
        upds.setArray(arr3);

        SDVariable result = sd.scatterMin(refs, idxs, upds);
        assertArrayEquals(new long[]{3, 3}, result.eval().shape());
        assertEquals(expected, result.eval());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReciprocal(Nd4jBackend backend) {
        INDArray inArr = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray expected = Nd4j.onesLike(inArr).divi(inArr);
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", inArr);
        SDVariable reciprocal = sd.math().reciprocal(in);
        INDArray res = reciprocal.eval();
        assertEquals(expected, res);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGather2(Nd4jBackend backend) {

        INDArray in = Nd4j.rand(DataType.FLOAT, 10, 10);
        INDArray indices = Nd4j.createFromArray(0, 1, 5);

        SameDiff sd = SameDiff.create();

        SDVariable var = sd.var("in", in);
        SDVariable varIndices = sd.constant("indices", indices);
        SDVariable gather = sd.gather(var, varIndices, 0);

        INDArray exp = Nd4j.pullRows(in, 1, new int[]{0, 1, 5});  //Along dimension 1 -> equiv to "indexes for axis 0"
        INDArray act = gather.eval();

        assertEquals(exp, act);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGatherOp(Nd4jBackend backend) {

        INDArray in = Nd4j.rand(DataType.DOUBLE, 10, 10);
        INDArray indices = Nd4j.createFromArray(0, 1, 5);
        INDArray out = Nd4j.create(3, 10);

        DynamicCustomOp op = DynamicCustomOp.builder("gather")
                .addIntegerArguments(0) //Indexes are for dimension 0
                .addInputs(in, indices)
                .addOutputs(out)
                .build();

        Nd4j.exec(op);

        INDArray exp = Nd4j.pullRows(in, 1, new int[]{0, 1, 5});  //Along dimension 1 == indexes for dimension 0

        assertEquals(exp, out);

        //Shape function:
        val shapes = Nd4j.getExecutioner().calculateOutputShape(op);
        long[] expShape = new long[]{3, 10};

        assertEquals(1, shapes.size());

        assertArrayEquals(expShape, Shape.shape(shapes.get(0).asLong()));
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConditions(Nd4jBackend backend) {

        SameDiff sd = SameDiff.create();

        INDArray ia = Nd4j.create(new float[]{4, 2});
        SDVariable in = sd.var("in", 1, 2);
        sd.associateArrayWithVariable(ia, in);

        INDArray expFinite = Nd4j.create(new boolean[]{true, true});
        SDVariable finite = sd.math().isFinite(in);

        INDArray expInfinite = Nd4j.create(new boolean[]{false, false});
        SDVariable infinite = sd.math().isInfinite(in);

        INDArray expNaN = Nd4j.create(new boolean[]{false, false});
        SDVariable isnan = sd.math().isNaN(in);

        assertEquals(expFinite, finite.eval());
        assertEquals(expInfinite, infinite.eval());
        assertEquals(expNaN, isnan.eval());

    }


    private static int binArrToInt(int[] arr) {
        int x = 0;
        int m = 1;
        for (int i = arr.length - 1; i >= 0; i--) {
            if (arr[i] == 1) {
                x += m;
            }
            m *= 2;
        }
        return x;
    }




    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSDVariableLength(Nd4jBackend backend) {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Nd4j.ones(100);
        assertEquals(100,sameDiff.var(arr).length().eval().getInt(0));

        INDArray arr2 = Nd4j.ones(5,5);
        assertEquals(25,sameDiff.var(arr2).length().eval().getInt(0));

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGetVariable(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        INDArray arr = Nd4j.linspace(1, 100, 100).reshape('c', 10L, 10L);
        System.out.println(arr);
        SDVariable x = sd.var(arr);
        assertEquals(Nd4j.linspace(1,10,10),x.get(SDIndex.point(sd.constant(0).reshape(1))).eval());
        assertEquals(arr.get(NDArrayIndex.point(0),NDArrayIndex.point(1)),x.get(SDIndex.point(0),SDIndex.point(1)).eval());
        assertEquals(arr.get(NDArrayIndex.interval(0,2)),x.get(SDIndex.interval(0,2)).eval());
        assertEquals(arr.get(NDArrayIndex.interval(0,2)),x.get(SDIndex.interval(sd.constant(0).reshape(1),sd.constant(2).reshape(1))).eval());
        assertEquals(arr.get(NDArrayIndex.interval(0,2,2)),x.get(SDIndex.interval(sd.constant(0).reshape(1),sd.constant(2).reshape(1),sd.constant(2).reshape(1))).eval());

    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGetVariableView(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        INDArray arr = Nd4j.linspace(1, 100, 100).reshape('c', 10L, 10L);
        System.out.println(arr);
        SDVariable x = sd.var(arr);
        //assertEquals(Nd4j.linspace(1,10,10),x.getView(SDIndex.point(sd.constant(0).reshape(1))).eval());
        //assertEquals(arr.get(NDArrayIndex.point(0),NDArrayIndex.point(1)),x.getView(SDIndex.point(0),SDIndex.point(1)).eval());
        assertEquals(arr.get(NDArrayIndex.interval(0,2)),x.getView(SDIndex.interval(0,2)).eval());
        assertEquals(arr.get(NDArrayIndex.interval(0,2)),x.getView(SDIndex.interval(sd.constant(0).reshape(1),sd.constant(2).reshape(1))).eval());
        assertEquals(arr.get(NDArrayIndex.interval(0,2,2)),x.getView(SDIndex.interval(sd.constant(0).reshape(1),sd.constant(2).reshape(1),sd.constant(2).reshape(1))).eval());

    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testIndexInterval(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        INDArray arr = Nd4j.createFromArray(10,10,10);
        int jaxis = 1;
        SDVariable paramsShape = sd.var(arr);
        SDVariable innerShape = paramsShape.getView(
                SDIndex.interval(sd.constant(jaxis),sd.constant(-1)));

        assertEquals(Nd4j.createFromArray(10,10),innerShape.eval());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testIndexInterval2(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();

        // Create a linspace array with a shape of 2,2,5,5
        INDArray arr = Nd4j.linspace(1, 100, 100).reshape(2,2,5,5);
        INDArray expected = arr.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(1, 6, 1, false), NDArrayIndex.interval(0, 5, 1, false));

        // Create a SDVariable from the array
        SDVariable paramsShape = sd.var(arr);

        // Create an inner shape with given intervals
        SDVariable innerShape = paramsShape.getView(
                SDIndex.all(),
                SDIndex.all(),
                SDIndex.interval(sd.constant(1), sd.constant(6), sd.constant(1), sd.constant(0)),
                SDIndex.interval(sd.constant(0), sd.constant(5), sd.constant(1), sd.constant(0))
        );

        // Perform the evaluation
        INDArray result = innerShape.eval();

        // Assert that the result matches the expected result
        assertEquals(expected, result);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testIndexPoints(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();

        // Create a linspace array with a shape of 2,2,2,2,5,5
        INDArray arr = Nd4j.linspace(1, 400, 400).reshape(2,2,2,2,5,5);
        INDArray expected = arr.get(NDArrayIndex.point(0), NDArrayIndex.point(1));

        // Create a SDVariable from the array
        SDVariable paramsShape = sd.var(arr);

        // Create an inner shape with given points
        SDVariable innerShape = paramsShape.getView(
                SDIndex.point(sd.constant(0)),
                SDIndex.point(sd.constant(1))
        );

        // Perform the evaluation
        INDArray result = innerShape.eval();

        // Assert that the result matches the expected result
        assertEquals(expected, result);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testIndexRange(Nd4jBackend backend) {
        SameDiff sameDiff = SameDiff.create();
        SDVariable input = sameDiff.placeHolder("input",DataType.INT64);
        SDVariable range = sameDiff.range(sameDiff.constant(0), input.rank(), sameDiff.constant(1), DataType.INT64);
        //0 1 1
        SDVariable mask = range.gt(0.0).castTo(DataType.INT64);

        //1 0 0
        SDVariable sliceMask = range.eq(0).castTo(DataType.INT64);


        //2 0 0
        SDVariable sliceIndex = sliceMask.mul(3.0);

        //1 2 3 -> 0 2 3
        SDVariable outputShape = input.shape().mul(mask).add(sliceIndex);

        System.out.println(outputShape.eval(Collections.singletonMap("input",Nd4j.ones(1,2,3))));



    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testArrayIndices(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        INDArray arr = Nd4j.linspace(1, 100, 100).reshape('c', 10L, 10L);
        System.out.println(arr);
        SDVariable x = sd.var(arr);
        SDVariable get = x.get(sd.var(Nd4j.createFromArray(0,1,2,3,4)));
        INDArray assertion = Nd4j.linspace(1,50,50).reshape(5,10);
        assertEquals(assertion,get.eval());

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCreateView(Nd4jBackend backend) {
        Nd4j.getExecutioner().enableVerboseMode(true);
        Nd4j.getExecutioner().enableDebugMode(true);
        SameDiff sd = SameDiff.create();
        INDArray input = Nd4j.linspace(1,4,4).reshape(2,2);
        SDVariable newOne = sd.constant(Nd4j.linspace(1,4,4).reshape(2,2));
        SDVariable view = sd.createView(newOne,CreateView.createPoint(sd,1));
        INDArray eval = view.eval();
        assertEquals(input.getRow(1),eval);
        SDVariable putResult = view.put(sd.constant(1),sd.constant(Nd4j.ones(2)),sd.constant(0));
        System.out.println(putResult.eval());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCreateViewBp(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);
        Nd4j.getExecutioner().enableVerboseMode(true);
        Nd4j.getExecutioner().enableDebugMode(true);
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("in", DataType.FLOAT, 2, 3);
        SDVariable viewIn = sd.createView(in,CreateView.createPoint(sd,1));
        SDVariable expandDims = sd.expandDims(viewIn,0);
        SDVariable w = sd.var("w", Nd4j.rand(DataType.FLOAT, 3, 4));
        SDVariable b = sd.var("b", Nd4j.rand(DataType.FLOAT, 1, 4));
        SDVariable mmul = expandDims.mmul(w);
        SDVariable add = mmul.add(b);
        SDVariable tanh = sd.math().tanh(add);
        SDVariable loss = sd.variance(tanh, true);
        loss.markAsLoss();
        INDArray inArr = Nd4j.rand(DataType.FLOAT, 2, 3);
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
        Assertions.assertEquals(VariableType.CONSTANT, w.getVariableType());
        assertEquals(VariableType.VARIABLE, b.getVariableType());
        assertEquals(VariableType.ARRAY, add.getVariableType());
        assertEquals(VariableType.ARRAY, tanh.getVariableType());

        //Sanity check on training:
        sd.fit(new SingletonMultiDataSetIterator(new DataSet(inArr, null).toMultiDataSet()), 1);


    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testArrayIndicesPut(Nd4jBackend backend) {
        Nd4j.getExecutioner().enableVerboseMode(true);
        Nd4j.getExecutioner().enableDebugMode(true);
        SameDiff sd = SameDiff.create();
        INDArray arr = Nd4j.linspace(1, 100, 100).reshape('c', 10L, 10L);
        SDVariable x = sd.var(arr);
        SDVariable get = x.get(sd.var(Nd4j.createFromArray(0,1,2,3,4)));
        INDArray assertion = Nd4j.linspace(1,50,50).reshape(5,10);
        assertEquals(assertion,get.eval());

        SDVariable putInTo = sd.zerosLike(x);
        SDVariable putIndices = sd.range(sd.constant(0),sd.sizeAt(x,0),sd.constant(1),DataType.INT64);
        SDVariable put = putInTo.put(putIndices, x, putIndices);
        INDArray xEval = x.eval();
        INDArray putEval = put.eval();
        assertEquals(xEval,putEval);

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testArrayIndicesPut3d(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        INDArray arr = Nd4j.linspace(1, 125, 125).reshape('c', 5L,5L,5L);
        SDVariable x = sd.var(arr);
        SDVariable get = x.get(sd.var(Nd4j.createFromArray(0,1,2)));
        INDArray assertion = Nd4j.linspace(1,75,75).reshape(3,5,5);
        assertEquals(assertion,get.eval());

        SDVariable putInTo = sd.zerosLike(x);
        SDVariable putIndices = sd.range(sd.constant(0),sd.constant(5),sd.constant(1),DataType.INT64);
        SDVariable put = putInTo.put(putIndices, x, putIndices);
        INDArray eval = put.eval();
        assertEquals(arr,eval);

    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testViewAll(Nd4jBackend backend) {
        Nd4j.getExecutioner().enableDebugMode(true);
        Nd4j.getExecutioner().enableVerboseMode(true);
        SameDiff sd = SameDiff.create();
        INDArray arr = Nd4j.linspace(1, 100, 100).reshape('c', 10L, 10L);
        SDVariable x = sd.var(arr);

        SDVariable view = sd.createView(x, new SDVariable[]{CreateView.createAll(sd)});
        INDArray eval = view.eval();
        assertEquals(arr,eval);

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testViewInterval(Nd4jBackend backend) {
        Nd4j.getExecutioner().enableDebugMode(true);
        Nd4j.getExecutioner().enableVerboseMode(true);
        SameDiff sd = SameDiff.create();
        INDArray arr = Nd4j.linspace(1, 100, 100).reshape('c', 10L, 10L);
        SDVariable x = sd.var(arr);

        SDVariable view = sd.createView(x, new SDVariable[]{CreateView.createInterval(sd,0,1,1,1)});
        INDArray eval = view.eval();
        assertEquals(arr.get(NDArrayIndex.interval(0,1,true)),eval);

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNewAxis(Nd4jBackend backend) {
        Nd4j.getExecutioner().enableDebugMode(true);
        Nd4j.getExecutioner().enableVerboseMode(true);
        SameDiff sd = SameDiff.create();
        INDArray arr = Nd4j.linspace(1, 100, 100).reshape('c', 10L, 10L);
        SDVariable x = sd.var(arr);

        SDVariable view = sd.createView(x, new SDVariable[]{CreateView.createNewAxis(sd)});
        INDArray eval = view.eval();
        assertEquals(arr.get(NDArrayIndex.newAxis()),eval);

    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPoint(Nd4jBackend backend) {
        Nd4j.getExecutioner().enableDebugMode(true);
        Nd4j.getExecutioner().enableVerboseMode(true);
        SameDiff sd = SameDiff.create();
        INDArray arr = Nd4j.linspace(1, 100, 100).reshape('c', 10L, 10L);
        SDVariable x = sd.var(arr);

        SDVariable view = sd.createView(x, new SDVariable[]{CreateView.createPoint(sd,1)});
        INDArray eval = view.eval();
        assertEquals(arr.get(NDArrayIndex.point(1)),eval);

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGet(Nd4jBackend backend) {

        SameDiff sd = SameDiff.create();
        INDArray arr = Nd4j.linspace(1, 100, 100).reshape('c', 10L, 10L);
        SDVariable x = sd.var(arr);

        INDArray expOut1 = arr.get(NDArrayIndex.point(4), NDArrayIndex.point(5)).reshape();
        SDVariable result1 = x.get(SDIndex.point(4), SDIndex.point(5));
        assertEquals(expOut1, result1.eval());

        INDArray expOut2 = arr.get(NDArrayIndex.point(4), NDArrayIndex.all()).reshape(10);
        SDVariable result2 = x.get(SDIndex.point(4), SDIndex.all());
        assertEquals(expOut2, result2.eval());

        INDArray expOut3 = arr.get(NDArrayIndex.interval(3, 8)).reshape(5, 10);
        SDVariable result3 = x.get(SDIndex.interval(3, 8));
        assertEquals(expOut3, result3.eval());

        INDArray expOut4 = arr.get(NDArrayIndex.point(5), NDArrayIndex.interval(3, 8)).reshape(5);
        SDVariable result4 = x.get(SDIndex.point(5), SDIndex.interval(3, 8));
        assertEquals(expOut4, result4.eval());

        INDArray expOut5 = arr.get(NDArrayIndex.interval(5, 6), NDArrayIndex.all());
        SDVariable result5 = x.get(SDIndex.point(5, true), SDIndex.all());
        assertEquals(expOut5, result5.eval());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGetRank3(Nd4jBackend backend) {

        SameDiff sd = SameDiff.create();
        INDArray arr = Nd4j.linspace(1, 1000, 1000).reshape('c', 10, 10, 10);
        SDVariable x = sd.var(arr);

        INDArray y1 = arr.get(NDArrayIndex.point(2), NDArrayIndex.all(), NDArrayIndex.all());
        SDVariable s1 = x.get(SDIndex.point(2), SDIndex.all(), SDIndex.all());
        INDArray s1a = s1.eval();
        assertEquals(s1a, y1);

        INDArray y2 = arr.get(NDArrayIndex.all(), NDArrayIndex.point(2), NDArrayIndex.all());
        SDVariable s2 = x.get(SDIndex.all(), SDIndex.point(2), SDIndex.all());
        INDArray s2a = s2.eval();
        assertEquals(s2a, y2);

        INDArray y3 = arr.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(2));
        SDVariable s3 = x.get(SDIndex.all(), SDIndex.all(), SDIndex.point(2));
        INDArray s3a = s3.eval();
        assertEquals(s3a, y3);

        INDArray y4 = arr.get(NDArrayIndex.point(2), NDArrayIndex.all(), NDArrayIndex.interval(3, 5));
        SDVariable s4 = x.get(SDIndex.point(2), SDIndex.all(), SDIndex.interval(3, 5));
        INDArray s4a = s4.eval();
        assertEquals(s4a, y4);

        INDArray y5 = arr.get(NDArrayIndex.interval(3, 5), NDArrayIndex.point(2), NDArrayIndex.all());
        SDVariable s5 = x.get(SDIndex.interval(3, 5), SDIndex.point(2), SDIndex.all());
        INDArray s5a = s5.eval();
        assertEquals(s5a, y5);

        INDArray y6 = arr.get(NDArrayIndex.all(), NDArrayIndex.interval(3, 5), NDArrayIndex.point(2));
        SDVariable s6 = x.get(SDIndex.all(), SDIndex.interval(3, 5), SDIndex.point(2));
        INDArray s6a = s6.eval();
        assertEquals(s6a, y6);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTensorArray1(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        TensorArray tensorArray = sd.tensorArray(DataType.FLOAT);
        INDArray arr1 = Nd4j.create(new double[]{1, 2, 3, 4}, new int[]{2, 2});
        SDVariable var1 = sd.var(arr1);
        INDArray arr2 = Nd4j.create(new double[]{5, 6, 7, 8}, new int[]{2, 2});
        SDVariable var2 = sd.var(arr2);
        SDVariable write0 = tensorArray.write(var2, 0, var1);
        SDVariable write1 = tensorArray.write(write0, 1, var2);
        SDVariable result = tensorArray.stack(write1);
        sd.output((Map<String,INDArray>)null, result.name());
        assertEquals(Nd4j.pile(arr1, arr2), result.eval());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTensorArray2(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        TensorArray tensorArray = sd.tensorArray(DataType.FLOAT);
        INDArray arr1 = Nd4j.create(new double[]{1, 2, 3, 4}, new int[]{2, 2});
        SDVariable var1 = sd.var(arr1);
        INDArray arr2 = Nd4j.create(new double[]{5, 6, 7, 8}, new int[]{2, 2});
        SDVariable var2 = sd.var(arr2);
        SDVariable write1 = tensorArray.write(var2, 0, var1);
        SDVariable write2 = tensorArray.write(write1, 1, var2);
        SDVariable result1 = tensorArray.read(0);
        SDVariable result2 = tensorArray.read(1);

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTensorArray3(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        TensorArray tensorArray = sd.tensorArray(DataType.FLOAT);
        INDArray arr1 = Nd4j.create(new double[]{1, 2, 3, 4}, new int[]{2, 2});
        INDArray arr2 = Nd4j.create(new double[]{5, 6, 7, 8}, new int[]{2, 2});
        INDArray arr3 = Nd4j.pile(arr1, arr2);
        SDVariable var = sd.var(arr3);
        SDVariable unstack = tensorArray.unstack(var, var);
        SDVariable result1 = tensorArray.read(0);
        SDVariable result2 = tensorArray.read(1);
        result1.addControlDependency(unstack);
        result2.addControlDependency(unstack);
        assertEquals(arr1, result1.eval());
        assertEquals(arr2, result2.eval());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testFill(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        INDArray shape = Nd4j.createFromArray(2, 2);
        INDArray expOut = Nd4j.valueArrayOf(new int[]{2, 2}, 42.0);
        SDVariable x = sd.constant(shape);
        SDVariable result = sd.fill(x, DataType.DOUBLE, 42);
        assertEquals(expOut, result.eval());
    }

    private static <T> T getObject(String fieldName, Object from, Class<?> fromClass) {
        try {
            Field f = fromClass.getDeclaredField(fieldName);
            f.setAccessible(true);
            return (T) f.get(from);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPermute(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        INDArray arr = Nd4j.create(new double[]{
                        /////////////
                        1, 2, 3, 4,
                        5, 6, 7, 8,
                        9, 10, 11, 12,
                        //////////////
                        13, 14, 15, 16,
                        17, 18, 19, 20,
                        21, 22, 23, 24
                        /////////////
                },
                new int[]{2, 3, 4});

        INDArray expOut = Nd4j.create(new double[]{
                        /////////////
                        1, 2, 3, 4,
                        13, 14, 15, 16,
                        /////////////
                        5, 6, 7, 8,
                        17, 18, 19, 20,
                        /////////////
                        9, 10, 11, 12,
                        21, 22, 23, 24
                        /////////////
                },
                new int[]{3, 2, 4});

        SDVariable x = sd.var(arr);
        SDVariable result = sd.permute(x, 1, 0, 2);
        assertEquals(expOut, result.eval());

    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testExecutionDifferentShapesAccumAlongDim(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", Nd4j.linspace(1, 12, 12).reshape(3, 4));

        SDVariable sum = in.sum(1);
        INDArray exp = in.getArr().sum(1).reshape(3);

        INDArray out = sum.eval();
        assertEquals(exp, out);

        //Now, replace with minibatch 5:
        in.setArray(Nd4j.linspace(1, 20, 20).reshape(5, 4));
        INDArray out2 = sum.eval();
        assertArrayEquals(new long[]{5}, out2.shape());

        exp = in.getArr().sum(1).reshape(5);
        assertEquals(exp, out2);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testExecutionDifferentShapesIndexAccumAlongDim(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", Nd4j.linspace(1, 12, 12).reshape(3, 4));

        SDVariable sum = in.argmax(1);
        INDArray exp = in.getArr().argMax(1).reshape(3);

        INDArray out = sum.eval();
        assertEquals(exp, out);

        //Now, replace with minibatch 5:
        in.setArray(Nd4j.linspace(1, 20, 20).reshape(5, 4));
        INDArray out2 = sum.eval();
        assertArrayEquals(new long[]{5}, out2.shape());

        exp = in.getArr().argMax(1).reshape(5);
        assertEquals(exp, out2);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Disabled
    public void testExternalErrorsSimple(Nd4jBackend backend) {
        INDArray externalGrad = Nd4j.linspace(1, 12, 12).reshape(3, 4);

        SameDiff sd = SameDiff.create();
        SDVariable var = sd.var("var", externalGrad);
        SDVariable out = var.mul("out", 0.5);

        Map<String, INDArray> gradMap = new HashMap<>();
        gradMap.put("out", externalGrad);
        ExternalErrorsFunction fn = SameDiffUtils.externalErrors(sd, null, out);

        Map<String, INDArray> m = new HashMap<>();
        m.put("out-grad", externalGrad);
        Map<String, INDArray> grads = sd.calculateGradients(m, sd.getVariables().keySet());

        INDArray gradVar = grads.get(var.name());

        assertEquals(externalGrad.mul(0.5), gradVar);

        //Now, update and execute again:
        externalGrad = Nd4j.linspace(1, 12, 12).reshape(3, 4).muli(10);

        m.put("out-grad", externalGrad);
        grads = sd.calculateGradients(m, sd.getVariables().keySet());

        gradVar = var.getGradient().getArr();

        assertEquals(externalGrad.mul(0.5), gradVar);

        //Test model serialization:
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testUpdatingGradient(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", Nd4j.linspace(1, 12, 12).reshape(3, 4));
        SDVariable w = sd.var("w", Nd4j.linspace(1, 20, 20).reshape(4, 5));
        SDVariable out = sd.mmul(in, w);
        SDVariable loss = out.std("out", true);

        INDArray outArr = loss.eval();
        Map<String,INDArray> grads = sd.calculateGradients(null, in.name(), w.name(), out.name());

        Map<String, INDArray> origGrad = new HashMap<>();
        origGrad.put("in", grads.get(in.name()).dup());
        origGrad.put("w", grads.get(w.name()).dup());
        origGrad.put("out", grads.get(out.name()).dup());

        in.getArr().assign(Nd4j.rand(in.getArr().shape()));
        INDArray outArr2 = loss.eval();
        grads = sd.calculateGradients(null, in.name(), w.name(), out.name());

        assertNotEquals(outArr, outArr2);

        //Ensure gradients are also changed:
        assertNotEquals(origGrad.get("in"), grads.get(in.name()));
        assertNotEquals(origGrad.get("w"), grads.get(w.name()));
        assertNotEquals(origGrad.get("out"), grads.get(out.name()));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testUpdatingGradientSimple(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", Nd4j.linspace(1, 12, 12).reshape(3, 4));
        SDVariable out = in.mul(2.0);
        SDVariable loss = out.std("out", true);

        INDArray outArr = loss.eval();
        Map<String,INDArray> grads = sd.calculateGradients(null, in.name(), out.name());

        Map<String, INDArray> origGrad = new HashMap<>();
        origGrad.put("in", grads.get(in.name()).dup());
        origGrad.put("out", grads.get(out.name()).dup());

        double stdBefore = in.getArr().stdNumber().doubleValue();
        in.getArr().assign(Nd4j.rand(in.getArr().shape()));
        double stdAfter = in.getArr().stdNumber().doubleValue();
        System.out.println("Before vs. after: " + stdBefore + ", " + stdAfter);
        INDArray outArr2 = loss.eval();
        grads = sd.calculateGradients(null, in.name(), out.name());

        assertNotEquals(outArr, outArr2);

        //Ensure gradients are also changed:
        assertNotEquals(origGrad.get("in"), grads.get(in.name()));
        assertNotEquals(origGrad.get("out"), grads.get(out.name()));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Disabled
    public void testShapeUpdating(Nd4jBackend backend) {

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", DataType.FLOAT, 3, 5);
        SDVariable w = sd.var("W", DataType.FLOAT, 5, 4);
        SDVariable b = sd.var("b", DataType.FLOAT, 1, 4);
        SDVariable z = in.mmul(w).add(b);
        SDVariable out = sd.math().tanh("tanh", z);
        ExternalErrorsFunction fn = SameDiffUtils.externalErrors(sd, null, out);

        INDArray inA = Nd4j.linspace(1, 15, 15, DataType.FLOAT).reshape(3, 5);
        INDArray wA = Nd4j.linspace(1, 20, 20, DataType.FLOAT).reshape(5, 4);
        INDArray bA = Nd4j.linspace(1, 4, 4, DataType.FLOAT);
        in.setArray(inA);
        w.setArray(wA);
        b.setArray(bA);

        INDArray grad = Nd4j.linspace(1, 12, 12, DataType.FLOAT).reshape(3, 4);
        Map<String, INDArray> phMap = new HashMap<>();
        phMap.put(fn.getGradPlaceholderName(), grad);

        out.eval();
        sd.calculateGradients(phMap, "in", "W", "b");


        sd.getFunction("grad").summary();

        in.setArray(Nd4j.linspace(1, 10, 10).reshape(2, 5));
        grad = Nd4j.linspace(1, 8, 8).reshape(2, 4);
        phMap.put(fn.getGradPlaceholderName(), grad);

        Map<String,INDArray> grads = sd.calculateGradients(phMap, sd.getVariables().keySet());
        INDArray inGrad = grads.get(in.name());
        assertArrayEquals(new long[]{2, 5}, inGrad.shape());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMultiOutput1(Nd4jBackend backend) {

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", Nd4j.create(3, 4));
        SDVariable mean = in.mean();
        SDVariable sum = in.sum();

        try {
            sd.createGradFunction();
            fail("Expected exception");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("No loss variables"),e.getMessage());
        }

        SDVariable add = mean.add(sum);
        sd.createGradFunction();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMultiOutput2(Nd4jBackend backend) {
        //Edge case: no functions
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", Nd4j.scalar(0.0));
        SDVariable in2 = sd.var("in2", Nd4j.scalar(1.0));

        try {
            sd.createGradFunction();
            fail("Expected exception");
        } catch (IllegalStateException e) {
            assertTrue( e.getMessage().contains("No loss variables"),e.getMessage());
        }

        SDVariable add = in.add(in2);
        sd.createGradFunction();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void sameDiffPlaceholderGrad(Nd4jBackend backend) {
        INDArray x = Nd4j.ones(2, 2);
        INDArray y = Nd4j.ones(2, 2);

        SameDiff sd = SameDiff.create();

        SDVariable xSd = sd.var("x", DataType.FLOAT, x.shape());
        SDVariable ySd = sd.var("y", DataType.FLOAT, y.shape());

        SDVariable add = ySd.add("add", xSd);

        Map<String, INDArray> placeholders = new HashMap<>();
        placeholders.put("x", x);
        placeholders.put("y", y);
        Map<String,INDArray> grads = sd.calculateGradients(placeholders, xSd.name(), ySd.name());
        INDArray xGradientEnforced = grads.get("x");
        assertNotNull(xGradientEnforced);
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
        Assertions.assertEquals(VariableType.CONSTANT, w.getVariableType());
        assertEquals(VariableType.VARIABLE, b.getVariableType());
        assertEquals(VariableType.ARRAY, add.getVariableType());
        assertEquals(VariableType.ARRAY, tanh.getVariableType());

        //Sanity check on training:
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

        //Sanity check on fitting:
        sd.fit(new SingletonMultiDataSetIterator(new MultiDataSet(new INDArray[]{inArr2}, null)), 1);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConvertToVariable(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("in", DataType.FLOAT, 1, 3);
        INDArray const1 =  Nd4j.rand(DataType.FLOAT, 3, 4);
        SDVariable w = sd.constant("w",const1);
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

        //Sanity check on training:
        sd.fit(new SingletonMultiDataSetIterator(new DataSet(inArr, null).toMultiDataSet()), 1);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDoubleUseOfArray(Nd4jBackend backend) {
        //If array is reused, gradient check will fail
        INDArray a = Nd4j.rand(DataType.DOUBLE, new int[]{3, 4});
        SameDiff sd = SameDiff.create();
        SDVariable a1 = sd.var("a", a);
        SDVariable a2 = sd.var("b", a);
        a1.add(a2).norm2("out");
        String err = OpValidation.validate(new TestCase(sd)
                .gradientCheck(true));
        assertNull(err);

        a1.setArray(a);
        a2.setArray(a);
        err = OpValidation.validate(new TestCase(sd)
                .gradientCheck(true));
        assertNull(err);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMultiGradientRecurrent(Nd4jBackend backend) {
        final INDArray input = Nd4j.rand(DataType.DOUBLE, new int[]{3, 4, 2});
        final INDArray[] output = new INDArray[(int) input.size(2)];
        for (int i = 0; i < input.size(2); i++) {
            final INDArray x_i = input.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(i));

            output[i] = x_i;
            if (i > 0) {
                output[i] = output[i].add(Nd4j.squeeze(output[i - 1], 2));
            }

            output[i] = Nd4j.expandDims(output[i], 2);
        }
        final INDArray out = Nd4j.concat(2, output).norm2();

        SameDiff sd = SameDiff.create();
        final SDVariable sdInput = sd.var("input", input);

        final long timeSteps = sdInput.getShape()[2];
        SDVariable[] outputSlices = new SDVariable[(int) timeSteps];
        SDVariable prev = null;
        for (int i = 0; i < timeSteps; i++) {
            final val x_i = sdInput.get(SDIndex.all(), SDIndex.all(), SDIndex.point(i));

            outputSlices[i] = x_i;
            if (prev != null) {
                outputSlices[i] = outputSlices[i].add(sd.squeeze(prev, 2));
            }

            outputSlices[i] = sd.expandDims(outputSlices[i], 2);
            prev = outputSlices[i];
        }

        SDVariable t = sd.concat(2, outputSlices);
        t.norm2("out");
        String err = OpValidation.validate(new TestCase(sd)
                .testFlatBufferSerialization(TestCase.TestSerialization.BOTH)
                .expectedOutput("out", out)
                .gradientCheck(true));

        assertNull(err);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMultiGradientManualRecurrent(Nd4jBackend backend) {
        final INDArray input = Nd4j.rand(DataType.DOUBLE, new int[]{3, 4, 2});
        final INDArray[] output = new INDArray[(int) input.size(2)];
        for (int i = 0; i < input.size(2); i++) {
            final INDArray x_i = input.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(i));

            output[i] = x_i;
            if (i > 0) {
                output[i] = output[i].add(Nd4j.squeeze(output[i - 1], 2));
            }

            output[i] = Nd4j.expandDims(output[i], 2);
        }
        final INDArray out = Nd4j.concat(2, output).norm2();

        SameDiff sd = SameDiff.create();
        final SDVariable sdInput = sd.var("input", input);

        final long timeSteps = sdInput.getShape()[2];
        SDVariable[] outputSlices = new SDVariable[(int) timeSteps];
        final SDVariable[] inputSlices = sd.unstack(new String[]{"X_0", "X_1"}, sdInput, 2, 2);

        final val x_0 = inputSlices[0];
        outputSlices[0] = x_0;
        outputSlices[0] = sd.expandDims("X_0-e", outputSlices[0], 2);

        final val x_1 = inputSlices[1];
        outputSlices[1] = x_1;
        outputSlices[1] = outputSlices[1].add(sd.squeeze("X_0-s", outputSlices[0], 2));
        outputSlices[1] = sd.expandDims("X_1-e", outputSlices[1], 2);

        SDVariable t = sd.concat(2, outputSlices);
        t.norm2("out");
        String err = OpValidation.validate(new TestCase(sd)
                .testFlatBufferSerialization(TestCase.TestSerialization.BOTH)
                .expectedOutput("out", out)
                .gradientCheck(true));

        assertNull(err);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMultiGradient(Nd4jBackend backend) {
        final INDArray input = Nd4j.rand(DataType.DOUBLE, new int[]{3, 4, 2});
        SameDiff sd = SameDiff.create();
        final SDVariable sdInput = sd.var("input", input);

        final SDVariable[] inputSlices = sd.unstack(new String[]{"X_0", "X_1"}, sdInput, 2, 2);
        final val temp = inputSlices[0].add(inputSlices[1]).div(inputSlices[1]).mul(inputSlices[0]);
        final val out = temp.add(temp).add(inputSlices[1]);
        out.norm2("out");

        String err = OpValidation.validate(new TestCase(sd)
                .testFlatBufferSerialization(TestCase.TestSerialization.BOTH)
                .gradientCheck(true));

        assertNull(err);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNonScalarOutput1(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable linspace = sd.linspace("at", DataType.DOUBLE, 1, 15, 15);
        SDVariable a = sd.reshape("a", linspace, 3, 5);
        SDVariable b = sd.var("b", Nd4j.ones(DataType.DOUBLE, 3, 5));

        SDVariable out = a.mul(b);
        out.markAsLoss();
        out.eval();

        out.eval();
        sd.grad("a").eval();

        String err = OpValidation.validate(new TestCase(sd)
                .testFlatBufferSerialization(TestCase.TestSerialization.BOTH)
                .gradientCheck(true));

        assertNull(err);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNonScalarOutput2(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.reshape("a", sd.linspace("at", DataType.DOUBLE, 1, 15, 15), 3, 5);
        SDVariable b = sd.var("b", Nd4j.ones(DataType.DOUBLE, 3, 5));

        SDVariable out = a.mul(b).mean(1);
        out.markAsLoss();
        out.eval();

        //System.out.println(out.eval());
        INDArray actGrad = sd.grad("a").eval();

        INDArray expGrad = Nd4j.valueArrayOf(new long[]{3, 5}, 0.2, DataType.DOUBLE);
        assertEquals(expGrad, actGrad);

        String err = OpValidation.validate(new TestCase(sd).gradientCheck(true));
        assertNull(err);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNonScalarOutput3(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.reshape("a", sd.linspace("at", DataType.DOUBLE, 1, 15, 15), 3, 5);
        SDVariable b = sd.var("b", Nd4j.ones(DataType.DOUBLE, 3, 5));//.add(3);

        SDVariable out = a.mul(b).mean(0, 1);
        out.markAsLoss();

        out.eval();

        Map<String,INDArray> g = sd.calculateGradients(null, "a");
        //System.out.println(out.eval());
        INDArray gradAct = g.get("a");
        INDArray expGrad = Nd4j.valueArrayOf(new long[]{3, 5}, 1.0 / 12, DataType.DOUBLE);

        String err = OpValidation.validate(new TestCase(sd).gradientCheck(true));
        assertNull(err);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNonScalarOutput4(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.var("a", DataType.DOUBLE, 3, 4);
        SDVariable b = sd.placeHolder("b", DataType.DOUBLE, 4, 5);
        a.setArray(Nd4j.rand(DataType.DOUBLE, 3, 4));

        SDVariable out = a.mmul("mmul", b);

        Map<String, INDArray> m = new HashMap<>();
        m.put("b", Nd4j.rand(DataType.DOUBLE, 4, 5));
        Map<String,INDArray> g = sd.calculateGradients(m, "a", "b");

        b.setArray(m.get("b"));

        String err = OpValidation.validate(new TestCase(sd)
                .testFlatBufferSerialization(TestCase.TestSerialization.BOTH)
                .gradientCheck(true));

        assertNull(err);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNonScalarOutput5(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable linspace = sd.linspace(DataType.DOUBLE, 1, 75, 75);
        SDVariable a = sd.reshape("a", linspace, 15, 5);
        SDVariable b = sd.var("b", Nd4j.ones(DataType.DOUBLE, 15, 5));

        SDVariable out = a.mul(b);
        out.markAsLoss();
        out.eval();

        INDArray outEvaled = out.eval();
        INDArray gradOutput = sd.grad("a").eval();
        INDArray bOutputEval = sd.grad("b").eval();
        String err = OpValidation.validate(new TestCase(sd)
                .testFlatBufferSerialization(TestCase.TestSerialization.BOTH)
                .gradientCheck(true));

        assertNull(err);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSameDiffBackprop1(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        final SDVariable a = sd.var("a", Nd4j.rand(4, 4));
        final SDVariable b = sd.var("b", Nd4j.rand(4, 4));
        final SDVariable c = sd.var("c", Nd4j.rand(4, 4));
        final SDVariable d = sd.var("d", Nd4j.rand(4, 4));

        final SDVariable out = a.mmul(b).add(c.mmul(d)).sum();
        out.markAsLoss();

        Map<String,INDArray> g = sd.calculateGradients(null, sd.getVariables().keySet());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSameDiffNoGradForConstantAndPlaceholder(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        final SDVariable a = sd.var("a", Nd4j.rand(4, 4));
        final SDVariable b = sd.constant("b", Nd4j.rand(4, 4));
        final SDVariable c = sd.placeHolder("c", Nd4j.dataType(), 4, 4);

        a.add(b.add(c)).sum().markAsLoss();

        sd.calculateGradients(Collections.singletonMap("c", Nd4j.rand(4, 4)), sd.getVariables().keySet());
        assertNotNull(sd.grad("a"));
        assertNull(sd.grad("b"));
        assertNull(sd.grad("c"));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDuplicateNamePlaceholder(Nd4jBackend backend) {

        for (int i = 0; i < 2; i++) {
            SameDiff sd = SameDiff.create();
            SDVariable x1 = i == 0 ? sd.placeHolder("a", DataType.FLOAT, 5, 3) : sd.var("a", DataType.FLOAT, 5, 3);
            SDVariable x2 = i == 0 ? sd.placeHolder("b", DataType.FLOAT, 5, 3) : sd.var("b", DataType.FLOAT, 5, 3);
            try {
                sd.placeHolder("a", DataType.FLOAT, 5, 3);
                fail("Expected exception");
            } catch (Throwable t) {
                String m = t.getMessage();
                assertNotNull(m);
            }

            try {
                sd.var("a", DataType.FLOAT, 1, 2);
                fail("Expected exception");
            } catch (Throwable t) {
                String m = t.getMessage();
                assertNotNull(m);
                assertTrue(m.contains("already exists"),m);
            }

            try {
                sd.var("a", Nd4j.zeros(1));
                fail("Expected exception");
            } catch (Throwable t) {
                String m = t.getMessage();
                assertNotNull(m);
                assertTrue(m.contains("already exists"),m);
            }

            try {
                sd.var("a", LongShapeDescriptor.fromShape(new long[]{1}, DataType.FLOAT));
                fail("Expected exception");
            } catch (Throwable t) {
                String m = t.getMessage();
                assertNotNull(m);
                assertTrue(m.contains("already exists"),m);
            }

            try {
                sd.constant("a", Nd4j.zeros(1));
                fail("Expected exception");
            } catch (Throwable t) {
                String m = t.getMessage();
                assertNotNull(m);
                assertTrue(m.contains("already exists"),m);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSameDiffGetArrayScalar(Nd4jBackend backend) {
        final INDArray array = Nd4j.rand(1, 1);
        final SameDiff sd = SameDiff.create();
        final SDVariable a = sd.var("a", array.shape());
        a.getArr();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testVariableRenaming(Nd4jBackend backend) {

        SameDiff sd = SameDiff.create();
        SDVariable v1 = sd.var("x", Nd4j.rand(DataType.FLOAT, 3, 4));
        SDVariable v2 = sd.var("y", Nd4j.rand(DataType.FLOAT, 4, 5));
        SDVariable v3 = v1.mmul("oldName", v2);

        INDArray out = sd.outputSingle(null, "oldName");

        SDVariable renamed = v3.rename("newName");
        assertTrue(v3 == renamed);
        assertEquals("newName", renamed.name());

        assertNull(sd.getVariable("oldName"));
        assertNotNull(sd.getVariable("newName"));

        INDArray out2 = sd.outputSingle(null, "newName");

        assertEquals(out, out2);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testVariableRenaming2(Nd4jBackend backend) {

        SameDiff sd = SameDiff.create();
        SDVariable v1 = sd.placeHolder("x", DataType.FLOAT, 3, 4);
        SDVariable v2 = sd.var("y", Nd4j.rand(DataType.FLOAT, 4, 5));
        SDVariable v3 = v1.mmul("oldName", v2);
        SDVariable v4 = v3.std("out", false);
        v4.markAsLoss();
        INDArray out = sd.outputSingle(Collections.singletonMap("x", Nd4j.rand(DataType.FLOAT, 3, 4)), "out");

        sd.setTrainingConfig(TrainingConfig.builder()
                .updater(new Adam(1e-3))
                .dataSetFeatureMapping("x")
                .markLabelsUnused()
                .build());

        sd.fit(new DataSet(Nd4j.rand(DataType.FLOAT, 3, 4), null));
        v3.rename("newName");
        sd.fit(new DataSet(Nd4j.rand(DataType.FLOAT, 3, 4), null));
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
                            .contains(Arrays.toString(v.placeholderShape())),msg);
                }
            }

            try {
                v.setArray(wrongRank1);
                fail("Expected exception");
            } catch (Exception t) {
                String msg = t.getMessage();
                assertTrue(msg.contains("shape") && msg.contains("[1]") && msg
                        .contains(Arrays.toString(v.placeholderShape())),msg);
            }

            try {
                v.setArray(wrongRank2);
                fail("Expected exception");
            } catch (Exception t) {
                String msg = t.getMessage();
                assertTrue(msg.contains("shape") && msg.contains("[3, 4, 5]") && msg
                        .contains(Arrays.toString(v.placeholderShape())),msg);
            }
        }

        //Also try training:
        SDVariable sum = sd.math.mergeAdd(new SDVariable[]{ph1, ph2, ph3, ph4});
        SDVariable mean = sum.add(scalar).mean();
        mean.markAsLoss();
        MultiDataSet mds = new MultiDataSet(new INDArray[]{wrongShape, wrongShape, wrongShape, wrongShape}, null);

        sd.setTrainingConfig(TrainingConfig.builder()
                .dataSetFeatureMapping("ph1", "ph2", "ph3", "ph4")
                .markLabelsUnused()
                .updater(new Adam(1e-3)).build());

        try {
            sd.fit(mds);
        } catch (Exception t) {
            String msg = t.getMessage();
            assertTrue( msg.contains("shape") && msg.contains("[2, 3]"),msg);
        }
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testInferenceWithoutLabel(Nd4jBackend backend) {
        //We don't need a value for the label placeholder to calculate most values here

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
        //We don't need an array for 2 of the placeholders to calculate the

        SameDiff sd = SameDiff.create();

        int nIn = 4;
        int minibatch = 3;
        SDVariable input = sd.placeHolder("in", DataType.FLOAT, -1, 4);
        SDVariable label = sd.placeHolder("label", DataType.FLOAT, -1, 3);

        SDVariable input2 = sd.placeHolder("in2", DataType.FLOAT);    //Scalar

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

        Map<String, INDArray> out = sd.output((Map<String,INDArray>)null, "x", "y", "z", "tanh", "stdev");
        for (Map.Entry<String, INDArray> e : out.entrySet()) {
            assertEquals(DataType.FLOAT, e.getValue().dataType(),e.getKey());
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

        out = sd.output((Map<String,INDArray>)null, "x", "y", "z", "tanh", "stdev");
        for (Map.Entry<String, INDArray> e : out.entrySet()) {
            assertEquals(DataType.DOUBLE, e.getValue().dataType(),e.getKey());
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
                assertEquals(DataType.FLOAT, e.getValue().dataType(),e.getKey());
            } else {
                assertEquals(DataType.DOUBLE, e.getValue().dataType(),e.getKey());
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

        out = sd.output(ph, "x", "y", "xD", "yD", "a", "r");
        for (Map.Entry<String, INDArray> e : out.entrySet()) {
            assertEquals(DataType.DOUBLE, e.getValue().dataType(),e.getKey());
        }

        assertEquals(DataType.DOUBLE, y.getArr().dataType());
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGradFnRequiredVars(Nd4jBackend backend) {
        //User can explicitly request that gradients for specific vars are available when differentiating (creating grad function),
        // even if they normally wouldn't be needed or calculated

        for (boolean reqPhVar : new boolean[]{false, true}) {
//        for(boolean reqPhVar : new boolean[]{true}){

            SameDiff sd = SameDiff.create();
            SDVariable ph = sd.placeHolder("in", DataType.FLOAT, -1, 5);
            SDVariable add = ph.add(1.0);
            SDVariable w = sd.var("w", Nd4j.rand(DataType.FLOAT, 5, 4));
            SDVariable b = sd.var("b", Nd4j.rand(DataType.FLOAT, 1, 4));

            SDVariable mmul = add.mmul(w).add(b);

            SDVariable loss = mmul.std(true);

            INDArray in = Nd4j.rand(DataType.FLOAT, 1, 5);

            if (reqPhVar) {
                sd.createGradFunction("in");
                assertNotNull(ph.gradient());
                assertNotNull(w.gradient());
                assertNotNull(b.gradient());

                Map<String,INDArray> m = sd.calculateGradients(Collections.singletonMap("in", in), ph.name(), w.name());
                assertNotNull(m.get(ph.name()));
                assertNotNull(m.get(w.name()));
            } else {
                sd.createGradFunction();
                assertNull(ph.gradient());
                assertNotNull(w.gradient());
                assertNotNull(b.gradient());
            }
        }


    }

    @Test
    public void testBroadcastingOr() {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.constant(Nd4j.createFromArray(true, false, false, true).reshape(2, 2));
        sd.constant(42); // added statement
        SDVariable b = sd.constant(Nd4j.createFromArray(false, false).reshape(1, 2));
        SDVariable result = sd.math().or(a, b);
        INDArray eval = result.eval();
        INDArray assertion = Nd4j.createFromArray(new boolean[][]{
                {true,false},
                {false,true}
        });
        System.out.println(eval);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testIf() throws IOException {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.placeHolder("a", DataType.DOUBLE);
        SDVariable b = sd.var("b", Nd4j.createFromArray(5.0));
        SDVariable c = sd.var("c", Nd4j.createFromArray(9.0));

        SDVariable output = sd.ifCond("out", null, s -> a.lt(b), s -> c, s -> c.add(5));

        Map<String, INDArray> firstBranch = Maps.newHashMap();
        firstBranch.put("a", Nd4j.createFromArray(3.0));
        assertEquals(Nd4j.createFromArray(9.0), sd.output(firstBranch, "out").get("out"));

        Map<String, INDArray> secondBranch = Maps.newHashMap();
        secondBranch.put("a", Nd4j.createFromArray(7.0));
        System.out.println(sd.summary());
        INDArray outArr = sd.output(secondBranch, "out").get("out");
        assertEquals(Nd4j.createFromArray(14.0), outArr);

        ByteBuffer bb = sd.asFlatBuffers(false);
        sd = SameDiff.fromFlatBuffers(bb);

        assertEquals(Nd4j.createFromArray(9.0), sd.output(firstBranch, "out").get("out"));
        assertEquals(Nd4j.createFromArray(14.0), sd.output(secondBranch, "out").get("out"));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNestedIf() throws IOException {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.var("a", Nd4j.createFromArray(2.0));
        SDVariable b = sd.var("b", Nd4j.createFromArray(5.0));
        SDVariable c = sd.var("c", Nd4j.createFromArray(9.0));
        SDVariable d = sd.var("d", Nd4j.createFromArray(-7.0));

        SDVariable output = sd.ifCond("out", null,
                (s) -> a.lt(b),
                (s) -> s.ifCond(
                        (sd2) -> d.lte(0),
                        (sd2) -> c.add(1),
                        (sd2) -> d),
                (s) -> c.add(5));
        INDArray out = output.eval();
        assertEquals(Nd4j.createFromArray(10.0), out);

        sd = SameDiff.fromFlatBuffers(sd.asFlatBuffers(false));

        assertEquals(Nd4j.createFromArray(10.0), sd.output(Collections.emptyMap(), "out").get("out"));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWhile() throws IOException {

        SameDiff sd = SameDiff.create();
        SDVariable countIn = sd.constant(5);
        SDVariable sumIn = sd.constant(0);

        SDVariable[] sum = sd.whileLoop("while_1", new SDVariable[]{countIn, sumIn},
                (s, vars) -> vars[0].gt(0),
                (s, vars) -> new SDVariable[]{vars[0].sub(1), vars[1].add(vars[0])});

        INDArray out = sum[1].eval();
        assertEquals(15, out.getInt(0));

        String outName = sum[1].name();

        sd = SameDiff.fromFlatBuffers(sd.asFlatBuffers(false));

        assertEquals(15, sd.output(Collections.emptyMap(), outName).get(outName).getInt(0));
    }

    @Test
    @Disabled
    public void testNestedWhile() throws IOException {
        SameDiff sd = SameDiff.create();
        SDVariable countIn = sd.constant(5);
        SDVariable sumIn = sd.constant(0);
        SDVariable sum2 = sd.constant(0);
        //TODO creating constant instead of using sum2 causes errors

        SDVariable[] sum = sd.whileLoop(new SDVariable[]{countIn, sumIn},
                (s, vars) -> vars[0].gt(0),
                (s, vars) -> new SDVariable[]{vars[0].sub(1),
                        vars[1].add(s.whileLoop(new SDVariable[]{vars[0], sum2},
                                (sd2, vars2) -> vars2[0].gt(0),
                                (sd2, vars2) -> new SDVariable[]{vars2[0].sub(1), vars2[1].add(vars2[0])})[1])});

        INDArray out = sum[1].eval();
        assertEquals(35, out.getInt(0));

        String outName = sum[1].name();

        sd = SameDiff.fromFlatBuffers(sd.asFlatBuffers(false));

        assertEquals(35, sd.output(Collections.emptyMap(), outName).get(outName).getInt(0));

    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testForLoop() {
        SameDiff sd = SameDiff.create();
        SDVariable start = sd.var("loopiter",Nd4j.scalar(1.0));
        SDVariable end = sd.var("end",Nd4j.scalar(6.0));
        SameDiffSingleLambda sameDiffSingleLambda = (sameDiff, inputs) -> inputs[0].lt(inputs[1]);

        SDVariable[] sdVariables = sd.whileLoop(new SDVariable[]{start, end}, sameDiffSingleLambda, (sameDiff, inputs) -> {
            SDVariable add = inputs[0].add(1.0);
            return new SDVariable[]{
                    add,inputs[1]
            };
        });
        System.out.println(sd.summary());
        Map<String, INDArray> outputs = sd.outputAll(null);
        assertEquals(Nd4j.scalar(6.0),outputs.get(sdVariables[0].name()));
        assertEquals(Nd4j.scalar(6.0),outputs.get(sdVariables[1].name()));
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLooping() {
        SameDiff parent = SameDiff.create();
        SDVariable input = parent.placeHolder("input",DataType.FLOAT);
        SameDiff loopBody = SameDiff.create();
        SDVariable loopInput = loopBody.placeHolder("input", DataType.FLOAT);
        SDVariable output = loopBody.math().add("output",loopInput,1.0);
        SDVariable[] args = ControlFlow.initializeLoopBody(new String[]{"curr_iteration", "max_iterations", "cond_in"}, parent, 5, true);
        SDVariable[] childArgs = ControlFlow.initializeLoopBody(new String[]{"curr_iteration", "max_iterations", "cond_in"}, loopBody, 5, true);

        String[] inputNames = {
                "curr_iteration",
                "max_iterations",
                "cond_in",
                "input"
        };

        String[] outputNames = {
                "curr_iteration",
                "max_iterations",
                "cond_in",
                "output"
        };



        SDVariable[] finalArgs = new SDVariable[args.length + 1];
        for(int i = 0; i < args.length; i++) {
            finalArgs[i] = args[i];
        }
        finalArgs[3] = input;

        ControlFlow.LoopParams loopParams = ControlFlow.LoopParams.builder()
                .parent(parent)
                .functionBody(loopBody)
                .functionBodyInputs(inputNames)
                .functionBodyOutputs(outputNames)
                .loopVars(finalArgs)
                .loopName("loop")
                .functionName("func")
                .build();

        String[] finalOutputNames = new String[outputNames.length];
        for(int i = 0; i < finalOutputNames.length; i++) {
            finalOutputNames[i] = outputNames[i] + "_final";
        }

        SDVariable[] loopWithConditions = parent.loopWithConditions(finalOutputNames,loopParams);

        INDArray assertion = Nd4j.ones(5).addi(5);
        Map<String, INDArray> output2 = parent.output(Collections.singletonMap("input", Nd4j.ones(5)), "output_final");
        assertEquals(assertion,output2.get("output_final").reshape(assertion.shape()).castTo(assertion.dataType()));
        System.out.println(output2);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNestedWhileIf() throws IOException {
        SameDiff sd = SameDiff.create();
        SDVariable countIn = sd.constant(5);
        SDVariable sumIn = sd.constant(0);
        SDVariable hundred = sd.constant(100);

        SDVariable[] sum = sd.whileLoop(new SDVariable[]{countIn, sumIn},
                (s, vars) -> vars[0].gte(0),
                (s, vars) -> new SDVariable[]{vars[0].sub(1), vars[1].add(
                        s.ifCond((sd2) -> vars[0].eq(0),
                                (sd2) -> vars[0].add(100), //TODO replace with hundred and things break
                                (sd2) -> vars[0])
                )});

        INDArray out = sum[1].eval();
        assertEquals(115, out.getInt(0));

        String outName = sum[1].name();

        sd = SameDiff.fromFlatBuffers(sd.asFlatBuffers(false));

        assertEquals(115, sd.output(Collections.emptyMap(), outName).get(outName).getInt(0));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMod_1(Nd4jBackend backend) {
        val sd = SameDiff.create();
        val initial = sd.constant("initial", Nd4j.createFromArray(5.f, 6.f, 7.f));
        val four = sd.constant("four", 4.0f);
        val mod = initial.mod("mod",  four);

        val e = Nd4j.createFromArray(1.f, 2.f, 3.f);

        assertEquals(e, mod.eval());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void castShapeTest1(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.constant(Nd4j.createFromArray(1, 2, 3, 4));
        SDVariable casted = x.castTo(DataType.FLOAT);

        assertEquals(casted.dataType(), DataType.FLOAT);
    }

    @Test
    @Disabled // casted shape is null
    public void castShapeTestEmpty(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.constant(Nd4j.empty(DataType.INT));
        SDVariable casted = x.castTo(DataType.FLOAT);

        assertEquals(casted.dataType(), DataType.FLOAT);
        assertTrue(casted.getShapeDescriptor().isEmpty());
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEmptyShapeVar(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();

        try {
            sd.var(DataType.FLOAT, 1, 0, 2);
            fail("Expected exception");
        } catch (IllegalArgumentException e){
            String m = e.getMessage();
            assertTrue(m.contains("variable") && m.contains("empty") && m.contains("0"),m);
        }

        try {
            sd.var(Nd4j.create(1, 0, 2));
            fail("Expected exception");
        } catch (IllegalArgumentException e){
            String m = e.getMessage().toLowerCase();
            assertTrue(m.contains("variable") && m.contains("empty") && m.contains("0"),m);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPReLU(Nd4jBackend backend) {
        Nd4j.getExecutioner().enableDebugMode(true);
        Nd4j.getExecutioner().enableVerboseMode(true);
        SameDiff sd = SameDiff.create();

        SDVariable input = sd.constant(Nd4j.createFromArray(
                new int[][][]{{
                        {-10, 10, 10, -10},
                        {10, 10, -10, -10}
                }}
        ).castTo(DataType.DOUBLE));

        SDVariable alpha = sd.var(Nd4j.createFromArray(0.01, 0.1).castTo(DataType.DOUBLE));

        SDVariable out = sd.nn.prelu("out", input, alpha, 2);

        TestCase tc = new TestCase(sd).expected("out", Nd4j.createFromArray(new double[][][]{{
                {-0.1, 10, 10, -0.1},
                {10, 10, -1, -1}
        }}).castTo(DataType.DOUBLE)).gradientCheck(true);

        String err = OpValidation.validate(tc);
        assertNull(err);
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


//        Nd4j.getRandom().setSeed(0);
//        System.out.println(rand0.eval());
//
//        Nd4j.getRandom().setSeed(0);
//        System.out.println(rand1.eval());

        INDArray a0 = rand0.eval();
        Nd4j.getRandom().setSeed(0);
        INDArray a1 = rand1.eval();
        assertEquals(a0, a1);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCalculateGradientsAndOutputs(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("in", DataType.FLOAT, -1, 4);
        SDVariable w = sd.var("w", Nd4j.rand(DataType.FLOAT, 4, 3));
        SDVariable b = sd.var("b", Nd4j.rand(DataType.FLOAT, 3));
        SDVariable z = in.mmul(w).add("z", b);
        SDVariable softmax = sd.nn.softmax("softmax", z, 0);

        Map<String,INDArray> ph = Collections.singletonMap("in", Nd4j.rand(DataType.FLOAT, 2, 4));
        List<String> outputs = Arrays.asList("in", "z", "softmax");
        List<String> grads = Arrays.asList("in", "w", "z");

        OutAndGrad oag = sd.calculateGradientsAndOutputs(ph, outputs, grads);
        Map<String,INDArray> outs = oag.getOutputs();
        Map<String,INDArray> g = oag.getGradients();


        Map<String,INDArray> outExp = sd.output(ph, outputs);
        Map<String,INDArray> gExp = sd.calculateGradients(ph, grads);

        assertEquals(outExp, outs);
        assertEquals(gExp, g);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConcatVariableGrad(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable label = sd.var("label", DataType.FLOAT, 3, 4);
        SDVariable a = sd.var("a", DataType.FLOAT, 3, 2);
        SDVariable b = sd.var("b", DataType.FLOAT, 3, 2);
        INDArray inputArr = Nd4j.rand(3,4);
        INDArray labelArr =  Nd4j.rand(3,4);
        SDVariable c = sd.concat("concat", 1, a, b);
        SDVariable loss = sd.math().pow(c.sub(label), 2);
        sd.setLossVariables(loss);
        sd.associateArrayWithVariable(labelArr, label);
        sd.associateArrayWithVariable(inputArr.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 2)), a);
        sd.associateArrayWithVariable(inputArr.get(NDArrayIndex.all(), NDArrayIndex.interval(2, 4)), b);
        Map<String, INDArray> map = sd.calculateGradients(null, "a", "b", "concat");
        INDArray concatArray = Nd4j.hstack(map.get("a"), map.get("b"));
        assertEquals(concatArray, map.get("concat"));

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSliceVariableGrad(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable label = sd.var("label", DataType.FLOAT, 3, 4);
        SDVariable input = sd.var("input", DataType.FLOAT, 3, 4);
        INDArray inputArr =  Nd4j.rand(3,4);
        INDArray labelArr =  Nd4j.rand(3,4);
        SDVariable a = input.get(SDIndex.all(), SDIndex.interval(0, 2));
        SDVariable b = input.get(SDIndex.all(), SDIndex.interval(2, 4));
        SDVariable c = sd.concat("concat", 1, a, b);
        SDVariable loss = sd.math().pow(c.sub(label), 2);
        sd.setLossVariables(loss);
        sd.associateArrayWithVariable(labelArr, label);
        sd.associateArrayWithVariable(inputArr, input);
        Map<String, INDArray> map = sd.calculateGradients(null,"input", "concat");
        assertEquals(map.get("input"), map.get("concat"));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTrainingConfigJson(Nd4jBackend backend) {
        for(IEvaluation e : new IEvaluation[]{new Evaluation(), new RegressionEvaluation(), new EvaluationBinary(), new ROC(),
                new ROCMultiClass(), new ROCBinary(), new EvaluationCalibration()}) {
            TrainingConfig config =  TrainingConfig.builder()
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
        for(DataType dt : new DataType[]{DataType.FLOAT, DataType.DOUBLE,DataType.BFLOAT16}) {
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
        assertThrows(IllegalArgumentException.class,() -> {
            SameDiff sd = SameDiff.create();

            int nOut = 4;
            int minibatch = 10;
            SDVariable predictions = sd.var("in", DataType.DOUBLE, minibatch, nOut);
            SDVariable labels = sd.placeHolder("labels", DataType.DOUBLE, -1, nOut);

            LossReduce reduction = LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT;

            SDVariable   loss = sd.loss().absoluteDifference("loss", labels, predictions, null, reduction);

            try {
                loss.eval();
                fail("Exception should have been thrown");
            } catch (IllegalStateException e) {
                String msg = e.getMessage();
                assertTrue(msg.contains("\"labels\"") && msg.contains("No array was provided"),msg);
            }
        });

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

        SDVariable w1 = sd1.constant("c1",1.0f);
        SDVariable w2 = sd2.constant("c1",1.0f);

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
        int bS=2, iH=4,iW=3,  iC=4,oC=3,  kH=3,kW=2,  sH=1,sW=1,  pH=0,pW=0,  dH=1,dW=1;
        int       oH=2,oW=2;
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





    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testControlflowBackProp() {
        //ifCond();
        System.out.println("=".repeat(100));
        //TODO: figure out why Variable type + enter body has no gradient.
        //could be edge case we need to yet figure out or have something to do with
        //function nesting + control flow. This should be examined closer.
        Nd4j.getExecutioner().enableVerboseMode(true);
        Nd4j.getExecutioner().enableDebugMode(true);
        whileLoop();
    }

    private static void ifCond() {
        int batchSize = 4;
        int modelDim = 8;

        SameDiff sd = SameDiff.create();

        SDVariable features = sd.placeHolder("features", DataType.FLOAT, batchSize, modelDim);
        SDVariable labels = sd.placeHolder("labels", DataType.FLOAT, batchSize, modelDim);
        SDVariable var = sd.var("variable", new OneInitScheme('c'), DataType.FLOAT, batchSize, modelDim);
        SDVariable predictions = sd.ifCond("predictions", null,
                _sd -> features.sum().gt(0),
                _sd -> features.sub(var),
                _sd -> features.add(var));
        sd.loss.meanSquaredError("loss", labels, predictions, null);

        TrainingConfig config = new TrainingConfig.Builder()
                .updater(new Adam(0.1))
                .dataSetFeatureMapping("features")
                .dataSetLabelMapping("labels")
                .build();
        sd.setTrainingConfig(config);

        RecordReader reader = new CollectionRecordReader(
                Collections.nCopies(batchSize, Collections.nCopies(2 * modelDim, new IntWritable(1))));
        DataSetIterator iterator = new RecordReaderDataSetIterator(
                reader, batchSize, modelDim, 2 * modelDim - 1, true);

        System.out.println(sd.output(iterator, "predictions").get("predictions")); // forward pass works

        sd.fit(iterator, 1); // backward pass throws exception
    }

    private static void whileLoop() {
        int batchSize = 4;
        int modelDim = 8;

        Nd4j.getExecutioner().enableDebugMode(true);
        Nd4j.getExecutioner().enableVerboseMode(true);
        SameDiff sd = SameDiff.create();

        SDVariable features = sd.placeHolder("features", DataType.FLOAT, batchSize, modelDim);
        SDVariable labels = sd.placeHolder("labels", DataType.FLOAT, batchSize, modelDim);
        SDVariable var = sd.var("variable", new OneInitScheme('c'), DataType.FLOAT, batchSize, modelDim);
        SDVariable predictions = sd.whileLoop(
                new String[]{"predictions","variable2"}, null,
                new SDVariable[]{features,var},
                (_sd, inputs) -> inputs[0].sum().gt(0),
                (_sd, inputs) -> new SDVariable[]{inputs[0].sub(inputs[1]),inputs[1]})[0];
        SDVariable loss2 = sd.loss.meanSquaredError("loss", labels, predictions, null);

        System.out.println(sd.summary(true));

        TrainingConfig config = new TrainingConfig.Builder()
                .updater(new Adam(0.1))
                .dataSetFeatureMapping("features")
                .dataSetLabelMapping("labels")
                .build();
        sd.setTrainingConfig(config);

        RecordReader reader = new CollectionRecordReader(
                Collections.nCopies(batchSize, Collections.nCopies(2 * modelDim, new IntWritable(1))));
        DataSetIterator iterator = new RecordReaderDataSetIterator(
                reader, batchSize, modelDim, 2 * modelDim - 1, true);

        System.out.println(sd.output(iterator, "predictions").get("predictions")); // forward pass works

        sd.fit(iterator, 1); // backward pass throws exception
    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConv2DDifferentWeightsFormat(Nd4jBackend backend) {
        int bS=2, iH=4,iW=3,  iC=4,oC=3,  kH=3,kW=2,  sH=1,sW=1,  pH=0,pW=0,  dH=1,dW=1;
        int       oH=2,oW=2;
        SameDiff sd = SameDiff.create();

        INDArray inArr = Nd4j.linspace(DataType.FLOAT, 25, -0.5, 96).reshape(new long[]{bS, iC, iH, iW});
        INDArray weights = Nd4j.rand(DataType.FLOAT, oC, iC, kH, kW);

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
                .weightsFormat(WeightsFormat.OIYX)
                .build();

        SDVariable out = sd.cnn().conv2d(sdInput, sdWeights, sdBias, c);

        assertArrayEquals(new long[]{bS, oC, oH, oW}, out.eval().shape());

        weights = weights.permute(0,2,3,1);
        SDVariable permutedWeights = sd.var("weights2", weights);

        // Shape per format tip:
        //[3, 4, 3, 2] - OIYX
        //[3, 3, 2, 4] - OYXI
        //[3, 2, 4, 2] - YXIO
        Conv2DConfig c2 = Conv2DConfig.builder()
                .kH(kH).kW(kW)
                .pH(pH).pW(pW)
                .sH(sH).sW(sW)
                .dH(dH).dW(dW)
                .paddingMode(PaddingMode.VALID)
                .weightsFormat(WeightsFormat.OYXI)
                .build();

        SDVariable out2 = sd.cnn().conv2d(sdInput, permutedWeights, sdBias, c2);
        assertEquals(out.eval(), out2.eval());
    }
}
