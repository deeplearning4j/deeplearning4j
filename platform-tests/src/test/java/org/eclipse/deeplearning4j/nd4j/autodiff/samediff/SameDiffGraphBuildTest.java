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

import static org.junit.jupiter.api.Assertions.*;

import java.util.*;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.jupiter.api.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.*;
import org.nd4j.autodiff.validation.OpValidation;
import org.nd4j.autodiff.validation.TestCase;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.ops.transforms.Transforms;

@Slf4j
@NativeTag
@Tag(TagNames.SAMEDIFF)
public class SameDiffGraphBuildTest extends BaseNd4jTestWithBackends {

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

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSameDiffShapeNonNumerical() {
        SameDiff sd = SameDiff.create();
        SDVariable var = sd.create(null, sd.constant(8), DataType.BOOL);
        assertEquals(8, var.shape().eval().getLong(0)); // throws exception    }
        sd.setShape(var, var.shape())[0].eval();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSameDiffCreate() {
        SameDiff sd = SameDiff.create();
        SDVariable var = sd.create(null, sd.constant(8), DataType.INT32);
        assertEquals(DataType.INT, var.eval().dataType());
        assertEquals(DataType.INT, var.dataType());
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
    public void testNames(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable in1 = sd.var("in", new long[]{3, 2});
        SDVariable in2 = sd.var("in2", new long[]{3, 3});

        val m = in1.add(1.0);
        val f = m.add(2.0);
        val s = in2.add(5.0);

        Map<String, INDArray> map = sd.outputAll(null);
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
    public void testSDVariableLength(Nd4jBackend backend) {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Nd4j.ones(100);
        assertEquals(100, sameDiff.var(arr).length().eval().getInt(0));

        INDArray arr2 = Nd4j.ones(5, 5);
        assertEquals(25, sameDiff.var(arr2).length().eval().getInt(0));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGetVariable(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        INDArray arr = Nd4j.linspace(1, 100, 100).reshape('c', 10L, 10L);
        System.out.println(arr);
        SDVariable x = sd.var(arr);
        assertEquals(Nd4j.linspace(1, 10, 10), x.get(SDIndex.point(sd.constant(0).reshape(1))).eval());
        assertEquals(arr.get(NDArrayIndex.point(0), NDArrayIndex.point(1)), x.get(SDIndex.point(0), SDIndex.point(1)).eval());
        assertEquals(arr.get(NDArrayIndex.interval(0, 2)), x.get(SDIndex.interval(0, 2)).eval());
        assertEquals(arr.get(NDArrayIndex.interval(0, 2)), x.get(SDIndex.interval(sd.constant(0).reshape(1), sd.constant(2).reshape(1))).eval());
        assertEquals(arr.get(NDArrayIndex.interval(0, 2, 2)), x.get(SDIndex.interval(sd.constant(0).reshape(1), sd.constant(2).reshape(1), sd.constant(2).reshape(1))).eval());
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
        assertEquals(arr.get(NDArrayIndex.interval(0, 2)), x.getView(SDIndex.interval(0, 2)).eval());
        assertEquals(arr.get(NDArrayIndex.interval(0, 2)), x.getView(SDIndex.interval(sd.constant(0).reshape(1), sd.constant(2).reshape(1))).eval());
        assertEquals(arr.get(NDArrayIndex.interval(0, 2, 2)), x.getView(SDIndex.interval(sd.constant(0).reshape(1), sd.constant(2).reshape(1), sd.constant(2).reshape(1))).eval());
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
            assertTrue(e.getMessage().contains("No loss variables"), e.getMessage());
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
            assertTrue(e.getMessage().contains("No loss variables"), e.getMessage());
        }

        SDVariable add = in.add(in2);
        sd.createGradFunction();
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

        Map<String, INDArray> g = sd.calculateGradients(null, "a");
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
        Map<String, INDArray> g = sd.calculateGradients(m, "a", "b");

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
                assertTrue(m.contains("already exists"), m);
            }

            try {
                sd.var("a", Nd4j.zeros(1));
                fail("Expected exception");
            } catch (Throwable t) {
                String m = t.getMessage();
                assertNotNull(m);
                assertTrue(m.contains("already exists"), m);
            }

            try {
                sd.var("a", LongShapeDescriptor.fromShape(new long[]{1}, DataType.FLOAT));
                fail("Expected exception");
            } catch (Throwable t) {
                String m = t.getMessage();
                assertNotNull(m);
                assertTrue(m.contains("already exists"), m);
            }

            try {
                sd.constant("a", Nd4j.zeros(1));
                fail("Expected exception");
            } catch (Throwable t) {
                String m = t.getMessage();
                assertNotNull(m);
                assertTrue(m.contains("already exists"), m);
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
    public void testEmptyShapeVar(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();

        try {
            sd.var(DataType.FLOAT, 1, 0, 2);
            fail("Expected exception");
        } catch (IllegalArgumentException e) {
            String m = e.getMessage();
            assertTrue(m.contains("variable") && m.contains("empty") && m.contains("0"), m);
        }

        try {
            sd.var(Nd4j.create(1, 0, 2));
            fail("Expected exception");
        } catch (IllegalArgumentException e) {
            String m = e.getMessage().toLowerCase();
            assertTrue(m.contains("variable") && m.contains("empty") && m.contains("0"), m);
        }
    }
}
