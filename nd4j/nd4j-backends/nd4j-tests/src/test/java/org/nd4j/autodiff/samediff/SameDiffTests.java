/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.autodiff.samediff;

import static org.junit.Assert.*;
import static org.junit.Assume.assumeNotNull;
import static org.nd4j.linalg.indexing.NDArrayIndex.all;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.After;
import org.junit.Before;
import org.junit.ClassRule;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.OpValidationSuite;
import org.nd4j.autodiff.samediff.api.OutAndGrad;
import org.nd4j.autodiff.samediff.impl.DefaultSameDiffConditional;
import org.nd4j.autodiff.validation.OpValidation;
import org.nd4j.autodiff.validation.TestCase;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.ExternalErrorsFunction;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.LocalResponseNormalizationConfig;
import org.nd4j.linalg.api.ops.impl.reduce3.ManhattanDistance;
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
import org.nd4j.linalg.checkutil.NDArrayCreationUtil;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.adapter.SingletonMultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.weightinit.impl.OneInitScheme;
import org.nd4j.weightinit.impl.UniformInitScheme;
import org.nd4j.weightinit.impl.ZeroInitScheme;

/**
 * Created by agibsonccc on 4/11/17.
 */
@Slf4j
public class SameDiffTests extends BaseNd4jTest {

    private DataType initialType;

    public SameDiffTests(Nd4jBackend b) {
        super(b);
    }

    @Override
    public char ordering() {
        return 'c';
    }

    @ClassRule
    public static TemporaryFolder folder = new TemporaryFolder();


    @Before
    public void before() {
        Nd4j.create(1);
        initialType = Nd4j.dataType();

        Nd4j.setDataType(DataType.DOUBLE);
        Nd4j.getRandom().setSeed(123);
    }

    @After
    public void after() {
        Nd4j.setDataType(initialType);

        NativeOpsHolder.getInstance().getDeviceNativeOps().enableDebugMode(false);
        NativeOpsHolder.getInstance().getDeviceNativeOps().enableVerboseMode(false);
    }

    public Map<String, INDArray> variablesForInput() {
        INDArray inputs = Nd4j.create(new double[][]{
                {0.52, 1.12, 0.77},
                {0.88, -1.08, 0.15},
                {0.52, 0.06, -1.30},
                {0.74, -2.49, 1.39}
        });

        INDArray labels = Nd4j.create(new double[]{1, 1, 0, 1}).reshape(4, 1);

        INDArray weights = Nd4j.zeros(3, 1);

        Map<String, INDArray> inputMap = new HashMap<>();
        inputMap.put("x", inputs);
        inputMap.put("w", weights);
        inputMap.put("y", labels);
        return inputMap;
    }

    @Test
    public void testVariableNaming_1() {
        val sd = SameDiff.create();

        val input = sd.var("inp", new long[]{2, 3});

        val nodeA = sd.math().square(input);
        val nodeB = sd.math().square(nodeA);

        sd.associateArrayWithVariable(Nd4j.create(new double[]{1, 2, 3, 4, 5, 6}, new long[]{2, 3}), input);

        sd.outputAll(null);

        nodeA.isPlaceHolder();
    }


    @Test
    public void testAddArgsAndOutput() {
        SameDiff sameDiff = SameDiff.create();
        val varOne = sameDiff.var("one", Nd4j.ones(2));
    }

    @Test
    public void testMseBackwards() {

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

    @Test
    public void testEvalVariable() {
        SameDiff sameDiff = SameDiff.create();
        INDArray ones = Nd4j.ones(4);
        INDArray twos = ones.add(ones);
        SDVariable inputOne = sameDiff.var("inputone", ones);
        SDVariable inputResult = inputOne.add("extravarname", inputOne);
        assertEquals(twos, inputResult.eval());
    }


    @Test
    public void testSum() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1, 4, 4, DataType.FLOAT)).reshape(1, 4);
        SDVariable x = sameDiff.var("x", arr);
        SDVariable result = sameDiff.sum(x, 1); //[1,4].sum(1) == [1]

        INDArray exp = Nd4j.scalar(arr.sumNumber().floatValue()).reshape(1);
        INDArray resultArr = result.eval();
        assertEquals(exp, resultArr);
    }

    @Test
    public void testAddEval() {
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

    @Test
    public void testWeightedXentWithLogits() {
        SameDiff sameDiff = SameDiff.create();
        INDArray targets = Nd4j.create(new long[]{1, 5});
        INDArray inputs = Nd4j.create(new long[]{1, 5});
        INDArray weights = Nd4j.create(new long[]{1, 5});

        SDVariable sdInputs = sameDiff.var("inputs", inputs);
        SDVariable sdWeights = sameDiff.var("weights", weights);
        SDVariable sdTargets = sameDiff.var("targets", targets);

        SDVariable res = sameDiff.loss().weightedCrossEntropyWithLogits(sdTargets, sdInputs, sdWeights);

        INDArray resultArray = res.eval();
        assertArrayEquals(new long[]{1, 5}, resultArray.shape());
    }

    @Test
    public void testMseForward() {

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

    @Test
    public void testDistance() {
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

    @Test
    public void testTensorGradMmul() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1, 4, 4)).reshape(2, 2);
        SDVariable x = sameDiff.var("x", arr);
        SDVariable y = sameDiff.var("y", arr);
        SDVariable result = sameDiff.mmul(x, y);
        SDVariable otherResult = result.add(result);
        Map<String,INDArray> m = sameDiff.outputAll(null);
        assertArrayEquals(new long[]{2, 2}, m.get(result.name()).shape());
    }


    @Test
    public void testEval() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Nd4j.linspace(1, 4, 4);
        SDVariable x = sameDiff.var("x", arr);
        SDVariable sigmoid = sameDiff.nn().sigmoid("s", x);
        INDArray assertion = Transforms.sigmoid(arr);
        INDArray eval = sameDiff.output(Collections.singletonMap("x", arr), Collections.singletonList("s")).get("s");
        assertEquals(assertion, eval);
    }

    @Test
    public void testFunctionInputsAndArgs() {
        SameDiff sameDiff = SameDiff.create();
        SDVariable var = sameDiff.var("one", Nd4j.scalar(1.0));
        SDVariable variable2 = sameDiff.var("two", Nd4j.scalar(1.0));
        val sum = var.add(variable2);
        INDArray out = sum.eval();
        assertArrayEquals(new long[0], out.shape());
    }


    @Test
    public void testCrossSameDiffVariableInitWithAlloc() {
        SameDiff first = SameDiff.create();
        SameDiff second = SameDiff.create();

        SDVariable firstVar = first.var("one", new long[]{2, 2});
        SDVariable secondVar = second.var(firstVar);
        assertEquals(firstVar.getArr(), secondVar.getArr());
        assertEquals(firstVar.name(), secondVar.name());
    }


    @Test
    public void testCrossSameDiffVariableInitWithPlaceHolder() {
        SameDiff first = SameDiff.create();
        SameDiff second = SameDiff.create();

        SDVariable firstVar = first.var("one", new long[]{2, 2});
        SDVariable secondVar = second.var(firstVar);
        assumeNotNull(firstVar.getArr());

        assertEquals(firstVar.getArr(), secondVar.getArr());
        assertEquals(firstVar.name(), secondVar.name());
    }


    @Test
    public void testVariableArrayReference() {
        SameDiff sameDiff = SameDiff.create();
        SDVariable arr = sameDiff.var("one", new long[]{2, 2});
        assertArrayEquals(new long[]{2, 2}, arr.getShape());
        assumeNotNull(arr.getArr());
        assertArrayEquals(new long[]{2, 2}, arr.getArr().shape());
    }

    @Test
    public void testEvalAddSelf() {
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

    @Test
    public void testEvalAdd() {
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

    @Test
    public void testDup() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1, 8, 8)).reshape(2, 2, 2);
        SDVariable x = sameDiff.var("x", arr);
        SDVariable y = sameDiff.var("y", arr);
        SameDiff tg2 = sameDiff.dup();
    }


    @Test
    public void testElementWiseDivAndRDiv() {
        SameDiff sameDiff = SameDiff.create();
        INDArray ones = Nd4j.ones(4);
        INDArray toDivBy = Nd4j.valueArrayOf(4, 0.25);
        Map<String, INDArray> xAndY = new HashMap<>();
        xAndY.put("x", ones);
        xAndY.put("y", toDivBy);
        sameDiff.defineFunction("div", new SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable x = sameDiff.var("x", inputs.get("x"));
                SDVariable y = sameDiff.var("y", inputs.get("y"));
                return new SDVariable[]{x.div("out", y)};
            }
        }, xAndY);

        sameDiff.defineFunction("rdiv", new SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable x = sameDiff.var("x", inputs.get("x"));
                SDVariable y = sameDiff.var("y", inputs.get("y"));
                return new SDVariable[]{x.rdiv("out", y)};
            }
        }, xAndY);

        INDArray assertionForDiv = Nd4j.valueArrayOf(4, 4.0);
        INDArray assertionForRDiv = Nd4j.valueArrayOf(4, 0.25);
        assertEquals(assertionForDiv, sameDiff.getFunction("div").outputSingle(null, "out"));
        assertEquals(assertionForRDiv, sameDiff.getFunction("rdiv").outputSingle(null, "out"));

    }


    @Test
    public void testNegativeGradient() {
        SameDiff sameDiff = SameDiff.create();
        INDArray ones = Nd4j.ones(4);
        Map<String, INDArray> xAndY = new HashMap<>();
        xAndY.put("x", ones);
        sameDiff.defineFunction("neg", new SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable x = sameDiff.var("x", inputs.get("x"));
                return new SDVariable[]{sameDiff.math().neg("out", x)};
            }
        }, xAndY);

        INDArray assertionForDiv = Nd4j.valueArrayOf(4, -1);
        assertEquals(assertionForDiv, sameDiff.getFunction("neg").outputSingle(null, "out"));

    }


    @Test
    public void testSumOp() {
        SameDiff sameDiff = SameDiff.create();
        INDArray sumInput = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put("x", sumInput);
        sameDiff.defineFunction("sum", new SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable input = sameDiff.var("x", inputs.get("x"));
                SDVariable sum = sameDiff.sum("sum", input, 1);
                return new SDVariable[]{sum};
            }
        }, inputs);

        INDArray assertion = sumInput.sum(1);
        INDArray out = sameDiff.getFunction("sum").output(Collections.emptyMap(), Collections.singletonList("sum"))
                .get("sum");
        assertEquals(assertion, out);
    }


    @Test
    public void testVariableReferenceNoFunction() {
        /**
         * Creating a variable should not create a differential function.
         */
        SameDiff sameDiff = SameDiff.create();
        SDVariable sdVariable = sameDiff.var("one", Nd4j.scalar(1.0));
        assumeNotNull(sameDiff.getVariable(sdVariable.name()));
    }


    @Test
    public void testVariableWithFunction() {
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


    @Test
    public void testUpdateVariable() {
        SameDiff sameDiff = SameDiff.create();
        SDVariable one = sameDiff.one("one", new long[]{1, 1});
        one.rename("one-diff");
        assertEquals(one.eval(), sameDiff.getVariable("one-diff").eval());
    }


    @Test
    public void testDefineFunctionArrayExistence() {
        SameDiff sameDiff = SameDiff.create();
        String testFunctionName = "testfunction";
        SDVariable[] inputVars = new SDVariable[]{
                sameDiff.var("one", new long[]{1, 1}),
                sameDiff.var("two", new long[]{1, 1}),

        };

        SameDiff functionDef = sameDiff.defineFunction(testFunctionName, new SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                return new SDVariable[]{variableInputs[0].add(variableInputs[1])};
            }
        }, inputVars);

        //1 input plus 2 outputs
        assertEquals(3, functionDef.variables().size());


    }

    @Test
    public void testAutoBroadcastAddMatrixVector() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray row = Nd4j.ones(2);
        INDArray assertion = arr.add(1.0);
        SDVariable left = sameDiff.var("arr", arr);
        SDVariable right = sameDiff.var("row", row);
        SDVariable test = left.add(right);
        assertEquals(assertion, test.eval());
    }


    @Test
    public void testNegativeOneShape() {
        val sd = SameDiff.create();
        SDVariable var = sd.placeHolder("test", DataType.FLOAT, -1, 3);
        assertTrue(var.isPlaceHolder());
    }

    @Test
    public void testShapeResolutionMinus1() {
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

    @Test
    public void testLabelInputPlaceHolderSgd() {

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


    @Test
    public void testSequentialMeansPlaceholder() {
        OpValidationSuite.ignoreFailing();
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
            assertArrayEquals(msg, new long[]{10}, shape);
        }
    }


    @Test
    public void testReductionShapes1() {

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


    @Test
    public void testReductionShapes2() {

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

    @Test
    public void testNames() {
        SameDiff sd = SameDiff.create();
        SDVariable in1 = sd.var("in", new long[]{3, 2});
        SDVariable in2 = sd.var("in2", new long[]{3, 3});

        val m = in1.add(1.0);
        val f = m.add(2.0);
        val s = in2.add(5.0);

        Map<String,INDArray> map = sd.outputAll(null);
        log.info("Result M: {}", map.get(m.name()));
        log.info("Result F: {}", map.get(f.name()));
        log.info("Result S: {}", map.get(s.name()));
    }

    @Test
    public void testRunLogisticRegression() {
        Map<String, INDArray> vars = this.variablesForInput();
        SameDiff outside = SameDiff.create();
        outside.defineFunction("activate", new SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
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
            }
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


    @Test
    public void testTransposeWithVector() {
        val sd = SameDiff.create();
        val matrix = Nd4j.linspace(1, 12, 12).reshape(4, 3);
        val vector = Nd4j.linspace(1, 4, 4).reshape(4, 1);
        val input1 = sd.var("input", matrix);
        val input2 = sd.var("input2", vector);
        val output = sd
                .mmul("output", input1, input2, MMulTranspose.builder().transposeA(true).transposeB(false).build());
        INDArray out = output.eval();
        assertArrayEquals(new long[]{3, 1}, out.shape());
    }

    @Test
    public void testSimpleDefineFunction() {
        SameDiff sameDiffOuter = SameDiff.create();
        Map<String, INDArray> inputs = variablesForInput();
        inputs.remove("y");
        String logisticForward = "logisticPredictions";
        sameDiffOuter.defineFunction(logisticForward, new SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {

                SDVariable input = sameDiff.var("x", inputs.get("x"));
                SDVariable w = sameDiff.var("w", inputs.get("w"));
                SDVariable preOutput = sameDiff.mmul(input, w);
                SDVariable sigmoid = sameDiff.nn().sigmoid(preOutput);
                return new SDVariable[]{sigmoid};
            }

        }, inputs);

        assertEquals(1, sameDiffOuter.definedFunctionNames().size());

        //note here that we don't add the duplicate ops with define function anymore
    }

    @Test
    public void testSumGradient() {
        SameDiff sameDiff = SameDiff.create();
        SDVariable twoByTwo = sameDiff.var("initial", Nd4j.linspace(1, 4, 4, DataType.FLOAT).reshape(2, 2));
        SDVariable sum = sameDiff.sum(twoByTwo, Integer.MAX_VALUE);
        Map<String,INDArray> grads = sameDiff.calculateGradients(Collections.emptyMap(), sameDiff.getVariables().keySet());
        assertEquals(Nd4j.ones(DataType.FLOAT, 2, 2), grads.get(twoByTwo.name()));
    }


    @Test
    public void testRsubScalar() {
        SameDiff sameDiff = SameDiff.create();
        Map<String, INDArray> params = new HashMap<>();
        INDArray var = Nd4j.valueArrayOf(4, 2);
        params.put("x", var);
        sameDiff.defineFunction("rsubop", new SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable input = sameDiff.var("x", inputs.get("x"));
                SDVariable ret = input.rsub("rsub", 1.0);
                return new SDVariable[]{ret};
            }
        }, params);

        SameDiff logisticGraph = sameDiff.getFunction("rsubop");
        INDArray output = logisticGraph.output(params, Collections.singletonList("rsub")).get("rsub");
        assertEquals(Nd4j.ones(4).muli(-1), output);
    }


    @Test
    public void testFunctionScalarResultPropagation() {
        SameDiff sameDiffOuter = SameDiff.create();
        Map<String, INDArray> inputs = variablesForInput();

        sameDiffOuter.defineFunction("logisticPredictions", new SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable input = sameDiff.var("x", inputs.get("x"));
                SDVariable w = sameDiff.var("w", inputs.get("w"));
                SDVariable preOutput = sameDiff.mmul(input, w);
                SDVariable sigmoid = sameDiff.nn().sigmoid(preOutput);
                return new SDVariable[]{sigmoid};
            }
        }, inputs);

        sameDiffOuter.defineFunction("oneminuspredictions", new SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable y = sameDiff.var("y", inputs.get("y"));
                SDVariable oneMinusPredictions = y.rsub("rsub", 1.0);
                return new SDVariable[]{oneMinusPredictions};
            }
        }, inputs);

        SameDiff logisticGraph = sameDiffOuter.getFunction("oneminuspredictions");
        Map<String, INDArray> inputsSubset = new HashMap<>();
        inputsSubset.put("y", inputs.get("y"));
        INDArray output = logisticGraph.output(inputsSubset, Collections.singletonList("rsub")).get("rsub");
        INDArray assertion = Nd4j.create(new double[]{0, 0, 1, 0}, new int[]{4, 1});
        assertEquals(assertion, output);

    }


    @Test
    public void testMmul() {
        SameDiff sameDiffOuter = SameDiff.create();
        Map<String, INDArray> inputs = variablesForInput();
        SDVariable x = sameDiffOuter.var("x", inputs.get("x"));
        SDVariable w = sameDiffOuter.var("w", inputs.get("w"));
        SDVariable output = sameDiffOuter.mmul(x, w);
    }


    @Test
    public void testGraphBuilding() {
        final SameDiff sameDiffOuter = SameDiff.create();
        Map<String, INDArray> inputs = variablesForInput();

        sameDiffOuter.defineFunction("logisticPredictions", new SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable input = sameDiff.var("x", inputs.get("x"));
                SDVariable w = sameDiff.var("w", inputs.get("w"));
                SDVariable y = sameDiff.var("y", inputs.get("y"));
                SDVariable preOutput = sameDiff.mmul(input, w);
                SDVariable sigmoid = sameDiff.nn().sigmoid(preOutput);

                return new SDVariable[]{sigmoid};
            }
        }, inputs);

        sameDiffOuter.defineFunction("loss", new SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable outputs = sameDiffOuter.invokeFunctionOn("logisticPredictions", sameDiff);
                SDVariable y = sameDiff.getVariable("y");
                SDVariable outputTimesY = outputs.mul(y);
                return new SDVariable[]{outputTimesY};

            }
        }, inputs);

        SameDiff logisticPrediction = sameDiffOuter.getFunction("logisticPredictions");
        List<String> logisticOpNameAssertions = Arrays.asList("mmul", "sigmoid");


    }


    @Test
    public void testScalarAdd() {
        SameDiff sameDiff = SameDiff.create();
        SDVariable twoByTwo = sameDiff.var("first", Nd4j.linspace(1, 4, 4).reshape('c', 2, 2));
        SDVariable add = twoByTwo.add(1.0);
        INDArray test = add.eval();
        INDArray assertion = Nd4j.linspace(1, 4, 4).reshape('c', 2, 2).add(1.0);
        assertEquals(assertion, test);
    }


    @Test
    public void testSums() {
        SameDiff sameDiff = SameDiff.create();
        INDArray ones = Nd4j.ones(7, 4);
        SDVariable sdVariable = sameDiff.var("ones", ones);
        SDVariable result = sdVariable.add(1.0);
        SDVariable total = sameDiff.sum(result, Integer.MAX_VALUE);
        INDArray out = total.eval();
        assertEquals(56, out.getDouble(0), 1e-1);
    }


    @Test
    public void testDenseLayerForwardPass() {
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

    @Test
    public void testActivationBackprop() {

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

            Map<String,INDArray> m = sd.output(Collections.emptyMap(), "out");
            INDArray outAct = m.get("out");
            assertEquals(a.toString(), outExp, outAct);

            // L = sum_i (label - out)^2
            //dL/dOut = 2(out - label)
            INDArray dLdOutExp = outExp.sub(labelArr).mul(2);
            INDArray dLdInExp = a.getActivationFunction().backprop(inArr.dup(), dLdOutExp.dup()).getFirst();

            Map<String,INDArray> grads = sd.calculateGradients(null, "out", "in");
//            sd.execBackwards(Collections.emptyMap());
//            SameDiff gradFn = sd.getFunction("grad");
            INDArray dLdOutAct = grads.get("out");
            INDArray dLdInAct = grads.get("in");

            assertEquals(a.toString(), dLdOutExp, dLdOutAct);
            assertEquals(a.toString(), dLdInExp, dLdInAct);
        }
    }


    @Test
    public void testPlaceholderReduceSimple() {
        SameDiff sd = SameDiff.create();
        SDVariable v = sd.var("in", new long[]{-1, 10});
        SDVariable vSum = sd.sum(v, 1);                             //Exception here
    }


    @Test
    public void testSequentialMeans() {
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", new long[]{10, 10, 10});
        SDVariable mean1 = sd.mean(in, 2);      //[10,10] out
        SDVariable mean2 = sd.mean(mean1, 1);   //[10,1] out - ***exception here***
    }

    @Test
    public void testBatchNormTest() {
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
        out = sd.nn().tanh("out", out);

        INDArray outArr = out.eval();
        assertArrayEquals(new long[]{1, 10}, outArr.shape());

    }

    @Test
    public void testLrn() {
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

    @Test
    public void testMoments() {
        SameDiff sd = SameDiff.create();

        INDArray input = Nd4j.create(new float[]{1, 2, 3, 4}, new long[]{2, 2});

        SDVariable sdInput = sd.var("input", input);

        val axis = new int[]{0, 1};
        SDVariable[] moments = sd.math().moments(sdInput, axis);
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

    @Test
    public void testNormalizeMoments() {
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


    @Test
    public void testDepthWiseConv2dBasic() {
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
                .isSameMode(false)
                .build();

        SDVariable out = sd.cnn().depthWiseConv2d(in, dW, b, c);
        out = sd.math().tanh("out", out);

        INDArray outArr = out.eval();
        //Expected output size: out = (in - k + 2*p)/s + 1 = (28-2+0)/1+1 = 27
        val outShape = outArr.shape();
        assertArrayEquals(new long[]{mb, depthWise * nIn, 27, 27}, outShape);
    }

    @Test
    public void validateMeanDiff() {
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

    @Test
    public void validateSumDiff() {
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

    @Test
    public void validateStdevDiff() {
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

    @Test
    public void validateVarDiff() {
        for (boolean biasCorrected : new boolean[]{true, false}) {
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

            assertEquals(exp, dLdIn);
        }
    }

    @Test
    public void validateMinDiff() {
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

    @Test
    public void validateMaxDiff() {
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

    @Test
    public void validateProdDiff() {
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

    @Test
    public void testSquare() {
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


    @Test
    public void testExpandDims() {
        for (int i = 0; i <= 2; i++) {
            SameDiff sd = SameDiff.create();
            SDVariable in = sd.var("in", Nd4j.create(2, 3));
            SDVariable expanded = sd.f().expandDims(in, i);

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

    @Test
    public void testZerosLike() {
        SameDiff sd = SameDiff.create();
        SDVariable var0 = sd.var("in", DataType.DOUBLE, new long[]{3, 4});
        SDVariable out = sd.zerosLike("out", var0);

        INDArray out1 = out.eval();
        assertEquals(Nd4j.zeros(3, 4), out1);

        sd.associateArrayWithVariable(Nd4j.create(3, 4), var0);
        INDArray out2 = out.eval();
        assertEquals(Nd4j.zeros(DataType.DOUBLE, 3, 4), out2);
    }

    @Test
    public void testOnesLike() {
        SameDiff sd = SameDiff.create();
        SDVariable var0 = sd.var("in", new long[]{3, 4});
        SDVariable out = sd.onesLike("out", var0);

        INDArray out1 = out.eval();
        assertEquals(Nd4j.ones(3, 4), out1);

        sd.associateArrayWithVariable(Nd4j.create(3, 4), var0);
        INDArray out2 = out.eval();
        assertEquals(Nd4j.ones(3, 4), out2);
    }


    @Test
    public void testOnesLikeBackprop() {
        SameDiff sd = SameDiff.create();
        SDVariable var0 = sd.var("in", new long[]{3, 4});
        SDVariable ones = sd.onesLike("ones", var0);
        SDVariable out = sd.sum("oun", ones);

        INDArray outArr = out.eval();
        assertEquals(Nd4j.scalar(12.0), outArr);

        Map<String,INDArray> m = sd.calculateGradients(Collections.emptyMap(), sd.getVariables().keySet());

        assertEquals(Nd4j.create(3, 4), m.get("in"));
    }


    @Test
    public void testManhattanAlongDim0() {
        Nd4j.getRandom().setSeed(12345);

        INDArray a = Nd4j.rand(new long[]{3, 4, 5});
        INDArray b = Nd4j.rand(new long[]{3, 4, 5});

        INDArray expOut = Nd4j.exec(new ManhattanDistance(a, b, 0));

        val expShape = new long[]{4, 5};

        assertArrayEquals(expShape, expOut.shape());
    }


    @Test
    public void testJaccardDistance() {
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

    @Test
    public void testPairwiseBooleanTransforms() {
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

    @Test
    public void testBooleanChecks() {
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
                    Nd4j.exec(new IsNonDecreasing(new INDArray[]{ia}, new INDArray[]{expOut}));
                    break;
                case 1:
                    t = sd.math().isStrictlyIncreasing(in1);
                    Nd4j.exec(new IsStrictlyIncreasing(new INDArray[]{ia}, new INDArray[]{expOut}));
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

    @Ignore(/*AS - 20191114 https://github.com/eclipse/deeplearning4j/issues/8393*/)
    @Test
    public void testIsStrictlyIncShape() {
        int nOut = 0;
        int minibatch = 0;

        INDArray ia = Nd4j.randn(minibatch, nOut);
        INDArray expOut = Nd4j.create(DataType.BOOL, ia.shape());

        Nd4j.exec(new IsStrictlyIncreasing(new INDArray[]{ia}, new INDArray[]{expOut}));
        System.out.println(expOut);
    }

    @Test
    public void testExpandDims2d() {
        val origShape = new long[]{3, 4};

        for (int i = 0; i < 3; i++) {
            for (Pair<INDArray, String> p : NDArrayCreationUtil
                    .getAllTestMatricesWithShape(origShape[0], origShape[1], 12345, DataType.FLOAT)) {
                INDArray inArr = p.getFirst().muli(100);

                SameDiff sd = SameDiff.create();
                SDVariable in = sd.var("in", inArr);
                SDVariable expand = sd.f().expandDims(in, i);

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

                assertEquals(msg, out, expOut);
            }
        }
    }

    @Test
    public void testSqueezeDims() {
        val origShape = new long[]{3, 4, 5};

        for (int i = 0; i < 3; i++) {

            val shape = origShape.clone();
            shape[i] = 1;

            for (Pair<INDArray, String> p : NDArrayCreationUtil
                    .getAll3dTestArraysWithShape(12345, shape, DataType.FLOAT)) {
                INDArray inArr = p.getFirst().muli(100);

                SameDiff sd = SameDiff.create();
                SDVariable in = sd.var("in", inArr);
                SDVariable squeeze = sd.f().squeeze(in, i);

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

                assertEquals(msg, out, expOut);
            }
        }
    }

    @Test
    public void testExpandSqueezeChain() {

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

                assertEquals(msg, out, inArr);  //expand -> squeeze: should be opposite ops
            }
        }
    }

    @Test
    public void testSqueezeExpandChain() {

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

                assertEquals(msg, out, inArr);  //squeeze -> expand: should be opposite ops
            }
        }
    }

    @Test
    public void testConfusionMatrix() {
        INDArray labels = Nd4j.createFromArray(1, 2, 4);
        INDArray pred = Nd4j.createFromArray(2, 2, 4);
        INDArray weights = Nd4j.createFromArray(10, 100, 1000);
        Integer numClasses = 5;
        SameDiff sd = SameDiff.create();
        SDVariable labelsVar = sd.constant("labels", labels);
        SDVariable predictionsVar = sd.constant("predictions", pred);
        SDVariable weightsVar = sd.constant("weights", weights);
        SDVariable cm = sd.math().confusionMatrix("cm", labelsVar, predictionsVar, numClasses, weightsVar);
        INDArray out = cm.eval();

        INDArray exp = Nd4j.create(new float[][]{{0, 0, 0, 0, 0}, {0, 0, 10, 0, 0}, {0, 0, 100, 0, 0},
                {0, 0, 0, 0, 0}, {0, 0, 0, 0, 1000}}).castTo(DataType.INT);

        assertEquals(exp, out);
    }

    @Test
    public void testArgMax() {
        Nd4j.getRandom().setSeed(12345);

        for (val dim : new int[][]{{0}, {1}, {Integer.MAX_VALUE}, {0, 1}, {}}) {
            INDArray inArr = Nd4j.rand(3, 4);
            SameDiff sd = SameDiff.create();

            SDVariable in = sd.var("in", inArr);
            SDVariable argmax = sd.argmax("argmax", in, dim);

            INDArray out = argmax.eval();

            INDArray exp = Nd4j.argMax(inArr, dim);

            assertEquals(exp, out);
        }
    }

    @Test
    public void testArgMin() {

        Nd4j.getRandom().setSeed(12345);

        for (val dim : new int[][]{{0}, {1}, {Integer.MAX_VALUE}, {0, 1}, {}}) {
            INDArray inArr = Nd4j.rand(3, 4);
            SameDiff sd = SameDiff.create();

            SDVariable in = sd.var("in", inArr);
            SDVariable argmin = sd.argmin("argmin", in, dim);

            INDArray out = argmin.eval();

            INDArray exp = Nd4j.argMax(inArr.neg(), dim);   //argmin(x) == argmax(-x)

            assertEquals(exp, out);
        }
    }

    @Test
    public void testScatterAdd() {
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

    @Test
    public void testScatterMul() {
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

    @Test
    public void testScatterSub() {
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

    @Test
    public void testScatterDiv() {
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

    @Test
    public void testScatterMax() {
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

    @Test
    public void testScatterMin() {
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

    @Test
    public void testReciprocal() {
        INDArray inArr = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray expected = Nd4j.onesLike(inArr).divi(inArr);
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", inArr);
        SDVariable reciprocal = sd.math().reciprocal(in);
        INDArray res = reciprocal.eval();
        assertEquals(expected, res);
    }

    @Test
    public void testGather2() {

        INDArray in = Nd4j.rand(DataType.FLOAT, 10, 10);
        INDArray indices = Nd4j.createFromArray(0, 1, 5);

        SameDiff sd = SameDiff.create();

        SDVariable var = sd.var("in", in);
        SDVariable varIndices = sd.constant("indices", indices);
        SDVariable gather = sd.gather(var, varIndices, 0);

        System.out.println(in);

        INDArray exp = Nd4j.pullRows(in, 1, new int[]{0, 1, 5});  //Along dimension 1 -> equiv to "indexes for axis 0"
        INDArray act = gather.eval();

        assertEquals(exp, act);
    }

    @Test
    public void testGatherOp() {

        INDArray in = Nd4j.rand(DataType.DOUBLE, 10, 10);
        INDArray indices = Nd4j.createFromArray(0, 1, 5);
        INDArray out = Nd4j.create(3, 10);

        DynamicCustomOp op = DynamicCustomOp.builder("gather")
                .addIntegerArguments(0) //Indexes are for dimension 0
                .addInputs(in, indices)
                .addOutputs(out)
                .build();

        Nd4j.exec(op);

        System.out.println(out);

        INDArray exp = Nd4j.pullRows(in, 1, new int[]{0, 1, 5});  //Along dimension 1 == indexes for dimension 0

        assertEquals(exp, out);

        //Shape function:
        val shapes = Nd4j.getExecutioner().calculateOutputShape(op);
        long[] expShape = new long[]{3, 10};

        assertEquals(1, shapes.size());

        assertArrayEquals(expShape, shapes.get(0).getShape());
    }


    @Test
    public void testConditions() {

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

    @Test
    public void testGet() {

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

    @Test
    public void testGetRank3() {

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

    @Test
    public void testTensorArray1() {
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

    @Test
    public void testTensorArray2() {
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

    @Test
    public void testTensorArray3() {
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

    @Test
    public void testFill() {
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

    @Test
    public void testPermute() {
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


    @Test
    public void testExecutionDifferentShapesAccumAlongDim() {
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

    @Test
    public void testExecutionDifferentShapesIndexAccumAlongDim() {
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

    @Test
    public void testExternalErrorsSimple() {
        INDArray externalGrad = Nd4j.linspace(1, 12, 12).reshape(3, 4);

        SameDiff sd = SameDiff.create();
        SDVariable var = sd.var("var", externalGrad);
        SDVariable out = var.mul("out", 0.5);

        Map<String, INDArray> gradMap = new HashMap<>();
        gradMap.put("out", externalGrad);
        ExternalErrorsFunction fn = sd.f().externalErrors(out);

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

    @Test
    public void testUpdatingGradient() {
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

    @Test
    public void testUpdatingGradientSimple() {
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

    @Test
    public void testShapeUpdating() {

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", DataType.FLOAT, 3, 5);
        SDVariable w = sd.var("W", DataType.FLOAT, 5, 4);
        SDVariable b = sd.var("b", DataType.FLOAT, 1, 4);
        SDVariable z = in.mmul(w).add(b);
        SDVariable out = sd.math().tanh("tanh", z);
        ExternalErrorsFunction fn = sd.f().externalErrors(out);

        INDArray inA = Nd4j.linspace(1, 15, 15, DataType.FLOAT).reshape(3, 5);
        INDArray wA = Nd4j.linspace(1, 20, 20, DataType.FLOAT).reshape(5, 4);
        INDArray bA = Nd4j.linspace(1, 4, 4, DataType.FLOAT);
        in.setArray(inA);
        w.setArray(wA);
        b.setArray(bA);

        INDArray grad = Nd4j.linspace(1, 12, 12, DataType.FLOAT).reshape(3, 4);
        Map<String, INDArray> phMap = new HashMap<>();
        phMap.put(fn.getGradPlaceholderName(), grad);

        log.info("--------------- out.eval() ---------------");
        out.eval();
        log.info("--------------- sd.execBackwards() #1 ---------------");
        sd.calculateGradients(phMap, "in", "W", "b");

        log.info("--------------- sd.execBackwards() #2 ---------------");
        System.out.println(sd.getFunction("grad").summary());

        in.setArray(Nd4j.linspace(1, 10, 10).reshape(2, 5));
        grad = Nd4j.linspace(1, 8, 8).reshape(2, 4);
        phMap.put(fn.getGradPlaceholderName(), grad);

        Map<String,INDArray> grads = sd.calculateGradients(phMap, sd.getVariables().keySet());
        INDArray inGrad = grads.get(in.name());
        assertArrayEquals(new long[]{2, 5}, inGrad.shape());
    }

    @Test
    public void testMultiOutput1() {

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", Nd4j.create(3, 4));
        SDVariable mean = in.mean();
        SDVariable sum = in.sum();

        try {
            sd.createGradFunction();
            fail("Expected exception");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("No loss variables"));
        }

        SDVariable add = mean.add(sum);
        sd.createGradFunction();
    }

    @Test
    public void testMultiOutput2() {
        //Edge case: no functions
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", Nd4j.scalar(0.0));
        SDVariable in2 = sd.var("in2", Nd4j.scalar(1.0));

        try {
            sd.createGradFunction();
            fail("Expected exception");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("No loss variables"));
        }

        SDVariable add = in.add(in2);
        sd.createGradFunction();
    }

    @Test
    public void sameDiffPlaceholderGrad() {
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


    @Test
    public void testConvertToConstant() {
        Nd4j.getRandom().setSeed(12345);

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("in", DataType.FLOAT, 1, 3);
        SDVariable w = sd.var("w", Nd4j.rand(DataType.FLOAT, 3, 4));
        SDVariable b = sd.var("b", Nd4j.rand(DataType.FLOAT, 1, 4));
        SDVariable mmul = in.mmul(w);
        SDVariable add = mmul.add(b);
        SDVariable tanh = sd.math().tanh(add);
        SDVariable loss = sd.variance(tanh, true);

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

        //Sanity check on training:
        sd.fit(new SingletonMultiDataSetIterator(new DataSet(inArr, null).toMultiDataSet()), 1);
    }

    @Test
    public void testPlaceholderToConstant() {
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

    @Test
    public void testConvertToVariable() {
        Nd4j.getRandom().setSeed(12345);

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("in", DataType.FLOAT, 1, 3);
        SDVariable w = sd.constant("w", Nd4j.rand(DataType.FLOAT, 3, 4));
        SDVariable b = sd.var("b", Nd4j.rand(DataType.FLOAT, 1, 4));
        SDVariable mmul = in.mmul(w);
        SDVariable add = mmul.add(b);
        SDVariable tanh = sd.math().tanh(add);
        SDVariable loss = sd.variance(tanh, true);

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

    @Test
    public void testDoubleUseOfArray() {
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

    @Test
    public void testMultiGradientRecurrent() {
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

    @Test
    public void testMultiGradientManualRecurrent() {
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
        final SDVariable[] inputSlices = sd.unstack(new String[]{"X_0", "X_1"}, sdInput, 2);

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

    @Test
    public void testMultiGradient() {
        final INDArray input = Nd4j.rand(DataType.DOUBLE, new int[]{3, 4, 2});
        SameDiff sd = SameDiff.create();
        final SDVariable sdInput = sd.var("input", input);

        final SDVariable[] inputSlices = sd.unstack(new String[]{"X_0", "X_1"}, sdInput, 2);
        final val temp = inputSlices[0].add(inputSlices[1]).div(inputSlices[1]).mul(inputSlices[0]);
        final val out = temp.add(temp).add(inputSlices[1]);
        out.norm2("out");

        String err = OpValidation.validate(new TestCase(sd)
                .testFlatBufferSerialization(TestCase.TestSerialization.BOTH)
                .gradientCheck(true));

        assertNull(err);
    }


    @Test
    public void testNonScalarOutput1() {
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

    @Test
    public void testNonScalarOutput2() {
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

    @Test
    public void testNonScalarOutput3() {
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

    @Test
    public void testNonScalarOutput4() {
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

    @Test
    public void testSameDiffBackprop1() {
        SameDiff sd = SameDiff.create();
        final SDVariable a = sd.var("a", Nd4j.rand(4, 4));
        final SDVariable b = sd.var("b", Nd4j.rand(4, 4));
        final SDVariable c = sd.var("c", Nd4j.rand(4, 4));
        final SDVariable d = sd.var("d", Nd4j.rand(4, 4));

        final SDVariable out = a.mmul(b).add(c.mmul(d)).sum();
        out.markAsLoss();

        Map<String,INDArray> g = sd.calculateGradients(null, sd.getVariables().keySet());
    }

    @Test
    public void testSameDiffNoGradForConstantAndPlaceholder() {
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

    @Test
    public void testDuplicateNamePlaceholder() {

        for (int i = 0; i < 2; i++) {
            SameDiff sd = SameDiff.create();
            SDVariable x1 = i == 0 ? sd.placeHolder("a", DataType.FLOAT, 5, 3) : sd.var("a", DataType.FLOAT, 5, 3);
            SDVariable x2 = i == 0 ? sd.placeHolder("b", DataType.FLOAT, 5, 3) : sd.var("b", DataType.FLOAT, 5, 3);
            try {
                sd.placeHolder("a", DataType.FLOAT, 5, 3);
                fail("Expected execption");
            } catch (Throwable t) {
                String m = t.getMessage();
                assertNotNull(m);
                assertTrue(m, m.contains("already exists"));
            }

            try {
                sd.var("a", DataType.FLOAT, 1, 2);
                fail("Expected execption");
            } catch (Throwable t) {
                String m = t.getMessage();
                assertNotNull(m);
                assertTrue(m, m.contains("already exists"));
            }

            try {
                sd.var("a", Nd4j.zeros(1));
                fail("Expected execption");
            } catch (Throwable t) {
                String m = t.getMessage();
                assertNotNull(m);
                assertTrue(m, m.contains("already exists"));
            }

            try {
                sd.var("a", LongShapeDescriptor.fromShape(new long[]{1}, DataType.FLOAT));
                fail("Expected execption");
            } catch (Throwable t) {
                String m = t.getMessage();
                assertNotNull(m);
                assertTrue(m, m.contains("already exists"));
            }

            try {
                sd.constant("a", Nd4j.zeros(1));
                fail("Expected execption");
            } catch (Throwable t) {
                String m = t.getMessage();
                assertNotNull(m);
                assertTrue(m, m.contains("already exists"));
            }
        }
    }

    @Test
    public void testSameDiffGetArrayScalar() {
        final INDArray array = Nd4j.rand(1, 1);
        final SameDiff sd = SameDiff.create();
        final SDVariable a = sd.var("a", array.shape());
        a.getArr();
    }

    @Test
    public void testVariableRenaming() {

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

    @Test
    public void testVariableRenaming2() {

        SameDiff sd = SameDiff.create();
        SDVariable v1 = sd.placeHolder("x", DataType.FLOAT, 3, 4);
        SDVariable v2 = sd.var("y", Nd4j.rand(DataType.FLOAT, 4, 5));
        SDVariable v3 = v1.mmul("oldName", v2);
        SDVariable v4 = v3.std("out", false);

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

    @Test
    public void testPlaceholderShapeValidation() {
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
                    assertTrue(msg, msg.contains("shape") && msg.contains("[2, 3]") && msg
                            .contains(Arrays.toString(v.placeholderShape())));
                }
            }

            try {
                v.setArray(wrongRank1);
                fail("Expected exception");
            } catch (Exception t) {
                String msg = t.getMessage();
                assertTrue(msg, msg.contains("shape") && msg.contains("[1]") && msg
                        .contains(Arrays.toString(v.placeholderShape())));
            }

            try {
                v.setArray(wrongRank2);
                fail("Expected exception");
            } catch (Exception t) {
                String msg = t.getMessage();
                assertTrue(msg, msg.contains("shape") && msg.contains("[3, 4, 5]") && msg
                        .contains(Arrays.toString(v.placeholderShape())));
            }
        }

        //Also try training:
        SDVariable sum = sd.math.mergeAdd(ph1, ph2, ph3, ph4);
        SDVariable mean = sum.add(scalar).mean();
        MultiDataSet mds = new MultiDataSet(new INDArray[]{wrongShape, wrongShape, wrongShape, wrongShape}, null);

        sd.setTrainingConfig(TrainingConfig.builder()
                .dataSetFeatureMapping("ph1", "ph2", "ph3", "ph4")
                .markLabelsUnused()
                .updater(new Adam(1e-3)).build());

        try {
            sd.fit(mds);
        } catch (Exception t) {
            String msg = t.getMessage();
            assertTrue(msg, msg.contains("shape") && msg.contains("[2, 3]"));
        }
    }


    @Test
    public void testInferenceWithoutLabel() {
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

    @Test
    public void testInferenceWithoutUnnecessaryPlaceholders() {
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


    @Test
    public void testConvertDTypes1() {

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
            assertEquals(e.getKey(), DataType.FLOAT, e.getValue().dataType());
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
            assertEquals(e.getKey(), DataType.DOUBLE, e.getValue().dataType());
        }

        assertEquals(DataType.DOUBLE, x.getArr().dataType());
        assertEquals(DataType.DOUBLE, y.getArr().dataType());
    }

    @Test
    public void testConvertDTypes2() {

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
                assertEquals(e.getKey(), DataType.FLOAT, e.getValue().dataType());
            } else {
                assertEquals(e.getKey(), DataType.DOUBLE, e.getValue().dataType());
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
            assertEquals(e.getKey(), DataType.DOUBLE, e.getValue().dataType());
        }

        assertEquals(DataType.DOUBLE, y.getArr().dataType());
    }


    @Test
    public void testGradFnRequiredVars() {
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

    @Test
    public void testNestedIf() throws IOException {
        SameDiff SD = SameDiff.create();
        SDVariable a = SD.var("a", Nd4j.createFromArray(2.0));
        SDVariable b = SD.var("b", Nd4j.createFromArray(5.0));
        SDVariable c = SD.var("c", Nd4j.createFromArray(9.0));
        SDVariable d = SD.var("d", Nd4j.createFromArray(-7.0));

        SDVariable output = SD.ifCond("out", null,
                (sd) -> a.lt(b),
                (sd) -> sd.ifCond(
                        (sd2) -> d.lte(0),
                        (sd2) -> c.add(1),
                        (sd2) -> d),
                (sd) -> c.add(5));
        INDArray out = output.eval();
        assertEquals(Nd4j.createFromArray(10.0), out);

        SD = SameDiff.fromFlatBuffers(SD.asFlatBuffers(false));

        assertEquals(Nd4j.createFromArray(10.0), SD.output(Collections.emptyMap(), "out").get("out"));
    }

    @Test
    public void testWhile() throws IOException {

        SameDiff SD = SameDiff.create();
        SDVariable countIn = SD.constant(5);
        SDVariable sumIn = SD.constant(0);

        SDVariable[] sum = SD.whileLoop("while_1", new SDVariable[]{countIn, sumIn},
                (sd, vars) -> vars[0].gt(0),
                (sd, vars) -> new SDVariable[]{vars[0].sub(1), vars[1].add(vars[0])});

        INDArray out = sum[1].eval();
        assertEquals(15, out.getInt(0));

        String outName = sum[1].name();

        SD = SameDiff.fromFlatBuffers(SD.asFlatBuffers(false));

        assertEquals(15, SD.output(Collections.emptyMap(), outName).get(outName).getInt(0));
    }

    @Test
    @Ignore
    public void testNestedWhile() throws IOException {
        SameDiff SD = SameDiff.create();
        SDVariable countIn = SD.constant(5);
        SDVariable sumIn = SD.constant(0);
        SDVariable sum2 = SD.constant(0);
        //TODO creating constant instead of using sum2 causes errors

        SDVariable[] sum = SD.whileLoop(new SDVariable[]{countIn, sumIn},
                (sd, vars) -> vars[0].gt(0),
                (sd, vars) -> new SDVariable[]{vars[0].sub(1),
                        vars[1].add(sd.whileLoop(new SDVariable[]{vars[0], sum2},
                                (sd2, vars2) -> vars2[0].gt(0),
                                (sd2, vars2) -> new SDVariable[]{vars2[0].sub(1), vars2[1].add(vars2[0])})[1])});

        INDArray out = sum[1].eval();
        assertEquals(35, out.getInt(0));

        String outName = sum[1].name();

        SD = SameDiff.fromFlatBuffers(SD.asFlatBuffers(false));

        assertEquals(35, SD.output(Collections.emptyMap(), outName).get(outName).getInt(0));

    }

    @Test
    public void testNestedWhileIf() throws IOException {
        SameDiff SD = SameDiff.create();
        SDVariable countIn = SD.constant(5);
        SDVariable sumIn = SD.constant(0);
        SDVariable hundred = SD.constant(100);

        SDVariable[] sum = SD.whileLoop(new SDVariable[]{countIn, sumIn},
                (sd, vars) -> vars[0].gte(0),
                (sd, vars) -> new SDVariable[]{vars[0].sub(1), vars[1].add(
                        sd.ifCond((sd2) -> vars[0].eq(0),
                                (sd2) -> vars[0].add(100), //TODO replace with hundred and things break
                                (sd2) -> vars[0])
                )});

        INDArray out = sum[1].eval();
        assertEquals(115, out.getInt(0));

        String outName = sum[1].name();

        SD = SameDiff.fromFlatBuffers(SD.asFlatBuffers(false));

        assertEquals(115, SD.output(Collections.emptyMap(), outName).get(outName).getInt(0));
    }

    @Test
    public void testMod_1(){
        val sd = SameDiff.create();
        val initial = sd.constant("initial", Nd4j.createFromArray(5.f, 6.f, 7.f));
        val four = sd.constant("four", 4.0f);
        val mod = initial.mod("mod",  four);

        val e = Nd4j.createFromArray(1.f, 2.f, 3.f);

        assertEquals(e, mod.eval());
    }

    @Test
    public void castShapeTest1(){
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.constant(Nd4j.createFromArray(1, 2, 3, 4));
        SDVariable casted = x.castTo(DataType.FLOAT);

        assertEquals(casted.dataType(), DataType.FLOAT);
    }

    @Test
    @Ignore // casted shape is null
    public void castShapeTestEmpty(){
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.constant(Nd4j.empty(DataType.INT));
        SDVariable casted = x.castTo(DataType.FLOAT);

        assertEquals(casted.dataType(), DataType.FLOAT);
        assertTrue(casted.getShapeDescriptor().isEmpty());
    }


    @Test
    public void testEmptyShapeVar(){
        SameDiff sd = SameDiff.create();

        try {
            sd.var(DataType.FLOAT, 1, 0, 2);
            fail("Expected exception");
        } catch (IllegalArgumentException e){
            String m = e.getMessage();
            assertTrue(m, m.contains("variable") && m.contains("empty") && m.contains("0"));
        }

        try {
            sd.var(Nd4j.create(1, 0, 2));
            fail("Expected exception");
        } catch (IllegalArgumentException e){
            String m = e.getMessage().toLowerCase();
            assertTrue(m, m.contains("variable") && m.contains("empty") && m.contains("0"));
        }
    }

    @Test
    public void testPReLU(){
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

    @Test
    public void testSameDiffSeedReproducibilityVarInit() {

        SameDiff sd0 = SameDiff.create();
        SameDiff sd1 = SameDiff.create();
        Nd4j.getRandom().setSeed(12345);
        SDVariable rand0 = sd0.var("random", new UniformInitScheme('c', 3), DataType.FLOAT, 3, 1);

        Nd4j.getRandom().setSeed(12345);
        SDVariable rand1 = sd1.var("random", new UniformInitScheme('c', 3), DataType.FLOAT, 3, 1);


        Nd4j.getRandom().setSeed(0);
        System.out.println(rand0.eval());

        Nd4j.getRandom().setSeed(0);
        System.out.println(rand1.eval());

        INDArray a0 = rand0.eval();
        Nd4j.getRandom().setSeed(0);
        INDArray a1 = rand1.eval();
        assertEquals(a0, a1);
    }


    @Test
    public void testCalculateGradientsAndOutputs(){
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("in", DataType.FLOAT, -1, 4);
        SDVariable w = sd.var("w", Nd4j.rand(DataType.FLOAT, 4, 3));
        SDVariable b = sd.var("b", Nd4j.rand(DataType.FLOAT, 3));
        SDVariable z = in.mmul(w).add("z", b);
        SDVariable softmax = sd.nn.softmax("softmax", z);

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
    
    @Test
	public void testConcatVariableGrad() {
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

	@Test
	public void testSliceVariableGrad() {
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
}
