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
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.*;
import org.nd4j.autodiff.samediff.api.OutAndGrad;
import org.nd4j.autodiff.util.SameDiffUtils;
import org.nd4j.autodiff.validation.OpValidation;
import org.nd4j.autodiff.validation.TestCase;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.ExternalErrorsFunction;
import org.nd4j.linalg.api.ops.impl.transforms.any.IsMax;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;

@Slf4j
@NativeTag
@Tag(TagNames.SAMEDIFF)
public class SameDiffGradientTest extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
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
            INDArray dLdIn = g.get("in");

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
            assertTrue(exp.equalsWithEps(dLdIn, eps));
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
        INDArray dLdIn = g.get("in");

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

        Map<String,INDArray> g = sd.calculateGradients(Collections.emptyMap(), sd.getVariables().keySet());
        INDArray dLdIn = g.get("in");

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
        INDArray dLdIn = g.get("in");

        //If L = prod(in)
        //then dL/dIn = prod(in) / in       i.e., product of input *excluding* in_i as (d/dx(xyzabc) = yzabc

        INDArray exp = arr.rdiv(p);
        assertEquals(exp, dLdIn);
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
    public void testGradFnRequiredVars(Nd4jBackend backend) {
        //User can explicitly request that gradients for specific vars are available when differentiating (creating grad function),
        // even if they normally wouldn't be needed or calculated

        for (boolean reqPhVar : new boolean[]{false, true}) {
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
}
