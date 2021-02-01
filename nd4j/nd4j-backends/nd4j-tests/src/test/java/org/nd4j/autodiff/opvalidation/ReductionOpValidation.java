/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.autodiff.opvalidation;

import lombok.extern.slf4j.Slf4j;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.OpValidationSuite;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.validation.OpTestCase;
import org.nd4j.autodiff.validation.OpValidation;
import org.nd4j.autodiff.validation.TestCase;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.indexaccum.custom.ArgAmax;
import org.nd4j.linalg.api.ops.impl.indexaccum.custom.ArgAmin;
import org.nd4j.linalg.api.ops.impl.loss.SoftmaxCrossEntropyWithLogitsLoss;
import org.nd4j.linalg.api.ops.impl.reduce.Moments;
import org.nd4j.linalg.api.ops.impl.reduce.NormalizeMoments;
import org.nd4j.linalg.api.ops.impl.reduce.SufficientStatistics;
import org.nd4j.linalg.api.ops.impl.reduce.floating.AMean;
import org.nd4j.linalg.api.ops.impl.reduce.floating.Entropy;
import org.nd4j.linalg.api.ops.impl.reduce.floating.Mean;
import org.nd4j.linalg.api.ops.impl.reduce.floating.Norm1;
import org.nd4j.linalg.api.ops.impl.reduce.floating.Norm2;
import org.nd4j.linalg.api.ops.impl.reduce.floating.NormMax;
import org.nd4j.linalg.api.ops.impl.reduce.floating.ShannonEntropy;
import org.nd4j.linalg.api.ops.impl.reduce.floating.SquaredNorm;
import org.nd4j.linalg.api.ops.impl.reduce.same.ASum;
import org.nd4j.linalg.api.ops.impl.reduce3.CosineDistance;
import org.nd4j.linalg.api.ops.impl.reduce3.CosineSimilarity;
import org.nd4j.linalg.api.ops.impl.reduce3.Dot;
import org.nd4j.linalg.api.ops.impl.reduce3.EuclideanDistance;
import org.nd4j.linalg.api.ops.impl.reduce3.HammingDistance;
import org.nd4j.linalg.api.ops.impl.reduce3.JaccardDistance;
import org.nd4j.linalg.api.ops.impl.reduce3.ManhattanDistance;
import org.nd4j.linalg.api.ops.impl.summarystats.StandardDeviation;
import org.nd4j.linalg.api.ops.impl.transforms.custom.SoftMax;
import org.nd4j.linalg.checkutil.NDArrayCreationUtil;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.common.primitives.Pair;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.*;

@Slf4j
@RunWith(Parameterized.class)
public class ReductionOpValidation extends BaseOpValidation {

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    public ReductionOpValidation(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testStdev() {
        List<String> errors = new ArrayList<>();

        for (Pair<INDArray, String> p : NDArrayCreationUtil.getAllTestMatricesWithShape(3, 4, 12345, DataType.DOUBLE)) {
            for (boolean biasCorrected : new boolean[]{false, true}) {
                SameDiff sd = SameDiff.create();
                SDVariable var = sd.var("in", p.getFirst());
                SDVariable stdev = var.std(biasCorrected);

                INDArray expOut = p.getFirst().std(biasCorrected);

                TestCase tc = new TestCase(sd)
                        .testName(p.getSecond() + " - biasCorrected=" + biasCorrected)
                        .expected(stdev, expOut)
                        .gradientCheck(false);

                String err = OpValidation.validate(tc);
                if (err != null) {
                    errors.add(err);
                }
            }
        }
        assertEquals(errors.toString(), 0, errors.size());
    }

    @Test
    public void testZeroCount() {
        List<String> allFailed = new ArrayList<>();
        for (int i = 0; i < 21; i++) {
            SameDiff sd = SameDiff.create();

            INDArray ia;
            if (i == 0) {
                //Not gradient checkable for 0 and 1 values
                ia = Nd4j.create(new int[]{2, 2}, new float[]{0, 1, 0, 1}).castTo(DataType.DOUBLE);
            } else {
                ia = Nd4j.rand(DataType.DOUBLE,2, 2);
            }

            SDVariable input = sd.var("in", DataType.DOUBLE, 2, 2);
            sd.associateArrayWithVariable(ia, input);

            SDVariable nonZero = sd.math().countNonZero(input);
            SDVariable zero = sd.math().countZero(input);

            SDVariable loss = nonZero.add(zero).castTo(DataType.DOUBLE).std(true);

            String error = OpValidation.validate(new TestCase(sd)
                    .expectedOutput(nonZero.name(), Nd4j.scalar(DataType.LONG, i == 0 ? 2.0 : 4.0))
                    .expectedOutput(zero.name(), Nd4j.scalar(DataType.LONG, i == 0 ? 2.0 : 0.0))
                    .gradientCheck(false)
            );
            if (error != null)
                allFailed.add(error);
        }
        assertEquals(allFailed.toString(), 0, allFailed.size());
    }


    @Test
    public void testZeroFraction() {
        List<String> allFailed = new ArrayList<>();
        for (int i = 0; i < 2; i++) {
            SameDiff sd = SameDiff.create();

            INDArray ia;
            if (i == 0) {
                //Not gradient checkable for 0 and 1 values
                ia = Nd4j.create(new int[]{2, 2}, new float[]{0, 1, 0, 1});
            } else {
                ia = Nd4j.rand(DataType.FLOAT, 2, 2);
            }

            SDVariable input = sd.var("in", 2, 2);
            sd.associateArrayWithVariable(ia, input);

            SDVariable zeroFraction = sd.math().zeroFraction(input);

            String error = OpValidation.validate(new TestCase(sd)
                    .expectedOutput(zeroFraction.name(), Nd4j.scalar(i == 0 ? 0.5f : 0.0f))
                    .gradientCheck(i != 0)
            );
            if (error != null)
                allFailed.add(error);
        }

        assertEquals(allFailed.toString(), 0, allFailed.size());
    }

    @Test
    public void testReductionGradientsSimple() {
        //OpValidationSuite.ignoreFailing();  //TODO TEMPORARY DUE TO CRASHES
        //Test reductions: final and only function
        Nd4j.getRandom().setSeed(12345);

        List<String> failed = new ArrayList<>();
        for (int i = 0; i < 21; i++) {

            SameDiff sd = SameDiff.create();

            int nOut = 4;
            int minibatch = 10;
            SDVariable input = sd.var("in", minibatch, nOut);
            INDArray inputArr = Nd4j.randn(minibatch, nOut).muli(100);
            long length = nOut * minibatch;

            SDVariable loss;
            String name;
            TestCase tc = new TestCase(sd);
            boolean gradCheck = true;
            switch (i) {
                case 0:
                    loss = sd.mean("loss", input);
                    name = "mean";
                    tc.expectedOutput("loss", inputArr.mean());
                    break;
                case 1:
                    loss = sd.sum("loss", input);
                    name = "sum";
                    tc.expectedOutput("loss", inputArr.sum());
                    break;
                case 2:
                    loss = sd.standardDeviation("loss", input, true);
                    name = "stdev";
                    tc.expectedOutput("loss", inputArr.std(true));
                    break;
                case 3:
                    loss = sd.min("loss", input);
                    name = "min";
                    tc.expectedOutput("loss", inputArr.min());
                    break;
                case 4:
                    loss = sd.max("loss", input);
                    name = "max";
                    tc.expectedOutput("loss", inputArr.max());
                    break;
                case 5:
                    loss = sd.variance("loss", input, true);
                    name = "variance";
                    tc.expectedOutput("loss", inputArr.var());
                    break;
                case 6:
                    inputArr = Nd4j.rand(minibatch, nOut).addi(0.5);
                    loss = sd.prod("loss", input);
                    tc.expectedOutput("loss", inputArr.prod());
                    name = "prod";
                    break;
                case 7:
                    loss = sd.norm1("loss", input);
                    name = "norm1";
                    tc.expectedOutput("loss", inputArr.norm1());
                    break;
                case 8:
                    loss = sd.norm2("loss", input);
                    name = "norm2";
                    tc.expectedOutput("loss", inputArr.norm2());
                    break;
                case 9:
                    loss = sd.normmax("loss", input);
                    name = "normmax";
                    tc.expectedOutput("loss", inputArr.normmax());
                    break;
                case 10:
                    loss = sd.math().countNonZero("loss", input, 0,1);
                    name = "countNonZero";
                    tc.expectedOutput("loss", Nd4j.scalar(inputArr.length()));
                    gradCheck = false;  //Long out, not floating point
                    break;
                case 11:
                    loss = sd.math().countZero("loss", input, 0,1);
                    name = "countZero";
                    tc.expectedOutput("loss", Nd4j.scalar(0L));
                    gradCheck = false;  //Long out, not floating point
                    break;
                case 12:
                    loss = sd.math().amax("loss", input, 0,1);
                    name = "amax";
                    tc.expectedOutput("loss", inputArr.amax());
                    break;
                case 13:
                    loss = sd.math().amin("loss", input, 0,1);
                    name = "amin";
                    tc.expectedOutput("loss", inputArr.amin());
                    break;
                case 14:
                    loss = sd.math().asum("loss", input, 0,1);
                    name = "asum";
                    tc.expectedOutput("loss", Nd4j.getExecutioner().exec(new ASum(inputArr.dup())));
                    break;
                case 15:
                    loss = sd.math().amean("loss", input, 0,1);
                    name = "amean";
                    tc.expectedOutput("loss", Nd4j.getExecutioner().exec(new AMean(inputArr.dup())));
                    break;
                case 16:
                    loss = sd.math().entropy("loss", input, 0,1);
                    name = "entropy";
                    inputArr = Nd4j.linspace(0.01, 0.99, length, DataType.DOUBLE).reshape('c', minibatch, nOut);
                    tc.expected("loss", inputArr.mul(Transforms.log(inputArr, true)).sum(Integer.MAX_VALUE).negi());
                    break;
                case 17:
                    inputArr = Nd4j.rand(minibatch, nOut);
                    name = "logsumexp";
                    loss = sd.math().logSumExp("loss", input);
                    INDArray expArr = Transforms.exp(inputArr);
                    double sum = expArr.sumNumber().doubleValue();
                    tc.expected("loss", Nd4j.scalar(Math.log(sum)));
                    break;
                case 18:
                    inputArr = Nd4j.rand(minibatch, nOut);
                    name = "sqnorm";
                    loss = sd.squaredNorm("loss", input);
                    double norm2 = inputArr.norm2Number().doubleValue();
                    tc.expected("loss", Nd4j.scalar(norm2 * norm2));
                    break;
                case 19:
                    inputArr = Nd4j.rand(minibatch, nOut);
                    name = "logEntropy";
                    loss = sd.math().logEntropy("loss", input, 0,1);
                    double logEntropy = inputArr.logEntropyNumber().doubleValue();
                    tc.expected(loss, Nd4j.scalar(logEntropy));
                    break;
                case 20:
                    inputArr = Nd4j.rand(minibatch, nOut);
                    name = "shannonEntropy";
                    loss = sd.math().shannonEntropy("loss", input, 0);
                    double shannonEntropy = inputArr.shannonEntropyNumber().doubleValue();
                    tc.expected(loss, Nd4j.scalar(shannonEntropy));
                    if (OpValidationSuite.IGNORE_FAILING) {
                        continue;
                    }
                    break;
                default:
                    throw new RuntimeException();
            }


            String msg = "test: " + i + " - " + name;
            log.info("*** Starting test: " + msg);

            sd.associateArrayWithVariable(inputArr, input);
            if(gradCheck) {
                sd.addLossVariable(loss);
            }

            tc.testName(msg);
            if(!gradCheck){
                tc.gradientCheck(false);
            }

            String error = OpValidation.validate(tc, true);
            if (error != null)
                failed.add(error);
        }

        assertEquals(failed.toString(), 0, failed.size());
    }

    @Test
    public void testReductionGradients1() {
        //Test reductions: final, but *not* the only function
        Nd4j.getRandom().setSeed(12345);

        List<String> failed = new ArrayList<>();

        for (int dim : new int[]{0, Integer.MAX_VALUE}) {    //These two cases are equivalent here

            for (int i = 0; i < 16; i++) {

                SameDiff sd = SameDiff.create();

                int nOut = 4;
                int minibatch = 10;
                SDVariable input = sd.var("in", DataType.DOUBLE, minibatch, nOut);
                SDVariable label = sd.var("label", DataType.DOUBLE, minibatch, nOut);

                SDVariable diff = input.sub(label);
                SDVariable sqDiff = diff.mul(diff);
                SDVariable msePerEx = sd.mean("msePerEx", sqDiff, 1);

                SDVariable loss;
                String name;
                TestCase tc = new TestCase(sd);
                boolean uDistInput = false;
                boolean gradientCheckable = true;
                INDArray exp = null;
                switch (i) {
                    case 0:
                        loss = sd.mean("loss", msePerEx, dim);
                        name = "mean";
                        break;
                    case 1:
                        loss = sd.sum("loss", msePerEx, dim);
                        name = "sum";
                        break;
                    case 2:
                        loss = sd.standardDeviation("loss", msePerEx, true, dim);
                        name = "stdev";
                        break;
                    case 3:
                        loss = sd.min("loss", msePerEx, dim);
                        name = "min";
                        break;
                    case 4:
                        loss = sd.max("loss", msePerEx, dim);
                        name = "max";
                        break;
                    case 5:
                        loss = sd.variance("loss", msePerEx, true, dim);
                        name = "variance";
                        break;
                    case 6:
                        loss = sd.prod("loss", msePerEx, dim);
                        name = "prod";
                        break;
                    case 7:
                        loss = sd.norm1("loss", msePerEx, dim);
                        name = "norm1";
                        break;
                    case 8:
                        loss = sd.norm2("loss", msePerEx, dim);
                        name = "norm2";
                        break;
                    case 9:
                        loss = sd.normmax("loss", msePerEx, dim);
                        name = "normmax";
                        break;
                    case 10:
                        loss = sd.math().entropy("loss", msePerEx, dim);
                        name = "entropy";
                        break;
                    case 11:
                        name = "logEntropy";
                        loss = sd.math().logEntropy("loss", msePerEx, dim);
                        uDistInput = true;
                        break;
                    case 12:
                        loss = sd.math().amax("loss", msePerEx, dim);
                        name = "amax";
                        break;
                    case 13:
                        loss = sd.math().amin("loss", msePerEx, dim);
                        name = "amin";
                        break;
                    case 14:
                        loss = sd.math().asum("loss", msePerEx, dim);
                        name = "asum";
                        break;
                    case 15:
                        loss = sd.math().amean("loss", msePerEx, dim);
                        name = "amean";
                        break;
                    default:
                        throw new RuntimeException();
                }


                String msg = "(test " + i + " - " + name + ", dimension=" + dim + ")";
                log.info("*** Starting test: " + msg);

                INDArray inputArr = uDistInput ? Nd4j.rand(DataType.DOUBLE, minibatch, nOut) : Nd4j.randn(DataType.DOUBLE, minibatch, nOut).muli(100);
                INDArray labelArr = uDistInput ? Nd4j.rand(DataType.DOUBLE, minibatch, nOut) : Nd4j.randn(DataType.DOUBLE, minibatch, nOut).muli(100);

                sd.associateArrayWithVariable(inputArr, input);
                sd.associateArrayWithVariable(labelArr, label);

                tc.gradientCheck(gradientCheckable);
                if(exp != null){
                    tc.expectedOutput(loss.name(), exp);
                }

                String error = OpValidation.validate(tc);
                if (error != null) {
                    failed.add(name);
                }
            }
        }

        assertEquals("Failed: " + failed, 0, failed.size());
    }

    @Test
    public void testReductionGradients2() {
        //Test reductions: NON-final function
        Nd4j.getRandom().setSeed(12345);

        int d0 = 3;
        int d1 = 4;
        int d2 = 5;

        List<String> failed = new ArrayList<>();
        for (int reduceDim : new int[]{0, 1, 2}) {
            for (int i = 0; i < 18; i++) {

                long[] outShape;
                switch (reduceDim) {
                    case 0:
                        outShape = new long[]{d1, d2};
                        break;
                    case 1:
                        outShape = new long[]{d0, d2};
                        break;
                    case 2:
                        outShape = new long[]{d0, d1};
                        break;
                    default:
                        throw new RuntimeException();
                }

                SameDiff sd = SameDiff.create();
                sd.setLogExecution(false);

                SDVariable in = sd.var("in", d0, d1, d2);
                SDVariable label = sd.var("label", outShape);
                SDVariable second = in.mul(2);

                double maxRelError = 1e-4;
                double minAbsError = 1e-4;
                INDArray inputArr = Nd4j.randn(DataType.DOUBLE, d0, d1, d2).muli(1000);
                INDArray labelArr = Nd4j.randn(DataType.DOUBLE, outShape).muli(1000);
                SDVariable reduced;
                String name;
                TestCase tc = new TestCase(sd);
                boolean gradCheck = true;
                INDArray exp = null;
                switch (i) {
                    case 0:
                        reduced = sd.mean("reduced", second, reduceDim);
                        name = "mean";
                        break;
                    case 1:
                        inputArr.divi(100);
                        labelArr.divi(100);
                        reduced = sd.sum("reduced", second, reduceDim);
                        name = "sum";
                        break;
                    case 2:
                        reduced = sd.standardDeviation("reduced", second, true, reduceDim);
                        inputArr.divi(1000);
                        labelArr.divi(1000);
                        name = "stdev";
                        break;
                    case 3:
                        reduced = sd.min("reduced", second, reduceDim);
                        name = "min";
                        break;
                    case 4:
                        reduced = sd.max("reduced", second, reduceDim);
                        name = "max";
                        break;
                    case 5:
                        //Variance is a bit finniky for gradient checks, due to huge score/output...
                        maxRelError = 1e-3;
                        minAbsError = 1;        //Most gradients are in the range 1k to >100k
                        inputArr.divi(10);
                        labelArr.divi(100);
                        BooleanIndexing.replaceWhere(inputArr, Nd4j.rand(inputArr.shape()).muli(100).addi(100), Conditions.absLessThan(1.0));
                        reduced = sd.variance("reduced", second, true, reduceDim);
                        name = "variance";
                        break;
                    case 6:
                        inputArr.assign(Nd4j.rand(DataType.DOUBLE, new int[]{d0, d1, d2}).addi(0.5));
                        labelArr.assign(Nd4j.rand(DataType.DOUBLE, outShape).addi(0.5));
                        reduced = sd.prod("reduced", second, reduceDim);
                        name = "prod";
                        break;
                    case 7:
                        maxRelError = 1e-4;
                        inputArr.assign(Nd4j.rand(DataType.DOUBLE, new int[]{d0, d1, d2}).muli(10));
                        labelArr.assign(Nd4j.rand(DataType.DOUBLE, outShape).muli(10));
                        reduced = sd.norm1("reduced", second, reduceDim);
                        name = "norm1";
                        break;
                    case 8:
                        maxRelError = 1e-3; //Norm2 can also run into numerical precision issues
                        reduced = sd.norm2("reduced", second, reduceDim);
                        name = "norm2";
                        break;
                    case 9:
                        inputArr = Nd4j.rand(DataType.DOUBLE, new int[]{d0, d1, d2});
                        labelArr = Nd4j.rand(DataType.DOUBLE, outShape);
                        reduced = sd.normmax("reduced", second, reduceDim);
                        name = "normmax";
                        break;
                    case 10:
                        reduced = sd.argmax("reduced", second, reduceDim);
                        gradCheck = false;
                        exp = inputArr.mul(2).argMax(reduceDim);
                        name = "argmax";
                        break;
                    case 11:
                        reduced = sd.argmin("reduced", second, reduceDim);
                        gradCheck = false;
                        exp = Nd4j.argMin(inputArr.mul(2), reduceDim);
                        name = "argmin";
                        break;
                    case 12:
                        reduced = sd.math().countNonZero("reduced", second, reduceDim);
                        gradCheck = false;
                        exp = inputArr.mul(2).neq(0).castTo(DataType.LONG).sum(reduceDim);
                        name = "countNonZero";
                        break;
                    case 13:
                        reduced = sd.math().countZero("reduced", second, reduceDim);
                        gradCheck = false;
                        exp = inputArr.mul(2).eq(0).castTo(DataType.LONG).sum(reduceDim);
                        name = "countZero";
                        break;
                    case 14:
                        reduced = sd.math().amax("reduced", second, reduceDim);
                        name = "amax";
                        break;
                    case 15:
                        reduced = sd.math().amin("reduced", second, reduceDim);
                        name = "amin";
                        break;
                    case 16:
                        reduced = sd.math().asum("reduced", second, reduceDim);
                        name = "asum";
                        break;
                    case 17:
                        reduced = sd.math().amean("reduced", second, reduceDim);
                        name = "amean";
                        break;
                    default:
                        throw new RuntimeException();
                }

                SDVariable add = reduced.castTo(DataType.DOUBLE).add(1.0);

                SDVariable diff = label.sub(add);
                SDVariable sqDiff = diff.mul(diff);
                SDVariable mseLoss = sd.mean("loss", sqDiff);


                String msg = "(test " + i + " - " + name + ", dimension=" + reduceDim + ")";
                log.info("*** Starting test: " + msg);

                sd.associateArrayWithVariable(inputArr, in);
                sd.associateArrayWithVariable(labelArr, label);

                tc.gradCheckMaxRelativeError(maxRelError);
                tc.gradCheckMinAbsError(minAbsError);
                tc.gradientCheck(gradCheck);
                if(exp != null){
                    tc.expected(reduced, exp);
                }

                String error = OpValidation.validate(tc);
                if (error != null) {
                    failed.add(name + " - " + error);
                }
            }
        }

        assertEquals("Failed: " + failed, 0, failed.size());
    }


    @Test
    public void testReduce3() {
        Nd4j.getRandom().setSeed(12345);

        int d0 = 3;
        int d1 = 4;
        int d2 = 5;

        List<String> failed = new ArrayList<>();
        for (int[] reduceDims : new int[][]{{Integer.MAX_VALUE}, {0, 1, 2}, {0}, {1}, {2}, {0, 1}, {0, 2}, {1, 2}}) {
            for (int i = 0; i < 7; i++) {

                SameDiff sd = SameDiff.create();
                sd.setLogExecution(false);


                SDVariable in = sd.var("in", d1, d1, d2);
                SDVariable in2 = sd.var("in2", d0, d1, d2);

                INDArray inArr = Nd4j.randn(new int[]{d0, d1, d2}).muli(100);
                INDArray in2Arr = Nd4j.randn(inArr.shape()).muli(100);

                INDArray exp;
                SDVariable reduced;
                String name;
                TestCase tc = new TestCase(sd);
                Double maxRelError = null;
                switch (i) {
                    case 0:
                        reduced = sd.math().manhattanDistance(in, in2, reduceDims);
                        name = "manhattan";
                        exp = Nd4j.getExecutioner().exec(new ManhattanDistance(inArr, in2Arr, null, false, false, reduceDims));
                        break;
                    case 1:
                        reduced = sd.math().euclideanDistance(in, in2, reduceDims);
                        name = "euclidean";
                        exp = Nd4j.getExecutioner().exec(new EuclideanDistance(inArr, in2Arr, null, false, false, reduceDims));
                        break;
                    case 2:
                        inArr.muli(1e-4);
                        in2Arr.muli(1e-4);
                        reduced = sd.math().cosineSimilarity(in, in2, reduceDims);
                        name = "cosine";
                        exp = Nd4j.getExecutioner().exec(new CosineSimilarity(inArr, in2Arr, null, false, false, reduceDims));
                        maxRelError = 1e-4;
                        break;
                    case 3:
                        reduced = sd.math().cosineDistance(in, in2, reduceDims);
                        name = "cosinedistance";
                        exp = Nd4j.getExecutioner().exec(new CosineDistance(inArr, in2Arr, null, false, false, reduceDims));
                        maxRelError = 1e-4;
                        break;
                    case 4:
                        reduced = sd.math().hammingDistance(in, in2, reduceDims);
                        name = "hamming";
                        exp = Nd4j.getExecutioner().exec(new HammingDistance(inArr, in2Arr, null, false, false, reduceDims));
                        break;
                    case 5:
                        name = "jaccard";
                        reduced = sd.math().jaccardDistance(name, in, in2, reduceDims);
                        inArr.divi(100).addi(0.1);
                        in2Arr.divi(100).addi(0.1);
                        exp = Nd4j.getExecutioner().exec(new JaccardDistance(inArr, in2Arr, null, false, false, reduceDims));

                        if (OpValidationSuite.IGNORE_FAILING && reduceDims.length == 2)
                            continue;
                        break;
                    case 6:
                        if (OpValidationSuite.IGNORE_FAILING) {
                            //https://github.com/deeplearning4j/deeplearning4j/issues/6069
                            continue;
                        }
                        name = "dot";
                        reduced = sd.dot(name, in, in2, reduceDims);
                        exp = Nd4j.getExecutioner().exec(new Dot(inArr, in2Arr, null, true, false, reduceDims));
                        break;
                    default:
                        throw new RuntimeException();
                }

                //Sum: note that this should be a no-op for the full array cases
                SDVariable sum = sd.sum(reduced, Integer.MAX_VALUE);


                String msg = "(test " + i + " - " + name + ", dimensions=" + Arrays.toString(reduceDims) + ")";
                log.info("*** Starting test: " + msg);

                sd.associateArrayWithVariable(inArr, in);
                sd.associateArrayWithVariable(in2Arr, in2);

                tc.expected(reduced, exp);

                if(maxRelError != null)
                    tc.gradCheckMaxRelativeError(maxRelError);

                String error = OpValidation.validate(tc, true);
                if (error != null) {
                    failed.add(msg + " - " + error);
                }
            }
        }

        assertEquals("Failed: " + failed, 0, failed.size());
    }

    @Test
    public void testMoments() {
        for (int[] axes : new int[][]{{0}, {1}, {0, 1}}) {
            INDArray input = Nd4j.linspace(1, 12, 12).reshape(3, 4);

            SameDiff sd = SameDiff.create();
            SDVariable in = sd.var("in", input);

            SDVariable[] moments = sd.math().moments(in, axes);
            INDArray expMean = input.mean(axes);
            INDArray expVar = input.var(false, axes);

            SDVariable loss;
            if (axes.length < 2) {
                loss = moments[0].add(moments[1]).std(true);
            } else {
                loss = moments[0].add(moments[1]).mean();
            }


            String msg = Arrays.toString(axes);

            TestCase tc = new TestCase(sd)
                    .testName(msg)
                    .expected(moments[0], expMean)
                    .expected(moments[1], expVar);

            String err = OpValidation.validate(tc);
            assertNull(err);
        }
    }

    @Test
    public void testMomentsOp() {
        int[] axes = new int[]{0};
            INDArray input = Nd4j.linspace(1, 12, 12).reshape(3, 4);

        INDArray outMean = Nd4j.createUninitialized(new long[]{4});
        INDArray outVar = Nd4j.createUninitialized(new long[]{4});

        OpTestCase tc = new OpTestCase(new Moments(input, outMean, outVar, axes));

        tc.expectedOutput(0, input.mean(axes).reshape(4));
        tc.expectedOutput(1, input.var(false, axes).reshape(4));

        String err = OpValidation.validate(tc);
        assertNull(err);
    }

    @Test
    public void testNormalizeMomentsOp() {
        INDArray data = Nd4j.linspace(1, 100, 100, DataType.DOUBLE).reshape(10, 10);
        INDArray ssSum = data.sum(0);
        INDArray ssSqSum = data.mul(data).sum(0);

        INDArray meanExp = data.mean(0);
        INDArray varExp = data.var(false, 0);

        INDArray mean = Nd4j.createUninitialized(DataType.DOUBLE, meanExp.shape());
        INDArray var = Nd4j.createUninitialized(DataType.DOUBLE, varExp.shape());

        OpTestCase op = new OpTestCase(new NormalizeMoments(Nd4j.scalar(DataType.DOUBLE, 10), ssSum, ssSqSum, mean, var));
        op.expectedOutput(0, meanExp);
        op.expectedOutput(1, varExp);

        String err = OpValidation.validate(op);
        assertNull(err);
    }

    @Test
    public void testAllAny() {

        INDArray allZeros = Nd4j.zeros(DataType.FLOAT, 3, 4);
        INDArray allOnes = Nd4j.ones(DataType.FLOAT, 3, 4);
        INDArray mixed = Nd4j.zeros(DataType.FLOAT, 3, 4);
        mixed.getRow(1).assign(1.0);

        INDArray[] in = new INDArray[]{allZeros, allOnes, mixed};
        boolean[] expAll = new boolean[]{false, true, false};
        boolean[] expAny = new boolean[]{false, true, true};

        for (int i = 0; i < 3; i++) {
            SameDiff sd = SameDiff.create();

            SDVariable s = sd.var("in", in[i]);
            SDVariable all = sd.all(s);
            SDVariable any = sd.any(s);

            String err = OpValidation.validate(new TestCase(sd)
                    .gradientCheck(false)
                    .expected(all, Nd4j.scalar(expAll[i]))
                    .expected(any, Nd4j.scalar(expAny[i])));

            assertNull(err);
        }
    }

    @Test
    public void testIndexAccum() {
        List<String> failed = new ArrayList<>();
        List<int[]> dims = Arrays.asList(new int[]{0}, new int[]{1}, new int[]{0, 1} /*, new int[0]*/);

        INDArray in = Nd4j.rand(DataType.DOUBLE,3, 4);

        for (int t = 0; t < 3; t++) {
            int[] d = dims.get(t);
            for (int i = 0; i < 7; i++) {

                int[] dim = d.length == 0 ? new int[0] : d;

                SameDiff sd = SameDiff.create();
                SDVariable s = sd.var("in", in);
                SDVariable reduce;

                String name;
                INDArray exp;
                switch (i) {
                    case 0:
                        reduce = s.argmax(dim);
                        exp = Nd4j.argMax(in, dim);
                        name = "argmax";
                        break;
                    case 1:
                        reduce = s.argmin(dim);
                        exp = Nd4j.argMin(in, dim);
                        name = "argmin";
                        break;
                    case 2:
                        reduce = sd.math().iamax(s, dim);
                        exp = Nd4j.getExecutioner().exec(new ArgAmax(in.dup(), dim))[0];
                        name = "iamax";
                        break;
                    case 3:
                        reduce = sd.math().iamin(s, dim);
                        exp = Nd4j.getExecutioner().exec(new ArgAmin(in.dup(), dim))[0];
                        name = "iamin";
                        break;
                    case 4:
                        reduce = sd.math().firstIndex(s, Conditions.greaterThan(0), dim);
                        exp = in.sum(dim).assign(0).castTo(DataType.INT64);
                        name = "firstindex";
                        break;
                    case 5:
                        reduce = sd.math().lastIndex(s, Conditions.greaterThan(0), dim);
                        if (t == 0) exp = Nd4j.createFromArray(2L, 2, 2, 2);
                        else if (t == 1) exp = Nd4j.createFromArray(3L, 3, 3);
                        else exp = Nd4j.scalar(11L);
                        name = "lastindex";
                        break;
                    case 6:
                        reduce = sd.matchConditionCount("count", s, Conditions.greaterThan(0), false, dim);
                        if (t == 0) exp = Nd4j.createFromArray(3L, 3, 3, 3);
                        else if (t == 1) exp = Nd4j.createFromArray(4L, 4, 4);
                        else exp = Nd4j.scalar(12L);
                        name = "matchConditionCount";
                        break;
                    default:
                        throw new RuntimeException();
                }
                SDVariable preCast = reduce;
                reduce = reduce.castTo(DataType.DOUBLE);

                SDVariable loss;
                if (dim == null || dim.length == 2) {
                    loss = reduce.mean();
                } else {
                    loss = reduce.std(true);
                }

                TestCase tc = new TestCase(sd)
                        .expected(preCast, exp)
                        .gradientCheck(false)
                        .testName(name + " - " + (dim == null ? null : Arrays.toString(dim)));

                log.info("Starting: {}", tc.testName());
                String err = OpValidation.validate(tc, true);
                if (err != null) {
                    failed.add(err);
                }
            }
        }

        assertEquals(failed.toString(), 0, failed.size());
    }


    @Test
    public void testReduce3_2() {
        Nd4j.getRandom().setSeed(12345);

        int d0 = 3;
        int d1 = 4;
        int d2 = 5;

        for (int[] reduceDims : new int[][]{{Integer.MAX_VALUE}, {0, 1, 2}, {0}, {1}, {2}, {0, 1}, {0, 2}, {1, 2}}) {
            for (int i = 0; i < 6; i++) {

                SameDiff sd = SameDiff.create();
                sd.setLogExecution(false);

                INDArray a = Nd4j.rand(DataType.DOUBLE, d0, d1, d2);
                INDArray b = Nd4j.rand(DataType.DOUBLE, d0, d1, d2);


                SDVariable in = sd.var("in", a);
                SDVariable in2 = sd.var("in2", b);

                INDArray expOut;
                SDVariable reduced;
                String name;
//                System.out.println(i);
                switch (i) {
                    case 0:
                        reduced = sd.math().manhattanDistance(in, in2, reduceDims);
                        name = "manhattan";
                        expOut = Nd4j.getExecutioner().exec(new ManhattanDistance(a, b, null, false, reduceDims));
                        break;
                    case 1:
                        reduced = sd.math().euclideanDistance(in, in2, reduceDims);
                        name = "euclidean";
                        expOut = Nd4j.getExecutioner().exec(new EuclideanDistance(a, b, null, false, reduceDims));
                        break;
                    case 2:
                        reduced = sd.math().cosineSimilarity(in, in2, reduceDims);
                        name = "cosine";
                        expOut = Nd4j.getExecutioner().exec(new CosineSimilarity(a, b, null, false, reduceDims));
                        break;
                    case 3:
                        reduced = sd.math().jaccardDistance(in, in2, reduceDims);
                        name = "jaccard";
                        expOut = Nd4j.getExecutioner().exec(new JaccardDistance(a, b, null, false, reduceDims));
                        break;
                    case 4:
                        reduced = sd.math().hammingDistance(in, in2, reduceDims);
                        name = "hamming";
                        expOut = Nd4j.getExecutioner().exec(new HammingDistance(a, b, null, false, reduceDims));
                        break;
                    case 5:
                        reduced = sd.math().cosineDistance(in, in2, reduceDims);
                        name = "reduced";
                        expOut = Nd4j.getExecutioner().exec(new CosineDistance(a, b, null, false, reduceDims));
                        break;
                    default:
                        throw new RuntimeException();
                }
//                System.out.println(i + " - end");


                long[] expShape;
                if (Arrays.equals(new int[]{0}, reduceDims)) {
                    expShape = new long[]{4, 5};
                } else if (Arrays.equals(new int[]{1}, reduceDims)) {
                    expShape = new long[]{3, 5};
                } else if (Arrays.equals(new int[]{2}, reduceDims)) {
                    expShape = new long[]{3, 4};
                } else if (Arrays.equals(new int[]{Integer.MAX_VALUE}, reduceDims)) {
                    expShape = new long[]{};
                } else if (Arrays.equals(new int[]{0, 1}, reduceDims)) {
                    expShape = new long[]{5};
                } else if (Arrays.equals(new int[]{0, 2}, reduceDims)) {
                    expShape = new long[]{4};
                } else if (Arrays.equals(new int[]{1, 2}, reduceDims)) {
                    expShape = new long[]{3};
                } else if (Arrays.equals(new int[]{0, 1, 2}, reduceDims)) {
                    expShape = new long[]{};
                } else {
                    throw new RuntimeException();
                }

                String msg = name + " - dims=" + Arrays.toString(reduceDims);

                INDArray out = reduced.eval();

                log.info(msg + " - expected shape: " + Arrays.toString(expShape) + ", out=" + Arrays.toString(out.shape())
                        + ", outExp=" + Arrays.toString(expOut.shape()));

                assertArrayEquals(msg, expShape, out.shape());
                assertArrayEquals(msg, expShape, expOut.shape());

                assertEquals(msg, expOut, out);
            }
        }
    }

    @Test
    public void testReductionsBackwards() {
//        for (int i = 0; i < 7; i++) {
        int i=5;
        {

            SameDiff sd = SameDiff.create();

            int nOut = 4;
            int minibatch = 3;
            SDVariable input = sd.var("in", DataType.DOUBLE, new long[]{minibatch, nOut});
            SDVariable label = sd.var("label", DataType.DOUBLE, new long[]{minibatch, nOut});

            SDVariable diff = input.sub(label);
            SDVariable sqDiff = diff.mul(diff);
            SDVariable msePerEx = sd.mean("msePerEx", sqDiff, 1);

            SDVariable loss;    //Scalar value
            String name;
            switch (i) {
                case 0:
                    loss = sd.mean("loss", msePerEx, 0);
                    name = "mean";
                    break;
                case 1:
                    loss = sd.sum("loss", msePerEx, 0);
                    name = "sum";
                    break;
                case 2:
                    loss = sd.standardDeviation("loss", msePerEx, true, 0);
                    name = "stdev";
                    break;
                case 3:
                    loss = sd.min("loss", msePerEx, 0);
                    name = "min";
                    break;
                case 4:
                    loss = sd.max("loss", msePerEx, 0);
                    name = "max";
                    break;
                case 5:
                    loss = sd.variance("loss", msePerEx, true, 0);
                    name = "variance";
                    break;
                case 6:
                    loss = sd.prod("loss", msePerEx, 0);
                    name = "prod";
                    break;
                default:
                    throw new RuntimeException();
            }


            String msg = "test: " + i + " - " + name;
            log.info("*** Starting test: " + msg);

            INDArray inputArr = Nd4j.rand(DataType.DOUBLE, minibatch, nOut);
            INDArray labelArr = Nd4j.rand(DataType.DOUBLE, minibatch, nOut);

            sd.associateArrayWithVariable(inputArr, input);
            sd.associateArrayWithVariable(labelArr, label);

            INDArray result = loss.eval();
            assertEquals(1, result.length());

            sd.calculateGradients(Collections.emptyMap(), sd.getVariables().keySet());
        }
    }

    @Test
    public void testDotProductAttention(){
        final INDArray keys = Nd4j.rand(new int[]{10, 4, 3});
        final INDArray values = Nd4j.rand(new int[]{10, 4, 3});
        final INDArray query = Nd4j.rand(new int[]{10, 4, 1});

        final INDArray exec = Nd4j.matmul(keys, query, true, false, false)
                .divi(Math.sqrt(keys.size(1)));
        Nd4j.exec((CustomOp) new SoftMax(exec, exec, 1));
        final INDArray finalOut = Nd4j.matmul(values, exec).norm1();

        SameDiff sd = SameDiff.create();
        SDVariable sdQ = sd.var("q", query);
        SDVariable sdK = sd.var("k", keys);
        SDVariable sdV = sd.var("v", values);

        SDVariable t = sd.nn.dotProductAttention(sdQ, sdK, sdV, null, true);
        t.norm1("out");

        String err = OpValidation.validate(new TestCase(sd)
                    .expectedOutput("out", finalOut)
                    .gradientCheck(true));
        assertNull(err);
    }

    @Test
    public void testDotProductAttentionWithMask(){
        final INDArray keys = Nd4j.rand(new int[]{10, 4, 3});
        final INDArray values = Nd4j.rand(new int[]{10, 4, 3});
        final INDArray query = Nd4j.rand(new int[]{10, 4, 1});
        final INDArray mask = Nd4j.rand(10, 3).gte(0.2).castTo(DataType.DOUBLE);


        final INDArray exec = Nd4j.matmul(keys, query, true, false, false)
                .divi(Math.sqrt(keys.size(1)));
        exec.addi(mask.reshape(10, 3, 1).sub(1).muli(1e9));
        Nd4j.exec((CustomOp) new SoftMax(exec, exec, 1));
        final INDArray finalOut = Nd4j.matmul(values, exec).norm1();

        SameDiff sd = SameDiff.create();
        SDVariable sdQ = sd.var("q", query);
        SDVariable sdK = sd.var("k", keys);
        SDVariable sdV = sd.var("v", values);
        SDVariable sdMask = sd.constant("mask", mask);

        SDVariable t = sd.nn.dotProductAttention(sdQ, sdK, sdV, sdMask, true);
        t.norm1("out");

        String err = OpValidation.validate(new TestCase(sd)
                .expectedOutput("out", finalOut)
                .gradCheckSkipVariables("mask")
                .gradientCheck(true));
        assertNull(err);
    }

    @Test
    public void testDotProductAttentionMultiHeadInputWithMask(){
        final INDArray keys = Nd4j.rand(new int[]{2, 5, 4, 3});
        final INDArray values = Nd4j.rand(new int[]{2, 5, 4, 3});
        final INDArray query = Nd4j.rand(new int[]{2, 5, 4, 2});
        final INDArray mask = Nd4j.rand(2, 3).gte(0.2).castTo(DataType.DOUBLE);


        final INDArray exec = Nd4j.matmul(keys, query, true, false, false)
                .divi(Math.sqrt(keys.size(-2)));
        exec.addi(Nd4j.tile(mask.reshape(2, 1, 3, 1), 1, 5, 1, 2).sub(1).muli(1e9));
        Nd4j.exec((CustomOp) new SoftMax(exec, exec, -2));
        final INDArray finalOut = Nd4j.matmul(values, exec).norm1();

        SameDiff sd = SameDiff.create();
        SDVariable sdQ = sd.var("q", query);
        SDVariable sdK = sd.var("k", keys);
        SDVariable sdV = sd.var("v", values);
        SDVariable sdMask = sd.constant("mask", mask);


        SDVariable t = sd.nn.dotProductAttention(sdQ, sdK, sdV, sdMask, true);
        t.norm1("out");

        String err = OpValidation.validate(new TestCase(sd)
                .expectedOutput("out", finalOut)
                .gradCheckSkipVariables("mask")
                .gradientCheck(true));
        assertNull(err);
    }

    @Test
    public void testDotProductAttentionMultiHeadInput(){
        final INDArray keys = Nd4j.rand(new int[]{2, 5, 4, 3});
        final INDArray values = Nd4j.rand(new int[]{2, 5, 4, 3});
        final INDArray query = Nd4j.rand(new int[]{2, 5, 4, 1});

        final INDArray exec = Nd4j.matmul(keys, query, true, false, false)
                .divi(Math.sqrt(keys.size(-2)));
        Nd4j.exec((CustomOp) new SoftMax(exec, exec, -2));
        final INDArray finalOut = Nd4j.matmul(values, exec).norm1();

        SameDiff sd = SameDiff.create();
        SDVariable sdQ = sd.var("q", query);
        SDVariable sdK = sd.var("k", keys);
        SDVariable sdV = sd.var("v", values);

        SDVariable t = sd.nn.dotProductAttention(sdQ, sdK, sdV, null, true);
        t.norm1("out");

        String err = OpValidation.validate(new TestCase(sd)
                .expectedOutput("out", finalOut)
                .gradientCheck(true));
        assertNull(err);
    }



    @Test
    public void testMultiHeadedDotProductAttention(){
        final INDArray k = Nd4j.rand(new int[]{10, 4, 5});
        final INDArray v = Nd4j.rand(new int[]{10, 4, 5});
        final INDArray q = Nd4j.rand(new int[]{10, 4, 2});

        final INDArray Wk = Nd4j.rand(new int[]{2, 3, 4});
        final INDArray Wv = Nd4j.rand(new int[]{2, 3, 4});
        final INDArray Wq = Nd4j.rand(new int[]{2, 3, 4});
        final INDArray Wo = Nd4j.rand(new int[]{2* 3, 8});

        final INDArray kP = Nd4j.tensorMmul(k, Wk, new int[][]{{1}, {2}}).permutei(0, 2, 3, 1);
        final INDArray vP = Nd4j.tensorMmul(v, Wv, new int[][]{{1}, {2}}).permutei(0, 2, 3, 1);
        final INDArray qP = Nd4j.tensorMmul(q, Wq, new int[][]{{1}, {2}}).permutei(0, 2, 3, 1);

        final INDArray mask = Nd4j.rand(10, 5).gte(0.2).castTo(DataType.DOUBLE);

        final DynamicCustomOp dot_product_attention = DynamicCustomOp
                .builder("dot_product_attention")
                .addInputs(qP, kP, vP, mask)
                .addIntegerArguments(1, 0)
                .build();

        final INDArray[] outputs = Nd4j.exec(dot_product_attention);
        final INDArray attOut = outputs[0].permutei(0, 3, 1, 2).reshape(k.size(0), q.size(2), Wv.size(0) * Wv.size(1));

        final INDArray out = Nd4j.tensorMmul(attOut, Wo, new int[][]{{2}, {0}}).permutei(0, 2, 1);
        final INDArray finalOut = out.norm2();

        SameDiff sd = SameDiff.create();
        SDVariable sdQ = sd.var("q", q);
        SDVariable sdK = sd.var("k", k);
        SDVariable sdV = sd.var("v", v);
        SDVariable sdWq = sd.var("Wq", Wq);
        SDVariable sdWk = sd.var("Wk", Wk);
        SDVariable sdWv = sd.var("Wv", Wv);
        SDVariable sdWo = sd.var("Wo", Wo);
        SDVariable sdMask = sd.constant("mask", mask);


        SDVariable t = sd.nn.multiHeadDotProductAttention(sdQ, sdK, sdV, sdWq, sdWk, sdWv, sdWo, sdMask, true);
        t.norm2("out");

        String err = OpValidation.validate(new TestCase(sd)
                .expectedOutput("out", finalOut)
                .gradientCheck(true)
                .gradCheckSkipVariables("mask"));

        assertNull(err);
    }

    @Test
    public void testDotProductAttentionWeirdInputs(){
        final INDArray keys = Nd4j.rand(new int[]{10, 4, 3});
        final INDArray values = Nd4j.rand(new int[]{10, 4, 3});
        final INDArray query = Nd4j.rand(new int[]{10, 4, 1});
        final INDArray mask = Nd4j.rand(10, 3).gte(0.2).castTo(DataType.DOUBLE);

        final INDArray exec = Nd4j.matmul(keys, query, true, false, false)
                .divi(Math.sqrt(keys.size(1)));
        exec.addi(mask.reshape(10, 3, 1).sub(1).muli(1e9));
        Nd4j.exec((CustomOp) new SoftMax(exec, exec, 1));
        final INDArray finalOut = Nd4j.matmul(values, exec).norm1();

        for (char queryOrder : new char[]{'f', 'c'}) {
            for (char keyOrder : new char[]{'f', 'c'}) {
                for (char valueOrder : new char[]{'f', 'c'}) {
                    log.info("-*- Starting Test: query order = {}, key order = {}, value order = {}-*-", queryOrder, keyOrder, valueOrder);
                    SameDiff sd = SameDiff.create();
                    SDVariable sdQ = sd.var("q", query.dup(queryOrder));
                    SDVariable sdK = sd.var("k", keys.dup(keyOrder));
                    SDVariable sdV = sd.var("v", values.dup(valueOrder));
                    SDVariable sdMask = sd.constant("mask", mask);

                    SDVariable t = sd.nn.dotProductAttention(sdQ, sdK, sdV, sdMask, true);
                    t.norm1("out").markAsLoss();

                    String err = OpValidation.validate(new TestCase(sd)
                            .expectedOutput("out", finalOut)
                            .gradientCheck(true)
                            .gradCheckPrint(false)
                            .gradCheckSkipVariables("mask"));
                    assertNull(err);
                }
            }
        }
    }

    @Test
    public void testMultiHeadedDotProductAttentionWeirdInputs(){
        final INDArray k = Nd4j.rand(new int[]{10, 4, 5});
        final INDArray v = Nd4j.rand(new int[]{10, 4, 5});
        final INDArray q = Nd4j.rand(new int[]{10, 4, 2});

        final INDArray Wk = Nd4j.rand(new int[]{2, 3, 4});
        final INDArray Wv = Nd4j.rand(new int[]{2, 3, 4});
        final INDArray Wq = Nd4j.rand(new int[]{2, 3, 4});
        final INDArray Wo = Nd4j.rand(new int[]{2* 3, 8});

        final INDArray mask = Nd4j.rand(10, 5).gte(0.2).castTo(DataType.DOUBLE);

        final INDArray kP = Nd4j.tensorMmul(k, Wk, new int[][]{{1}, {2}}).permutei(0, 2, 3, 1);
        final INDArray vP = Nd4j.tensorMmul(v, Wv, new int[][]{{1}, {2}}).permutei(0, 2, 3, 1);
        final INDArray qP = Nd4j.tensorMmul(q, Wq, new int[][]{{1}, {2}}).permutei(0, 2, 3, 1);

        final DynamicCustomOp dot_product_attention = DynamicCustomOp
                .builder("dot_product_attention")
                .addInputs(qP, kP, vP, mask)
                .addIntegerArguments(1, 0)
                .build();

        final INDArray[] outputs = Nd4j.exec(dot_product_attention);
        final INDArray attOut = outputs[0].permutei(0, 3, 1, 2).reshape(k.size(0), q.size(2), Wv.size(0) * Wv.size(1));

        final INDArray out = Nd4j.tensorMmul(attOut, Wo, new int[][]{{2}, {0}}).permutei(0, 2, 1);
        final INDArray finalOut = out.norm2();

        for (char orderWeights: new char[]{'f', 'c'}){
            for (char orderInput: new char[]{'f', 'c'}){
                log.info("-*- Starting Test: input Order = {}, weightOrder = {} -*-", orderInput, orderWeights);


                SameDiff sd = SameDiff.create();
                SDVariable sdQ = sd.var("q", q.dup(orderInput));
                SDVariable sdK = sd.var("k", k.dup(orderInput));
                SDVariable sdV = sd.var("v", v.dup(orderInput));
                SDVariable sdWq = sd.var("Wq", Wq.dup(orderWeights));
                SDVariable sdWk = sd.var("Wk", Wk.dup(orderWeights));
                SDVariable sdWv = sd.var("Wv", Wv.dup(orderWeights));
                SDVariable sdWo = sd.var("Wo", Wo.dup(orderWeights));
                SDVariable sdMask = sd.constant("mask", mask);


                SDVariable t = sd.nn.multiHeadDotProductAttention(sdQ, sdK, sdV, sdWq, sdWk, sdWv, sdWo, sdMask, true);
                t.norm2("out");

                String err = OpValidation.validate(new TestCase(sd)
                        .expectedOutput("out", finalOut)
                        .gradientCheck(false)
                        .gradCheckSkipVariables("mask"));

                assertNull(err);
            }
        }
    }
    @Test
    public void testSufficientStatisticsOp() {
        INDArray data = Nd4j.createFromArray(new double[]{
                5.5, 0.,  0.3, 5.5,1.5, 0.,  1.3, 6.5,8.6, 0.,   0., 0.4,2.5, 1.,  0.3, 4.5,1.5, 1.,
                1.3, 1.5,3.5, 0.,  1.3, 2.5,2.6, 2.,   3., 1.4,4.5, 1.,  0.3, 0.5
        }).reshape(2,2,2,4);
        INDArray axes = Nd4j.linspace(DataType.LONG, 0, 3, 1);

        OpTestCase op = new OpTestCase(new SufficientStatistics(data, axes));

        INDArray expected1 = Nd4j.scalar(8.0);
        INDArray expected2 = Nd4j.createFromArray(new double[]{
                30.2, 5., 7.8, 22.8
        });
        INDArray expected3 = Nd4j.createFromArray(new double[]{
                154.22,   7.,    14.34, 103.62
        });

        op.expectedOutput(0, expected1);
        op.expectedOutput(1, expected2);
        op.expectedOutput(2, expected3);

        String err = OpValidation.validate(op);
        assertNull(err);
    }

    @Test
    public void testStandardDeviation() {

        for (boolean keepDims : new boolean[]{false, true}) {
            SameDiff sameDiff = SameDiff.create();

            INDArray in = Nd4j.linspace(1, 8, 8).reshape(2, 4);
            SDVariable input = sameDiff.var(in);
            INDArray expected = Nd4j.createFromArray(new double[]{
                    2,         2,         2,         2
            });

            if(keepDims){
                expected = expected.reshape(1,4);
            }

            SDVariable output = new StandardDeviation(sameDiff, input, false, keepDims, new int[]{0}).outputVariable();

            TestCase tc = new TestCase(sameDiff)
                    .gradientCheck(true)
                    .expectedOutput(output.name(), expected);

            String err = OpValidation.validate(tc);
            assertNull(err);
        }
    }

    @Test
    public void testSquaredNorm() {

        for (boolean keepDims : new boolean[]{false, true}) {
            SameDiff sameDiff = SameDiff.create();

            INDArray in = Nd4j.linspace(1, 4, 4);
            SDVariable input = sameDiff.var(in);
            INDArray expected = Nd4j.scalar(30.0000);
            if(keepDims)
                expected = expected.reshape(1);

            SDVariable output = new SquaredNorm(sameDiff, input, keepDims, new int[]{0}).outputVariable();

            TestCase tc = new TestCase(sameDiff)
                    .gradientCheck(true)
                    .expectedOutput(output.name(), expected);

            String err = OpValidation.validate(tc);
            assertNull(err);
        }
    }

    @Test
    public void testShannonEntropy() {
        OpValidationSuite.ignoreFailing();  //AB 2020/02/11 https://github.com/eclipse/deeplearning4j/issues/8695

        SameDiff sameDiff = SameDiff.create();

        INDArray in = Nd4j.linspace(1, 4, 4).castTo(DataType.DOUBLE);
        SDVariable input = sameDiff.var(in);
        INDArray expected = Nd4j.scalar(-69.68162);

        SDVariable output = new ShannonEntropy(sameDiff, input, new int[]{0}).outputVariable();

        TestCase tc = new TestCase(sameDiff)
                .gradientCheck(true)
                .expectedOutput(output.name(), expected);

        String err = OpValidation.validate(tc);
        assertNull(err);
    }

    @Test
    public void testEntropy() {

        SameDiff sameDiff = SameDiff.create();

        INDArray in = Nd4j.linspace(1, 4, 4);
        SDVariable input = sameDiff.var(in);
        double expected = -10.2273;

        SDVariable output = new Entropy(sameDiff, input, new int[]{0}).outputVariable();

        TestCase tc = new TestCase(sameDiff)
                .gradientCheck(true)
                .expectedOutput(output.name(), Nd4j.scalar(expected));

        String err = OpValidation.validate(tc);
        assertNull(err);
    }

    @Test
    public void testAMean() {

        SameDiff sameDiff = SameDiff.create();

        INDArray in = Nd4j.linspace(1, 12, 12).reshape(3, 4);
        SDVariable input = sameDiff.var(in);
        INDArray expected = Nd4j.createFromArray(new double[]{
                5.0000,    6.0000,    7.0000,    8.0000
        });

        SDVariable output = new AMean(sameDiff, input, new int[]{0}).outputVariable();

        TestCase tc = new TestCase(sameDiff)
                .gradientCheck(true)
                .expectedOutput(output.name(), expected);

        String err = OpValidation.validate(tc);
        assertNull(err);
    }

    @Test
    public void testMean() {

        SameDiff sameDiff = SameDiff.create();

        INDArray in = Nd4j.linspace(1, 12, 12).reshape(3, 4);
        SDVariable input = sameDiff.var(in);
        INDArray expected = Nd4j.createFromArray(new double[]{
                5.0000,    6.0000,    7.0000,    8.0000
        });

        SDVariable output = new Mean(sameDiff, input, false, new int[]{0}).outputVariable();

        TestCase tc = new TestCase(sameDiff)
                .gradientCheck(true)
                .expectedOutput(output.name(), expected);

        String err = OpValidation.validate(tc);
        assertNull(err);
    }

    @Test
    public void testNorm1() {

        SameDiff sameDiff = SameDiff.create();

        INDArray in = Nd4j.linspace(1, 12, 12).reshape(3, 4);
        SDVariable input = sameDiff.var(in);
        INDArray expected = Nd4j.createFromArray(new double[]{
                15.0000,   18.0000,   21.0000,   24.0000
        });

        SDVariable output = new Norm1(sameDiff, input, false, new int[]{0}).outputVariable();

        TestCase tc = new TestCase(sameDiff)
                .gradientCheck(true)
                .expectedOutput(output.name(), expected);

        String err = OpValidation.validate(tc);
        assertNull(err);
    }

    @Test
    public void testNorm2() {

        SameDiff sameDiff = SameDiff.create();

        INDArray in = Nd4j.linspace(1, 12, 12).reshape(3, 4);
        SDVariable input = sameDiff.var(in);
        INDArray expected = Nd4j.createFromArray(new double[]{
                10.3441,   11.8322,   13.3791,   14.9666
        });

        SDVariable output = new Norm2(sameDiff, input, false, new int[]{0}).outputVariable();

        TestCase tc = new TestCase(sameDiff)
                .gradientCheck(true)
                .expectedOutput(output.name(), expected);

        String err = OpValidation.validate(tc);
        assertNull(err);
    }

    @Test
    public void testNormMax() {

        SameDiff sameDiff = SameDiff.create();

        INDArray in = Nd4j.linspace(1, 12, 12).reshape(3, 4);
        SDVariable input = sameDiff.var(in);
        INDArray expected = Nd4j.createFromArray(new double[]{
                9.0000,   10.0000,   11.0000,   12.0000
        });

        SDVariable output = new NormMax(sameDiff, input, false, new int[]{0}).outputVariable();

        TestCase tc = new TestCase(sameDiff)
                .gradientCheck(true)
                .expectedOutput(output.name(), expected);

        String err = OpValidation.validate(tc);
        assertNull(err);
    }

    @Test
    public void testSoftmaxCrossEntropyWithLogitsLoss() {
        OpValidationSuite.ignoreFailing();

        SameDiff sameDiff = SameDiff.create();

        INDArray labels = Nd4j.createFromArray(new double[]{
                0,1,1,0,0,0,1,0,1,0,1,1,1,0,1,0,1,0,0,1,1,0,1,0
        }).reshape(2,3,4);

        INDArray logits = Nd4j.linspace(DataType.DOUBLE, 0.1, 0.1, 24).reshape(2,3,4);
        INDArray expected = Nd4j.createFromArray(new double[]{
                0.26328, 1.46328, 1.72656, 0.     , 0.26328, 0.     , 1.46328, 0.26328, 1.72656, 0.     , 1.72656, 1.46328
        }).reshape(3,4);

        SDVariable sdLogits = sameDiff.var("logits", logits);
        SDVariable sdLabels = sameDiff.var("labels", labels);
        SDVariable loss = sameDiff.math().abs(sdLogits);


        SDVariable output = new SoftmaxCrossEntropyWithLogitsLoss(sameDiff, sdLogits, sdLabels, 0).outputVariable();
        sameDiff.setLossVariables(output);

        TestCase tc = new TestCase(sameDiff)
                .gradientCheck(true)
                .expectedOutput(output.name(), expected);

        String err = OpValidation.validate(tc);
        assertNull(err);
    }
}
