package org.nd4j.autodiff.opvalidation;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.validation.OpTestCase;
import org.nd4j.autodiff.validation.OpValidation;
import org.nd4j.autodiff.validation.TestCase;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.ops.impl.accum.*;
import org.nd4j.linalg.api.ops.impl.accum.bp.StandardDeviationBp;
import org.nd4j.linalg.api.ops.impl.accum.distances.*;
import org.nd4j.linalg.api.ops.impl.indexaccum.IAMax;
import org.nd4j.linalg.api.ops.impl.indexaccum.IAMin;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.fail;

@Slf4j
@RunWith(Parameterized.class)
public class ReductionOpValidation extends BaseOpValidation {

    public ReductionOpValidation(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testZeroCount() {
        List<String> allFailed = new ArrayList<>();
        for( int i=0; i<2; i++ ) {
            SameDiff sd = SameDiff.create();

            INDArray ia;
            if (i == 0) {
                //Not gradient checkable for 0 and 1 values
                ia = Nd4j.create(new int[]{2, 2}, new float[]{0, 1, 0, 1});
            } else {
                ia = Nd4j.rand(2,2);
            }

            SDVariable input = sd.var("in", new int[]{2, 2});
            sd.associateArrayWithVariable(ia, input);

            SDVariable nonZero = sd.countNonZero(input);
            SDVariable zero = sd.countZero(input);

            String error = OpValidation.validate(new TestCase(sd)
                    .expectedOutput(nonZero.getVarName(), Nd4j.trueScalar(2.0))
                    .expectedOutput(zero.getVarName(), Nd4j.trueScalar(2.0))
                    .gradientCheck(i != 0)
            );
            if(error != null)
                allFailed.add(error);
        }
        assertEquals(allFailed.toString(), 0, allFailed.size());
    }


    @Test
    public void testZeroFraction() {

        List<String> allFailed = new ArrayList<>();
        for( int i=0; i<2; i++ ) {
            SameDiff sd = SameDiff.create();

            INDArray ia;
            if (i == 0) {
                //Not gradient checkable for 0 and 1 values
                ia = Nd4j.create(new int[]{2, 2}, new float[]{0, 1, 0, 1});
            } else {
                ia = Nd4j.rand(2,2);
            }

            SDVariable input = sd.var("in", new int[]{2, 2});
            sd.associateArrayWithVariable(ia, input);

            SDVariable zeroFraction = sd.zeroFraction(input);

            String error = OpValidation.validate(new TestCase(sd)
                    .expectedOutput(zeroFraction.getVarName(), Nd4j.trueScalar(0.5))
                    .gradientCheck(i != 0)
            );
            if(error != null)
                allFailed.add(error);
        }

        assertEquals(allFailed.toString(), 0, allFailed.size());
    }

    @Test
    public void testReductionGradientsSimple() {
        //Test reductions: final and only function
        Nd4j.getRandom().setSeed(12345);

        List<String> failed = new ArrayList<>();

        for (int i = 0; i < 18; i++) {

            SameDiff sd = SameDiff.create();

            int nOut = 4;
            int minibatch = 10;
            SDVariable input = sd.var("in", new int[]{-1, nOut});
            INDArray inputArr = Nd4j.randn(minibatch, nOut).muli(100);
            long length = nOut * minibatch;

            SDVariable loss;
            String name;
            TestCase tc = new TestCase(sd);
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
                    loss = sd.countNonZero("loss", input);
                    name = "countNonZero";
                    tc.expectedOutput("loss", Nd4j.trueScalar(inputArr.length()));
                    break;
                case 11:
                    loss = sd.countZero("loss", input);
                    name = "countZero";
                    tc.expectedOutput("loss", Nd4j.trueScalar(0));
                    break;
                case 12:
                    loss = sd.amax("loss", input);
                    name = "amax";
                    tc.expectedOutput("loss", inputArr.amax());
                    break;
                case 13:
                    loss = sd.amin("loss", input);
                    name = "amin";
                    tc.expectedOutput("loss", inputArr.amin());
                    break;
                case 14:
                    loss = sd.asum("loss", input);
                    name = "asum";
                    tc.expectedOutput("loss", Nd4j.getExecutioner().exec(new ASum(inputArr.dup())).z());
                    break;
                case 15:
                    loss = sd.amean("loss", input);
                    name = "amean";
                    tc.expectedOutput("loss", Nd4j.getExecutioner().exec(new AMean(inputArr.dup())).z());
                    break;
                case 16:
                    loss = sd.entropy("loss", input);
                    name = "entropy";
                    inputArr = Nd4j.linspace(0.01, 0.99, length).reshape('c', minibatch, nOut);
                    tc.expected("loss", inputArr.mul(Transforms.log(inputArr, true)).sum(Integer.MAX_VALUE).negi());
                    break;
                case 17:
                    inputArr = Nd4j.rand(minibatch, nOut);
                    name = "logsumexp";
                    loss = sd.logSumExp("loss", input);
                    INDArray expArr = Transforms.exp(inputArr);
                    double sum = expArr.sumNumber().doubleValue();
                    tc.expected("loss", Nd4j.create(new double[]{Math.log(sum)}));
                    break;
                default:
                    throw new RuntimeException();
            }


            String msg = "test: " + i + " - " + name;
            log.info("*** Starting test: " + msg);

            sd.associateArrayWithVariable(inputArr, input);


            tc.testName(msg);
            String error = OpValidation.validate(tc, true);
            if(error != null)
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
                SDVariable input = sd.var("in", new int[]{-1, nOut});
                SDVariable label = sd.var("label", new int[]{-1, nOut});

                SDVariable diff = input.sub(label);
                SDVariable sqDiff = diff.mul(diff);
                SDVariable msePerEx = sd.mean("msePerEx", sqDiff, 1);

                SDVariable loss;
                String name;
                TestCase tc = new TestCase(sd);
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
                        loss = sd.countNonZero("loss", input, dim);
                        name = "countNonZero";
                        break;
                    case 11:
                        loss = sd.countZero("loss", input, dim);
                        name = "countZero";
                        break;
                    case 12:
                        loss = sd.amax("loss", input, dim);
                        name = "amax";
                        break;
                    case 13:
                        loss = sd.amin("loss", input, dim);
                        name = "amin";
                        break;
                    case 14:
                        loss = sd.asum("loss", input, dim);
                        name = "asum";
                        break;
                    case 15:
                        loss = sd.amean("loss", input, dim);
                        name = "amean";
                        break;
                    default:
                        throw new RuntimeException();
                }


                String msg = "(test " + i + " - " + name + ", dimension=" + dim + ")";
                log.info("*** Starting test: " + msg);

                INDArray inputArr = Nd4j.randn(minibatch, nOut).muli(100);
                INDArray labelArr = Nd4j.randn(minibatch, nOut).muli(100);

                sd.associateArrayWithVariable(inputArr, input);
                sd.associateArrayWithVariable(labelArr, label);

                String error = OpValidation.validate(tc);
                if(error != null){
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

                int[] outShape;
                switch (reduceDim) {
                    case 0:
                        outShape = new int[]{d1, d2};
                        break;
                    case 1:
                        outShape = new int[]{d0, d2};
                        break;
                    case 2:
                        outShape = new int[]{d0, d1};
                        break;
                    default:
                        throw new RuntimeException();
                }

                SameDiff sd = SameDiff.create();
                sd.setLogExecution(false);


                SDVariable in = sd.var("in", new int[]{-1, d1, d2});
                SDVariable label = sd.var("label", outShape);
                SDVariable second = in.mul(2);

                double maxRelError = 1e-5;
                double minAbsError = 1e-4;
                INDArray inputArr = Nd4j.randn(new int[]{d0, d1, d2}).muli(1000);
                INDArray labelArr = Nd4j.randn(outShape).muli(1000);
                SDVariable reduced;
                String name;
                TestCase tc = new TestCase(sd);
                switch (i) {
                    case 0:
                        reduced = sd.mean("reduced", second, reduceDim);
                        name = "mean";
                        break;
                    case 1:
                        reduced = sd.sum("reduced", second, reduceDim);
                        name = "sum";
                        break;
                    case 2:
                        reduced = sd.standardDeviation("reduced", second, true, reduceDim);
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
                        minAbsError = 1;        //Most gradients ane in the range 1k to >100k
                        inputArr.divi(10);
                        labelArr.divi(100);
                        BooleanIndexing.replaceWhere(inputArr, Nd4j.rand(inputArr.shape()).muli(100).addi(100), Conditions.absLessThan(1.0));
                        reduced = sd.variance("reduced", second, true, reduceDim);
                        name = "variance";
                        break;
                    case 6:
                        inputArr.divi(1000);
                        labelArr.divi(1000);
                        reduced = sd.prod("reduced", second, reduceDim);
                        name = "prod";
                        break;
                    case 7:
                        reduced = sd.norm1("reduced", second, reduceDim);
                        name = "norm1";
                        break;
                    case 8:
                        reduced = sd.norm2("reduced", second, reduceDim);
                        name = "norm2";
                        break;
                    case 9:
                        inputArr = Nd4j.rand(new int[]{d0, d1, d2});
                        labelArr = Nd4j.rand(outShape);
                        reduced = sd.normmax("reduced", second, reduceDim);
                        name = "normmax";
                        break;
                    case 10:
                        reduced = sd.argmax("reduced", second, reduceDim);
                        name = "argmax";
                        break;
                    case 11:
                        reduced = sd.argmin("reduced", second, reduceDim);
                        name = "argmin";
                        break;
                    case 12:
                        reduced = sd.countNonZero("loss", second, reduceDim);
                        name = "countNonZero";
                        break;
                    case 13:
                        reduced = sd.countZero("loss", second, reduceDim);
                        name = "countZero";
                        break;
                    case 14:
                        reduced = sd.amax("loss", second, reduceDim);
                        name = "amax";
                        break;
                    case 15:
                        reduced = sd.amin("loss", second, reduceDim);
                        name = "amin";
                        break;
                    case 16:
                        reduced = sd.asum("loss", second, reduceDim);
                        name = "asum";
                        break;
                    case 17:
                        reduced = sd.amean("loss", second, reduceDim);
                        name = "amean";
                        break;
                    default:
                        throw new RuntimeException();
                }

                SDVariable add = reduced.add(1.0);

                SDVariable diff = label.sub(add);
                SDVariable sqDiff = diff.mul(diff);
                SDVariable mseLoss = sd.mean("loss", sqDiff);


                String msg = "(test " + i + " - " + name + ", dimension=" + reduceDim + ")";
                log.info("*** Starting test: " + msg);

                sd.associateArrayWithVariable(inputArr, in);
                sd.associateArrayWithVariable(labelArr, label);

                String error = OpValidation.validate(tc);
                if(error != null){
                    failed.add(name);
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


                SDVariable in = sd.var("in", new int[]{-1, d1, d2});
                SDVariable in2 = sd.var("in2", new int[]{-1, d1, d2});

                INDArray inArr = Nd4j.randn(new int[]{d0, d1, d2}).muli(100);
                INDArray in2Arr = Nd4j.randn(inArr.shape()).muli(100);

                INDArray exp;
                SDVariable reduced;
                String name;
                TestCase tc = new TestCase(sd);
                switch (i) {
                    case 0:
                        reduced = sd.manhattanDistance(in, in2, reduceDims);
                        name = "manhattan";
                        exp = Nd4j.getExecutioner().exec(new ManhattanDistance(inArr, in2Arr), reduceDims);
                        break;
                    case 1:
                        reduced = sd.euclideanDistance(in, in2, reduceDims);
                        name = "euclidean";
                        exp = Nd4j.getExecutioner().exec(new EuclideanDistance(inArr, in2Arr), reduceDims);
                        break;
                    case 2:
                        reduced = sd.cosineSimilarity(in, in2, reduceDims);
                        name = "cosine";
                        exp = Nd4j.getExecutioner().exec(new CosineSimilarity(inArr, in2Arr), reduceDims);
                        break;
                    case 3:
                        reduced = sd.cosineDistance(in, in2, reduceDims);
                        name = "cosinedistance";
                        exp = Nd4j.getExecutioner().exec(new CosineDistance(inArr, in2Arr), reduceDims);
                        break;
                    case 4:
                        reduced = sd.hammingDistance(in, in2, reduceDims);
                        name = "hamming";
                        exp = Nd4j.getExecutioner().exec(new HammingDistance(inArr, in2Arr), reduceDims);
                        break;
                    case 5:
                        name = "jaccard";
                        reduced = sd.jaccardDistance(name, in, in2, reduceDims);
                        inArr.divi(100).addi(0.1);
                        in2Arr.divi(100).addi(0.1);
                        exp = Nd4j.getExecutioner().exec(new JaccardDistance(inArr, in2Arr), reduceDims);
                        break;
                    case 6:
                        name = "dot";
                        reduced = sd.dot(name, in, in2, reduceDims);
                        exp = Nd4j.getExecutioner().exec(new Dot(inArr, in2Arr), reduceDims);
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

                String error = OpValidation.validate(tc, true);
                if(error != null){
                    failed.add(msg);
                }
            }
        }

        assertEquals("Failed: " + failed, 0, failed.size());
    }

    @Test
    public void testMoments(){

        for( int[] axes : new int[][]{{0}, {1}, {0,1}}) {
            INDArray input = Nd4j.linspace(1, 12, 12).reshape(3, 4);

            SameDiff sd = SameDiff.create();
            SDVariable in = sd.var("in", input);

            SDVariable[] moments = sd.moments(in, axes);
            INDArray expMean = input.mean(axes);
            INDArray expStdev = input.std(true, axes);

            SDVariable loss = moments[0].add(moments[1]).std(true);

            String msg = Arrays.toString(axes);

            TestCase tc = new TestCase(sd)
                    .testName(msg)
                    .expected(moments[0], expMean)
                    .expected(moments[1], expStdev);

            String err = OpValidation.validate(tc);
            assertNull(err);
        }
    }

    @Test
    public void testMomentsOp(){
        int[] axes = new int[]{0};
        INDArray input = Nd4j.linspace(1, 12, 12).reshape(3, 4);

        INDArray outMean = Nd4j.createUninitialized(new long[]{4});
        INDArray outStd = Nd4j.createUninitialized(new long[]{4});

        OpTestCase tc = new OpTestCase(DynamicCustomOp.builder("moments")
                .addOutputs(outMean, outStd)
                .addInputs(input)
                .addIntegerArguments(axes)
                .build());

        tc.expectedOutput(0, input.mean(axes).reshape(4));
        tc.expectedOutput(1, input.std(axes).reshape(4));

        String err = OpValidation.validate(tc);
        assertNull(err);
    }

    @Test
    public void testNormalizeMomentsOp(){

        INDArray data = Nd4j.linspace(1, 100, 100).reshape(10,10);
        INDArray ssSum = data.sum(0);
        INDArray ssSqSum = data.mul(data).sum(0);

        INDArray meanExp = data.mean(0);
        INDArray varExp = data.var(true, 0);

        INDArray mean = Nd4j.createUninitialized(meanExp.shape());
        INDArray var = Nd4j.createUninitialized(varExp.shape());

        OpTestCase op = new OpTestCase(new NormalizeMoments(Nd4j.trueScalar(10), ssSum, ssSqSum, mean, var));
        op.expectedOutput(0, meanExp);
        op.expectedOutput(1, varExp);

        String err = OpValidation.validate(op);
        assertNull(err);
    }

    @Test
    public void testAllAny(){

        INDArray allZeros = Nd4j.create(3,4);
        INDArray allOnes = Nd4j.ones(3,4);
        INDArray mixed = Nd4j.zeros(3,4);
        mixed.getRow(1).assign(1.0);

        INDArray[] in = new INDArray[]{allZeros, allOnes, mixed};
        double[] expAll = new double[]{0,1,0};
        double[] expAny = new double[]{0,1,1};

        for( int i=0; i<3; i++ ){
            SameDiff sd = SameDiff.create();

            SDVariable s = sd.var("in", in[i]);
            SDVariable all = sd.f().all(s);
            SDVariable any = sd.f().any(s);

            String err = OpValidation.validate(new TestCase(sd)
                    .gradientCheck(false)
                    .expected(all, Nd4j.create(new double[]{expAll[i]}))
                    .expected(any, Nd4j.create(new double[]{expAny[i]})));

            assertNull(err);
        }
    }

    @Test
    public void testIndexAccum(){
        fail(); //JVM crash: https://github.com/deeplearning4j/deeplearning4j/issues/5582

        List<String> failed = new ArrayList<>();
        List<int[]> dims = Arrays.asList(new int[]{0}, new int[]{1}, new int[]{0,1}, new int[0]);

        INDArray in = Nd4j.rand(3,4);

        for(int[] d : dims){
            for( int i=0; i<4; i++ ){

                int[] dim = d.length == 0 ? null : d;

                SameDiff sd = SameDiff.create();
                SDVariable s = sd.var("in", in);
                SDVariable reduce;

                String name;
                INDArray exp;
                switch (i){
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
                        reduce = sd.iamax(s,dim);
                        exp = Nd4j.getExecutioner().exec(new IAMax(in.dup()), dim);
                        name = "iamax";
                        break;
                    case 3:
                        reduce = sd.iamin(s,dim);
                        exp = Nd4j.getExecutioner().exec(new IAMin(in.dup()), dim);
                        name = "iamin";
                        break;
                    default:
                        throw new RuntimeException();
                }

                SDVariable loss;
                if(dim == null || dim.length == 2){
                    loss = reduce.mean();
                } else {
                    loss = reduce.std(true);
                }

                TestCase tc = new TestCase(sd)
                        .expected(reduce, exp)
                        .testName(name + " - " + (dim == null ? null : Arrays.toString(dim)));

                log.info("Starting: {}", tc.testName());
                String err = OpValidation.validate(tc, true);
                if(err != null){
                    failed.add(err);
                }
            }
        }

        assertEquals(failed.toString(), 0, failed.size());
    }
}
