package org.nd4j.autodiff.opvalidation;

import lombok.extern.slf4j.Slf4j;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.validation.OpValidation;
import org.nd4j.autodiff.validation.TestCase;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarFMod;
import org.nd4j.linalg.api.ops.impl.transforms.*;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.GreaterThanOrEqual;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.LessThanOrEqual;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.OldMax;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.OldMin;
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.fail;

@Slf4j
public class TransformOpValidation {

    private DataBuffer.Type initialType;

    @Before
    public void before() throws Exception {
        Nd4j.create(1);
        initialType = Nd4j.dataType();

        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
        Nd4j.getRandom().setSeed(123);
    }

    @After
    public void after() throws Exception {
        Nd4j.setDataType(initialType);
    }


    @After
    public void tearDown() throws Exception {
        NativeOpsHolder.getInstance().getDeviceNativeOps().enableDebugMode(false);
        NativeOpsHolder.getInstance().getDeviceNativeOps().enableVerboseMode(false);
    }

    @Test
    public void testScalarOps() {
        int d0 = 2;
        int d1 = 3;
        int d2 = 4;

        int n = d0 * d1 * d2;

        List<String> failed = new ArrayList<>();

        for( int i=0; i<7; i++ ) {
            for (char inOrder : new char[]{'c', 'f'}) {
                SameDiff sd = SameDiff.create();

                INDArray inArr = Nd4j.linspace(1, n, n).reshape(inOrder, d0, d1, d2);
                SDVariable in = sd.var("in", inArr);
                TestCase tc = new TestCase(sd).gradientCheck(true);

                SDVariable out;
                String msg;
                switch (i){
                    case 0:
                        out = in.mul(2);
                        tc.expectedOutput(out.getVarName(), inArr.mul(2));
                        msg = "mul - " + inOrder;
                        break;
                    case 1:
                        out = in.div(2);
                        tc.expectedOutput(out.getVarName(), inArr.div(2));
                        msg = "div - " + inOrder;
                        break;
                    case 2:
                        out = in.add(2);
                        tc.expectedOutput(out.getVarName(), inArr.add(2));
                        msg = "add - " + inOrder;
                        break;
                    case 3:
                        out = in.sub(2);
                        tc.expectedOutput(out.getVarName(), inArr.sub(2));
                        msg = "sub - " + inOrder;
                        break;
                    case 4:
                        out = in.rdiv(2);
                        tc.expectedOutput(out.getVarName(), inArr.rdiv(2));
                        msg = "rdiv - " + inOrder;
                        break;
                    case 5:
                        out = in.rsub(2);
                        tc.expectedOutput(out.getVarName(), inArr.rsub(2));
                        msg = "rsub - " + inOrder;
                        break;
                    case 6:
                        out = sd.pow(2);
                        tc.expectedOutput(out.getVarName(), Transforms.pow(inArr, 2));
                        msg = "mul - " + inOrder;
                        break;
                    default:
                        throw new RuntimeException();
                }

                SDVariable loss = sd.standardDeviation(out, true);

                String err = OpValidation.validate(tc);
                if(err != null){
                    failed.add(err);
                }
            }
        }
        assertEquals(failed.toString(), 0, failed.size());
    }

    @Test
    public void testCross() {
        INDArray a = Nd4j.create(new float[]{4, 2, 1}, new int[]{1, 3});
        INDArray b = Nd4j.create(new float[]{1, 3, 4}, new int[]{1, 3});

        INDArray expOut = Nd4j.create(1, 3);

        DynamicCustomOp op = DynamicCustomOp.builder("cross").addInputs(a, b).addOutputs(expOut).build();
        Nd4j.getExecutioner().exec(op);

        SameDiff sd = SameDiff.create();

        SDVariable sdA = sd.var("a", expOut.shape());
        SDVariable sdB = sd.var("b", expOut.shape());


        sd.associateArrayWithVariable(a, sdA);
        sd.associateArrayWithVariable(b, sdB);

        SDVariable t = sd.cross("cross", sdA, sdB);
        SDVariable loss = sd.mean("loss", t);

        String err = OpValidation.validate(new TestCase(sd)
                        .expectedOutput("cross", expOut)
                        .gradientCheck(true));
        assertNull(err, err);
    }

    @Test
    public void testSpaceToDepth() {
        Nd4j.getRandom().setSeed(1337);

        int miniBatch = 128;
        int blockSize = 4;
        String dataFormat = "NHWC";
        int isNHWC = dataFormat.equals("NHWC") ? 1 : 0;
        int[] inputShape = new int[]{miniBatch, 2 * blockSize, 2 * blockSize, 1};

        INDArray input = Nd4j.randn(inputShape);
        SameDiff sd = SameDiff.create();
        SDVariable sdInput = sd.var("in", inputShape);

        INDArray expOut = Nd4j.create(miniBatch, 2, 2, blockSize * blockSize);
        DynamicCustomOp op = DynamicCustomOp.builder("space_to_depth")
                .addInputs(input)
                .addIntegerArguments(blockSize, isNHWC)
                .addOutputs(expOut).build();
        Nd4j.getExecutioner().exec(op);

        sd.associateArrayWithVariable(input, sdInput);

        SDVariable t = sd.spaceToDepth("std", sdInput, blockSize, dataFormat);
        SDVariable loss = sd.mean("loss", t);

        String err = OpValidation.validate(new TestCase(sd)
                .expectedOutput("std", expOut)
                .gradientCheck(true));
        assertNull(err, err);
    }

    @Test
    public void testDepthToSpace() {
        Nd4j.getRandom().setSeed(1337);

        int miniBatch = 128;
        int blockSize = 4;
        String dataFormat = "NHWC";
        int isNHWC = dataFormat.equals("NHWC") ? 1 : 0;
        int[] inputShape = new int[]{miniBatch, 2, 2, blockSize * blockSize};

        INDArray input = Nd4j.randn(inputShape);
        SameDiff sd = SameDiff.create();
        SDVariable sdInput = sd.var("in", inputShape);

        INDArray expOut = Nd4j.create(miniBatch, 2 * blockSize, 2 * blockSize, 1);
        DynamicCustomOp op = DynamicCustomOp.builder("depth_to_space")
                .addInputs(input)
                .addIntegerArguments(blockSize, isNHWC)
                .addOutputs(expOut).build();
        Nd4j.getExecutioner().exec(op);

        sd.associateArrayWithVariable(input, sdInput);

        SDVariable t = sd.depthToSpace("dts", sdInput, blockSize, dataFormat);
        SDVariable loss = sd.mean("loss", t);

        String err = OpValidation.validate(new TestCase(sd)
                .expectedOutput("dts", expOut)
                .gradientCheck(true));
        assertNull(err, err);
    }

    @Test
    public void testBatchToSpace() {
        Nd4j.getRandom().setSeed(1337);

        int miniBatch = 4;
        int[] inputShape = new int[]{miniBatch, 1, 1, 1};

        int M = 2;
        int[] blockShape = new int[]{M, 1};
        int[] cropShape = new int[]{M, 2};

        INDArray input = Nd4j.randn(inputShape);
        INDArray blocks = Nd4j.create(new float[]{2, 2}, blockShape);
        INDArray crops = Nd4j.create(new float[]{0, 0, 0, 0}, cropShape);

        SameDiff sd = SameDiff.create();

        SDVariable sdInput = sd.var("in", inputShape);

        INDArray expOut = Nd4j.create(1, 2, 2, 1);
        DynamicCustomOp op = DynamicCustomOp.builder("batch_to_space")
                .addInputs(input, blocks, crops)
                .addOutputs(expOut).build();
        Nd4j.getExecutioner().exec(op);

        sd.associateArrayWithVariable(input, sdInput);

        SDVariable t = sd.batchToSpace("bts", sdInput, new int[]{2, 2}, new int[][]{{0, 0}, {0, 0}});
        SDVariable loss = sd.mean("loss", t);

        String err = OpValidation.validate(new TestCase(sd)
                .expectedOutput("bts", expOut)
                .gradientCheck(true));
        assertNull(err, err);
    }

    @Test
    public void testSpaceToBatch() {
        Nd4j.getRandom().setSeed(7331);

        int miniBatch = 4;
        int[] inputShape = new int[]{1, 2, 2, 1};

        int M = 2;
        int[] blockShape = new int[]{M, 1};
        int[] paddingShape = new int[]{M, 2};

        INDArray input = Nd4j.randn(inputShape);
        INDArray blocks = Nd4j.create(new float[]{2, 2}, blockShape);
        INDArray padding = Nd4j.create(new float[]{0, 0, 0, 0}, paddingShape);

        SameDiff sd = SameDiff.create();

        SDVariable sdInput = sd.var("in", inputShape);

        INDArray expOut = Nd4j.create(miniBatch, 1, 1, 1);
        DynamicCustomOp op = DynamicCustomOp.builder("space_to_batch")
                .addInputs(input, blocks, padding)
                .addOutputs(expOut).build();
        Nd4j.getExecutioner().exec(op);

        sd.associateArrayWithVariable(input, sdInput);

        SDVariable t = sd.spaceToBatch("stb", sdInput, new int[]{2, 2}, new int[][]{{0, 0}, {0, 0}});
        SDVariable loss = sd.mean("loss", t);

        String err = OpValidation.validate(new TestCase(sd)
                .expectedOutput("stb", expOut)
                .gradientCheck(true));
        assertNull(err, err);
    }

    @Test
    public void testDynamicPartition() {
        SameDiff sd = SameDiff.create();

        INDArray ia = Nd4j.create(new float[]{4, 3, 5, 7, 8, 0}, new int[]{1, 6});
        INDArray partitions = Nd4j.create(new float[]{1, 0, 1, 0, 0, 1});
        int numPartitions = 2;

        SDVariable in = sd.var("in", new int[]{1, 6});
        SDVariable sdPartitions = sd.var("partitions", new int[]{1, 6});

        INDArray expOut1 = Nd4j.create(3L);
        INDArray expOut2 = Nd4j.create(3L);
        INDArray[] expOut = new INDArray[]{expOut1, expOut2};

        DynamicCustomOp dynamicPartition = DynamicCustomOp.builder("dynamic_partition")
                .addInputs(ia, partitions)
                .addIntegerArguments(numPartitions)
                .addOutputs(expOut1, expOut2).build();
        Nd4j.getExecutioner().exec(dynamicPartition);

        SDVariable[] parts = sd.dynamicPartition(new String[]{"dp0", "dp1"}, in, sdPartitions, numPartitions);

        // merge the output partitions together again, to retrieve a single
        // tensor and finally a scalar.
        SDVariable t = sd.mergeAdd(parts);
        SDVariable loss = sd.mean("loss", t);

        sd.associateArrayWithVariable(ia, in);

        String err = OpValidation.validate(new TestCase(sd)
                .expectedOutput("dp0", expOut[0])
                .expectedOutput("dp1", expOut[1])
                .gradientCheck(true));
        assertNull(err, err);
    }

    @Test
    public void testDynamicStitch() {
        SameDiff sd = SameDiff.create();

        INDArray ia = Nd4j.create(new float[]{5, 1, 3}, new int[]{1, 3});
        INDArray ib = Nd4j.create(new float[]{7, 2, 4}, new int[]{1, 3});
        INDArray indexA = Nd4j.create(new float[]{0, 1, 4}, new int[]{1, 3});
        INDArray indexB = Nd4j.create(new float[]{2, 3, 5}, new int[]{1, 3});

        INDArray expOut = Nd4j.create(new long[]{6});

        DynamicCustomOp dynamicStitch = DynamicCustomOp.builder("dynamic_stitch")
                .addInputs(indexA, indexB, ia, ib)
                .addOutputs(expOut).build();
        Nd4j.getExecutioner().exec(dynamicStitch);

        INDArray expOut2 = Nd4j.create(new double[]{5,1,7,2,3,4});
        assertEquals(expOut2, expOut);

        SDVariable in1 = sd.var("in1", new int[]{1, 3});
        SDVariable in2 = sd.var("in2", new int[]{1, 3});

        SDVariable index1 = sd.var("index1", new int[]{1, 3});
        SDVariable index2 = sd.var("index2", new int[]{1, 3});

        sd.associateArrayWithVariable(ia, in1);
        sd.associateArrayWithVariable(ib, in2);
        sd.associateArrayWithVariable(indexA, index1);
        sd.associateArrayWithVariable(indexB, index2);

        SDVariable t = sd.dynamicStitch("ds", new SDVariable[]{index1, index2}, new SDVariable[]{in1, in2});
        SDVariable loss = sd.mean("loss", t);

        String err = OpValidation.validate(new TestCase(sd)
                .expectedOutput("ds", expOut)
                .gradientCheck(true));
        assertNull(err, err);
    }

    @Test
    public void testDiag() {
        SameDiff sd = SameDiff.create();

        INDArray ia = Nd4j.create(new float[]{4, 2});
        SDVariable in = sd.var("in", new int[]{1, 2});
        INDArray expOut = Nd4j.create(new int[]{2, 2});
        DynamicCustomOp diag = DynamicCustomOp.builder("diag").addInputs(ia).addOutputs(expOut).build();
        Nd4j.getExecutioner().exec(diag);
        SDVariable t = sd.diag("diag", in);

        SDVariable loss = sd.standardDeviation("loss", t,false,0, 1);

        sd.associateArrayWithVariable(ia, in);

        String err = OpValidation.validate(new TestCase(sd)
                .expectedOutput("diag", expOut)
                .gradientCheck(true));
        assertNull(err, err);
    }

    @Test
    public void testDiagPart() {
        SameDiff sd = SameDiff.create();

        INDArray input = Nd4j.linspace(1,16,16).reshape(4,4);
        INDArray expOut = Nd4j.create(new float[]{1, 6, 11, 16});

        SDVariable in = sd.var("in", input);
        SDVariable t = sd.diagPart("dp", in);

        SDVariable loss = sd.standardDeviation("loss", t, true, 0, 1);

        String err = OpValidation.validate(new TestCase(sd)
                .expectedOutput("dp", expOut)
                .gradientCheck(true));
        assertNull(err, err);
    }

    @Test
    public void testEye(){

        int[] rows = new int[]{3,3,3,3};
        int[] cols = new int[]{3,2,2,2};
        int[][] batch = new int[][]{null, null, {4}, {3,3}};
        INDArray[] expOut = new INDArray[4];

        expOut[0] = Nd4j.eye(3);
        expOut[1] = Nd4j.create(new double[][]{{1,0,0},{0,1,0}});
        expOut[2] = Nd4j.create(4,3,2);
        for( int i=0; i<4; i++ ){
            expOut[2].get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all()).assign(expOut[1]);
        }
        expOut[3] = Nd4j.create(3,3,3,2);
        for( int i=0; i<3; i++ ){
            for( int j=0; j<3; j++ ) {
                expOut[3].get(NDArrayIndex.point(i), NDArrayIndex.point(j), NDArrayIndex.all(), NDArrayIndex.all()).assign(expOut[1]);
            }
        }


        for(int i=0; i<3; i++ ) {
            SameDiff sd = SameDiff.create();
            SDVariable eye = sd.eye("e", rows[i], cols[i], batch[i]);

            SDVariable loss = sd.standardDeviation("loss", eye, true);

            String err = OpValidation.validate(new TestCase(sd)
                    .expectedOutput("e", expOut[i])
                    .gradientCheck(true));
            assertNull(err, err);
        }

    }

    @Test
    public void testTransforms() {
        //Test transforms (non-pairwise)
        Nd4j.getRandom().setSeed(12345);

        List<String> allSkipped = new ArrayList<>();

        List<String> allFailed = new ArrayList<>();
        for (int i = 0; i < 72; i++) {

            SameDiff sd = SameDiff.create();

            int nOut = 4;
            int minibatch = 5;
            SDVariable in = sd.var("in", new int[]{-1, nOut});

            INDArray ia = Nd4j.randn(minibatch, nOut);

            int dim;
            SDVariable t;
            TestCase tc = new TestCase(sd);
            boolean stdevLoss = false;
            switch (i) {
                case 0:
                    t = in.add(5.0);
                    tc.expectedOutput(t.getVarName(), ia.add(5.0));
                    break;
                case 1:
                    t = in.sub(5.0);
                    tc.expectedOutput(t.getVarName(), ia.sub(5.0));
                    break;
                case 2:
                    t = in.mul(2.5);
                    tc.expectedOutput(t.getVarName(), ia.mul(2.5));
                    break;
                case 3:
                    t = in.div(4.0);
                    tc.expectedOutput(t.getVarName(), ia.div(4.0));
                    break;
                case 4:
                    t = in.rsub(5.0);
                    tc.expectedOutput(t.getVarName(), ia.rsub(5.0));
                    break;
                case 5:
                    t = in.rdiv(1.0);
                    tc.expectedOutput(t.getVarName(), ia.rdiv(1.0));
                    break;
                case 6:
                    t = sd.pow(in, 2.5);
                    ia = Nd4j.rand(minibatch, nOut);
                    tc.expectedOutput(t.getVarName(), Transforms.pow(ia, 2.5, true));
                    break;
                case 7:
                    t = sd.sigmoid(in);
                    ia = Nd4j.rand(minibatch, nOut).muli(2).subi(1.0);
                    tc.expectedOutput(t.getVarName(), Transforms.sigmoid(ia, true));
                    break;
                case 8:
                    t = sd.tanh(in);
                    ia = Nd4j.rand(minibatch, nOut).muli(2).subi(1.0);
                    tc.expectedOutput(t.getVarName(), Transforms.tanh(ia, true));
                    break;
                case 9:
                    t = sd.tan(in);
                    ia = Nd4j.rand(minibatch, nOut);
                    tc.expectedOutput(t.getVarName(), Transforms.tan(ia));
                    break;
                case 10:
                    t = sd.cos(in);
                    tc.expectedOutput(t.getVarName(), Transforms.cos(ia, true));
                    break;
                case 11:
                    t = sd.sin(in);
                    tc.expectedOutput(t.getVarName(), Transforms.sin(ia, true));
                    break;
                case 12:
                    t = sd.softplus(in);
                    tc.expectedOutput(t.getVarName(), Transforms.softPlus(ia, true));
                    break;
                case 13:
                    t = sd.log(in);
                    ia = Nd4j.rand(minibatch, nOut);
                    tc.expectedOutput(t.getVarName(), Transforms.log(ia, true));
                    break;
                case 14:
                    t = sd.neg(in);
                    tc.expectedOutput(t.getVarName(), ia.neg());
                    break;
                case 15:
                    t = sd.acos(in);
                    ia = Nd4j.rand(minibatch, nOut).muli(1.8).subi(0.9);
                    tc.expectedOutput(t.getVarName(), Transforms.acos(ia, true));
                    break;
                case 16:
                    t = sd.acosh(in);
                    ia = Nd4j.rand(minibatch, nOut).addi(1.01); //Only defined for x >= 1
                    tc.expectedOutput(t.getVarName(), Nd4j.getExecutioner().execAndReturn(new ACosh(ia.dup())));
                    break;
                case 17:
                    t = sd.asin(in);
                    ia = Nd4j.rand(minibatch, nOut).muli(1.8).subi(0.9);
                    tc.expectedOutput(t.getVarName(), Transforms.asin(ia, true));
                    break;
                case 18:
                    t = sd.atan(in);
                    ia = Nd4j.rand(minibatch, nOut).muli(4).subi(2);
                    tc.expectedOutput(t.getVarName(), Transforms.atan(ia, true));
                    break;
                case 19:
                    t = sd.atanh(in);
                    ia = Nd4j.rand(minibatch, nOut).muli(1.8).subi(0.9);
                    tc.expectedOutput(t.getVarName(), Transforms.atanh(ia, true));
                    break;
                case 20:
                    t = sd.cosh(in);
                    tc.expectedOutput(t.getVarName(), Transforms.cosh(ia, true));
                    break;
                case 21:
                    t = sd.cube(in);
                    tc.expectedOutput(t.getVarName(), Transforms.pow(ia, 3.0, true));
                    break;
                case 22:
                    t = sd.elu(in);
                    tc.expectedOutput(t.getVarName(), Transforms.elu(ia, true));
                    break;
                case 23:
                    //TODO SHOULDN'T THIS HAVE A DIMENSION ARG???
                    t = sd.softmax(in);
                    ia = Nd4j.rand(minibatch, nOut);
                    tc.expectedOutput(t.getVarName(), Nd4j.getExecutioner().execAndReturn(new OldSoftMax(ia.dup())));
                    break;
                case 24:
                    t = sd.sqrt(in);
                    ia = Nd4j.rand(minibatch, nOut);
                    tc.expectedOutput(t.getVarName(), Transforms.sqrt(ia, true));
                    break;
                case 25:
                    t = sd.square(in);
                    tc.expectedOutput(t.getVarName(), Transforms.pow(ia, 2.0, true));
                    break;
                case 26:
                    t = sd.transpose(in);
                    tc.expectedOutput(t.getVarName(), ia.transpose().dup());
                    break;
                case 27:
                    t = sd.abs(in);
                    tc.expectedOutput(t.getVarName(), Transforms.abs(ia, true));
                    break;
                case 28:
                    t = sd.sinh(in);
                    tc.expectedOutput(t.getVarName(), Transforms.sinh(ia, true));
                    break;
                case 29:
                    t = sd.asinh(in);
                    tc.expectedOutput(t.getVarName(), Nd4j.getExecutioner().execAndReturn(new ASinh(ia.dup())));
                    break;
                case 30:
                    t = sd.exp(in);
                    tc.expectedOutput(t.getVarName(), Transforms.exp(ia, true));
                    break;
                case 31:
                    t = sd.floor(in);
                    tc.expectedOutput(t.getVarName(), Transforms.floor(ia, true));
                    break;
                case 32:
                    t = sd.relu(in, 0.0);
                    ia = Nd4j.rand(minibatch, nOut);
                    tc.expectedOutput(t.getVarName(), Transforms.relu(ia, true));
                    break;
                case 33:
                    t = sd.hardTanh(in);
                    ia = Nd4j.rand(minibatch, nOut).muli(2).subi(1.0);
                    tc.expectedOutput(t.getVarName(), Transforms.hardTanh(ia, true));
                    break;
                case 34:
                    t = sd.logSigmoid(in);
                    tc.expectedOutput(t.getVarName(), Nd4j.getExecutioner().execAndReturn(new LogSigmoid(ia.dup())));
                    break;
                case 35:
                    t = sd.swish(in);
                    tc.expectedOutput(t.getVarName(), Nd4j.getExecutioner().execAndReturn(new Swish(ia.dup())));
                    break;
                case 36:
                    t = sd.sign(in);
                    tc.expectedOutput(t.getVarName(), Transforms.sign(ia, true));
                    break;
                case 37:
                    t = sd.softsign(in);
                    tc.expectedOutput(t.getVarName(), Transforms.softsign(ia, true));
                    break;
                case 38:
                    t = sd.leakyRelu(in, 0.0);
                    ia = Nd4j.rand(minibatch, nOut);
                    tc.expectedOutput(t.getVarName(), Transforms.leakyRelu(ia, true));
                    break;
                case 39:
                    t = sd.logSoftmax(in);
                    ia = Nd4j.rand(minibatch, nOut).muli(10).subi(5);
                    tc.expectedOutput(t.getVarName(), Transforms.log(Transforms.softmax(ia, true)));
                    stdevLoss = true;
                    break;
                case 40:
                    t = sd.selu(in);
                    tc.expectedOutput(t.getVarName(), Nd4j.getExecutioner().execAndReturn(new SELU(ia.dup())));
                    break;
                case 41:
                    t = sd.gt(in, 1.0);
                    tc.expectedOutput(t.getVarName(), ia.gt(1.0));
                    break;
                case 42:
                    t = sd.gte(in, 1.0);
                    tc.expectedOutput(t.getVarName(), ia.gte(1.0));
                    break;
                case 43:
                    t = sd.lt(in, 1.0);
                    tc.expectedOutput(t.getVarName(), ia.lt(1.0));
                    break;
                case 44:
                    t = sd.lte(in, 1.0);
                    tc.expectedOutput(t.getVarName(), ia.lte(1.0));
                    break;
                case 45:
                    t = sd.eq(in, 2.0);
                    ia = Nd4j.linspace(1, minibatch * nOut, minibatch * nOut).reshape('c', minibatch, nOut);
                    tc.expectedOutput(t.getVarName(), ia.eq(2.0));
                    break;
                case 46:
                    t = sd.neq(in, 2.0);
                    ia = Nd4j.linspace(1, minibatch * nOut, minibatch * nOut).reshape('c', minibatch, nOut);
                    tc.expectedOutput(t.getVarName(), ia.neq(2.0));
                    break;
                case 47:
                    t = sd.ceil(in);
                    tc.expectedOutput(t.getVarName(), Transforms.ceil(ia, true));
                    break;
                case 48:
                    ia = Nd4j.randn(ia.shape()).muli(2);
                    t = sd.clipByValue(in, -3, 2);
                    INDArray expOut48 = ia.dup();
                    BooleanIndexing.replaceWhere(expOut48, -3, Conditions.lessThan(-3));
                    BooleanIndexing.replaceWhere(expOut48, 2, Conditions.greaterThan(2));
                    tc.expectedOutput(t.getVarName(), expOut48);
                    break;
                case 49:
                    //Clip by norm, dimension 0, some below threshold, some above
                    double clip = 2.0;
                    t = sd.clipByNorm(in, clip, 0);
                    ia = Nd4j.rand(ia.shape());
                    ia.diviRowVector(ia.norm2(0)).muli(clip);  //Norm2 is now 'clip' (i.e., exactly at threshold
                    //System.out.println(ia.norm2(0));
                    ia.muliColumnVector(Nd4j.linspace(0.9, 1.1, ia.size(0)).transpose());
                    //System.out.println(ia.norm2(0));

                    INDArray expOut49 = Nd4j.create(ia.shape());
                    for (int j = 0; j < ia.columns(); j++) {
                        INDArray origCol = ia.getColumn(j);
                        if (origCol.norm2Number().doubleValue() < clip) {
                            expOut49.putColumn(j, origCol);
                        } else {
                            expOut49.putColumn(j, origCol.mul(clip / origCol.norm2Number().doubleValue()));
                        }
                    }
                    tc.expectedOutput(t.getVarName(), expOut49);
                    //System.out.println(expOut.norm2(0));
                    break;
                //TODO clip by norm along other dimensions
                case 50:
                    dim = 1;
                    t = sd.reverse(in, dim);
                    INDArray expOut50 = Nd4j.create(ia.shape());
                    DynamicCustomOp reverse = DynamicCustomOp.builder("reverse")
                            .addIntegerArguments(dim)
                            .addInputs(ia).addOutputs(expOut50).build();
                    Nd4j.getExecutioner().exec(reverse);
                    tc.expectedOutput(t.getVarName(), expOut50);
                    break;
                case 51:
                    dim = 0;
                    boolean exclusive = false;
                    boolean reverseBool = false;


                    t = sd.cumsum(in, exclusive, reverseBool, dim);
                    INDArray expOut51 = Nd4j.create(ia.shape());
                    DynamicCustomOp cumsum = DynamicCustomOp.builder("cumsum")
                            .addIntegerArguments((exclusive) ? 1 : 0, (reverseBool) ? 1 : 0, dim)
                            .addInputs(ia).addOutputs(expOut51).build();
                    Nd4j.getExecutioner().exec(cumsum);
                    tc.expectedOutput(t.getVarName(), expOut51);
                    break;
                case 52:
                    dim = 0;
                    boolean ex = false;
                    boolean revBool = false;
                    t = sd.cumprod(in, ex, revBool, dim);
                    INDArray expOut52 = Nd4j.create(ia.shape());
                    for( int s0=0; s0<ia.size(0); s0++){
                        for( int s1=0; s1<ia.size(1); s1++ ){
                            double prod = 1.0;
                            for(int x=0; x<=s0; x++ ){
                                prod *= ia.getDouble(x, s1);
                            }
                            expOut52.putScalar(s0, s1, prod);
                        }
                    }
                    tc.expectedOutput(t.getVarName(), expOut52);
                    break;
                case 53:
                    t = sd.diag(in);
                    ia = Nd4j.create(new float[]{4, 2});
                    in = sd.var("in", new int[]{1, 2});
                    INDArray expOut53 = Nd4j.create(new int[]{2, 2});
                    DynamicCustomOp op = DynamicCustomOp.builder("diag").addInputs(ia).addOutputs(expOut53).build();
                    Nd4j.getExecutioner().exec(op);
                    tc.expectedOutput(t.getVarName(), expOut53);
                    break;
                case 54:
                    t = sd.erf(in);
                    INDArray expOut54 = Nd4j.createUninitialized(ia.shape(), ia.ordering());
                    Nd4j.getExecutioner().exec(new Erf(ia, expOut54));
                    tc.expectedOutput(t.getVarName(), expOut54);
                    break;
                case 55:
                    t = sd.erfc(in);
                    tc.expectedOutput(t.getVarName(), Nd4j.getExecutioner().execAndReturn(new Erfc(ia, Nd4j.createUninitialized(ia.shape(), ia.ordering()))));
                    break;
                case 56:
                    t = sd.expm1(in);
                    tc.expectedOutput(t.getVarName(),Transforms.expm1(ia, true));
                    break;
                case 57:
                    t = sd.log1p(in);
                    ia = Nd4j.rand(minibatch, nOut);
                    tc.expectedOutput(t.getVarName(), Transforms.log1p(ia, true));
                    break;
                case 58:
                    t = sd.round(in);
                    tc.expectedOutput(t.getVarName(), Transforms.round(ia, true));
                    break;
                case 59:
                    ia = Nd4j.create(new float[]{4, 2});
                    in = sd.var("in", new int[]{1, 2});
                    t = sd.rsqrt(in);
                    tc.expectedOutput(t.getVarName(),Nd4j.getExecutioner().execAndReturn(new RSqrt(ia, Nd4j.create(ia.shape(), ia.ordering()))));
                    break;
                case 60:
                    t = sd.relu6(in, 0);
                    ia = Nd4j.rand(minibatch, nOut);
                    tc.expectedOutput(t.getVarName(),Transforms.relu6(ia, true));
                    break;
                case 61:
                    ia = Nd4j.create(new float[] {2, 2});
                    in = sd.var("in", new int[]{1, 2});
                    sd.associateArrayWithVariable(ia, in);
                    double value = 42;
                    t = sd.fill(in, value);
                    tc.expectedOutput(t.getVarName(), Nd4j.valueArrayOf(new int[]{2,2}, 42));
                    break;
                case 62:
                    t = sd.hardSigmoid(in);
                    tc.expectedOutput(t.getVarName(), Nd4j.getExecutioner().execAndReturn(new HardSigmoid(ia, ia.dup())));
                    break;
                case 63:
                    t = sd.scalarMax(in, 0.5);
                    tc.expectedOutput(t.getVarName(), Transforms.max(ia, 0.5, true));
                    break;
                case 64:
                    t = sd.scalarMin(in, 0.5);
                    tc.expectedOutput(t.getVarName(), Transforms.min(ia, 0.5, true));
                    break;
                case 65:
                    t = sd.assign(in, 0.5);
                    tc.expectedOutput(t.getVarName(), ia.dup().assign(0.5));
                    break;
                case 66:
                    t = sd.scalarFloorMod(in, 0.5);
                    tc.expectedOutput(t.getVarName(), Nd4j.getExecutioner().execAndReturn(new ScalarFMod(ia.dup(), 0.5)));
                    break;
                case 67:
                    t = sd.reciprocal(in);
                    tc.expectedOutput(t.getVarName(), ia.rdiv(1.0));
                    break;
                case 68:
                    t = sd.shape(in);
                    tc.expectedOutput(t.getVarName(), Nd4j.create(ArrayUtil.toDouble(ia.shape())));
                    break;
                case 69:
                    t = sd.rank(in);
                    tc.expectedOutput(t.getVarName(), Nd4j.create(new double[]{ia.rank()}));
                    break;
                case 70:
                    t = sd.onesLike(in);
                    tc.expectedOutput(t.getVarName(), Nd4j.ones(ia.shape()));
                    break;
                case 71:
                    ia = Nd4j.randn(nOut, nOut);
                    t = sd.diagPart(in);
                    tc.expectedOutput(t.getVarName(), Nd4j.trueVector(new double[]{ia.getDouble(0,0), ia.getDouble(1,1), ia.getDouble(2,2), ia.getDouble(3,3)}));
                    break;
                default:
                    throw new RuntimeException();
            }


            DifferentialFunction[] funcs = sd.functions();
            String name = funcs[0].opName();


            String msg = "test: " + i + " - " + name;
            log.info("*** Starting test: " + msg);

            SDVariable loss;
            if(stdevLoss){
                loss = sd.standardDeviation("loss", t, false, Integer.MAX_VALUE);   //.standardDeviation("loss", t, true, Integer.MAX_VALUE);
            } else {
                loss = sd.mean("loss", t);
            }

            sd.associateArrayWithVariable(ia, in);

            tc.testName(name);
            String error = OpValidation.validate(tc);
            if(error != null){
                allFailed.add(name);
            }
        }

        if (allSkipped.size() > 0) {
            log.info("All backward skipped transforms: " + allSkipped);
            log.info(allSkipped.size() + " backward passes were skipped.");
        }

        if (allFailed.size() > 0) {
            log.error("All failed transforms: " + allFailed);
            fail(allFailed.size() + " transforms failed");
        }
    }

    @Test
    public void testPairwiseTransforms() {
        /*
        add, sub, mul, div, rsub, rdiv
        eq, neq, gt, lt, gte, lte, or, and, xor
        min, max
        mmul
        tensormmul
         */
        //Test transforms (pairwise)
        Nd4j.getRandom().setSeed(12345);

        List<String> allFailed = new ArrayList<>();
        for (int i = 0; i < 23; i++) {

            SameDiff sd = SameDiff.create();

            int nOut = 4;
            int minibatch = 5;
            SDVariable in1 = sd.var("in1", new int[]{-1, nOut});
            SDVariable in2 = sd.var("in2", new int[]{-1, nOut});

            INDArray ia = Nd4j.randn(minibatch, nOut);
            INDArray ib = Nd4j.randn(minibatch, nOut);

            SDVariable t;
            TestCase tc = new TestCase(sd);
            switch (i) {
                case 0:
                    t = in1.add(in2);
                    tc.expectedOutput(t.getVarName(), ia.add(ib));
                    break;
                case 1:
                    t = in1.sub(in2);
                    tc.expectedOutput(t.getVarName(),ia.sub(ib));
                    break;
                case 2:
                    t = in1.mul(in2);
                    tc.expectedOutput(t.getVarName(), ia.mul(ib));
                    break;
                case 3:
                    t = in1.div(in2);
                    tc.expectedOutput(t.getVarName(), ia.div(ib));
                    break;
                case 4:
                    t = in1.rsub(in2);
                    tc.expectedOutput(t.getVarName(), ia.rsub(ib));
                    break;
                case 5:
                    t = in1.rdiv(in2);
                    tc.expectedOutput(t.getVarName(), ia.rdiv(ib));
                    break;
                case 6:
                    t = sd.eq(in1, in2);
                    tc.expectedOutput(t.getVarName(), ia.eq(ib));
                    break;
                case 7:
                    t = sd.neq(in1, in2);
                    tc.expectedOutput(t.getVarName(), ia.neq(ib));
                    break;
                case 8:
                    t = sd.gt(in1, in2);
                    tc.expectedOutput(t.getVarName(), ia.gt(ib));
                    break;
                case 9:
                    t = sd.lt(in1, in2);
                    tc.expectedOutput(t.getVarName(), ia.lt(ib));
                    break;
                case 10:
                    t = sd.gte(in1, in2);
                    INDArray expOut10 = ia.dup();
                    Nd4j.getExecutioner().exec(new GreaterThanOrEqual(new INDArray[]{ia, ib}, new INDArray[]{expOut10}));
                    tc.expectedOutput(t.getVarName(), expOut10);
                    break;
                case 11:
                    t = sd.lte(in1, in2);
                    INDArray expOut11 = ia.dup();
                    Nd4j.getExecutioner().exec(new LessThanOrEqual(new INDArray[]{ia, ib}, new INDArray[]{expOut11}));
                    tc.expectedOutput(t.getVarName(), expOut11);
                    break;
                case 12:
                    ia = Nd4j.getExecutioner().exec(new BernoulliDistribution(ia, 0.5));
                    ib = Nd4j.getExecutioner().exec(new BernoulliDistribution(ib, 0.5));
                    t = sd.or(in1, in2);
                    tc.expectedOutput(t.getVarName(), Transforms.or(ia, ib));
                    break;
                case 13:
                    ib = Nd4j.randn(nOut, nOut);
                    t = sd.mmul(in1, in2);
                    tc.expectedOutput(t.getVarName(), ia.mmul(ib));
                    break;
                case 14:
                    t = sd.max(in1, in2);
                    tc.expectedOutput(t.getVarName(), Nd4j.getExecutioner().execAndReturn(new OldMax(ia, ib, ia.dup(), ia.length())));
                    break;
                case 15:
                    t = sd.min(in1, in2);
                    tc.expectedOutput(t.getVarName(), Nd4j.getExecutioner().execAndReturn(new OldMin(ia, ib, ia.dup(), ia.length())));
                    break;
                case 16:
                    ia = Nd4j.getExecutioner().exec(new BernoulliDistribution(ia, 0.5));
                    ib = Nd4j.getExecutioner().exec(new BernoulliDistribution(ib, 0.5));
                    t = sd.and(in1, in2);
                    tc.expectedOutput(t.getVarName(), Transforms.and(ia, ib));
                    break;
                case 17:
                    ia = Nd4j.getExecutioner().exec(new BernoulliDistribution(ia, 0.5));
                    ib = Nd4j.getExecutioner().exec(new BernoulliDistribution(ib, 0.5));
                    t = sd.xor(in1, in2);
                    tc.expectedOutput(t.getVarName(), Transforms.xor(ia, ib));
                    break;
                case 18:
                    t = sd.assign(in1, in2);
                    tc.expectedOutput(t.getVarName(), ib);
                    break;
                case 19:
                    t = sd.atan2(in1, in2);
                    tc.expectedOutput(t.getVarName(), Transforms.atan2(ib, ia));    //Note: y,x order for samediff; x,y order for transforms
                    break;
                case 20:
                    t = sd.mergeAdd(in1, in2, in2);
                    tc.expectedOutput(t.getVarName(), ia.add(ib).add(ib));
                    break;
                case 21:
                    t = in1.squaredDifference(in2);
                    INDArray expOut21 = Nd4j.create(ia.shape(), ia.ordering());
                    DynamicCustomOp squareDiff = DynamicCustomOp.builder("squaredsubtract")
                            .addInputs(ia, ib)
                            .addOutputs(expOut21)
                            .build();
                    Nd4j.getExecutioner().exec(squareDiff);
                    tc.expectedOutput(t.getVarName(), expOut21);
                    break;
                case 22:
                    //set diag
                    ia = Nd4j.randn(nOut, nOut);
                    ib = Nd4j.randn(1, nOut).reshape(nOut);
                    INDArray expOut22 = ia.dup();
                    for( int j=0; j<nOut; j++ ){
                        expOut22.putScalar(j,j, ib.getDouble(j));
                    }
                    t = sd.setDiag(in1, in2);
                    tc.expectedOutput(t.getVarName(), expOut22);
                    break;
                default:
                    throw new RuntimeException();
            }


            DifferentialFunction[] funcs = sd.functions();
            String name = funcs[0].opName();

            String msg = "test: " + i + " - " + name;
            log.info("*** Starting test: " + msg);

            SDVariable loss = sd.mean("loss", t);

            sd.associateArrayWithVariable(ia, in1);
            sd.associateArrayWithVariable(ib, in2);

            tc.testName(name);
            String error = OpValidation.validate(tc);
            if(error != null){
                allFailed.add(name);
            }
        }

        if (allFailed.size() > 0) {
            log.error("All failed transforms: " + allFailed);
            fail(allFailed.size() + " transforms failed");
        }
    }
}
