package org.nd4j.autodiff.gradcheck;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.transforms.*;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.GreaterThanOrEqual;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.LessThanOrEqual;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.OldMax;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.OldMin;
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.*;

@Slf4j
public class GradCheckTransforms {

    static {
        Nd4j.create(1);
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
    }

    @Test
    public void testCross() {
        INDArray a = Nd4j.create(new float[] {4, 2 , 1}, new int[] {1, 3});
        INDArray b = Nd4j.create(new float[] {1, 3 , 4}, new int[] {1, 3});

        INDArray expOut = Nd4j.create(1,3);

        DynamicCustomOp op = DynamicCustomOp.builder("cross").addInputs(a, b).addOutputs(expOut).build();
        Nd4j.getExecutioner().exec(op);

        SameDiff sd = SameDiff.create();

        SDVariable sdA = sd.var("a", expOut.shape());
        SDVariable sdB = sd.var("a", expOut.shape());


        sd.associateArrayWithVariable(a, sdA);
        sd.associateArrayWithVariable(b, sdB);

        SDVariable t = sd.cross(sdA, sdB);
        SDVariable loss = sd.mean("loss", t);
        sd.exec();
        INDArray out = t.getArr();

        if (!expOut.equals(out)) {
            log.info("batch to space failed on forward");
        }

        try {
            GradCheckUtil.checkGradients(sd);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Test
    public void testSpaceToDepth() {
        Nd4j.getRandom().setSeed(1337);

        int miniBatch = 128;
        int blockSize = 4;
        String dataFormat = "NHWC";
        int isNHWC = dataFormat.equals("NHWC")? 1: 0;
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

        SDVariable t = sd.spaceToDepth(sdInput, blockSize, dataFormat);
        SDVariable loss = sd.mean("loss", t);
        sd.exec();
        INDArray out = t.getArr();

        if (!expOut.equals(out)) {
            log.info("depth to space failed on forward");
        }

        try {
            GradCheckUtil.checkGradients(sd);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Test
    public void testDepthToSpace() {
        Nd4j.getRandom().setSeed(1337);

        int miniBatch = 128;
        int blockSize = 4;
        String dataFormat = "NHWC";
        int isNHWC = dataFormat.equals("NHWC")? 1: 0;
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

        SDVariable t = sd.depthToSpace(sdInput, blockSize, dataFormat);
        SDVariable loss = sd.mean("loss", t);
        sd.exec();
        INDArray out = t.getArr();

        if (!expOut.equals(out)) {
            log.info("depth to space failed on forward");
        }

        try {
            GradCheckUtil.checkGradients(sd);
        } catch (Exception e) {
            e.printStackTrace();
        }
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

        SDVariable t = sd.batchToSpace(sdInput, new int[]{2, 2}, new int[][]{{0, 0}, {0, 0}});
        SDVariable loss = sd.mean("loss", t);
        sd.exec();
        INDArray out = t.getArr();

        if (!expOut.equals(out)) {
            log.info("batch to space failed on forward");
        }

        try {
            GradCheckUtil.checkGradients(sd);
        } catch (Exception e) {
            e.printStackTrace();
        }

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

        SDVariable t = sd.spaceToBatch(sdInput, new int[]{2, 2}, new int[][]{{0, 0}, {0, 0}});
        SDVariable loss = sd.mean("loss", t);
        sd.exec();
        INDArray out = t.getArr();

        if (!expOut.equals(out)) {
            log.info("space to batch failed on forward");
        }

        try {
            GradCheckUtil.checkGradients(sd);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Test
    public void testDiag() {
        SameDiff sd = SameDiff.create();

        INDArray ia = Nd4j.create(new float[] {4,2});
        SDVariable in = sd.var("in", new int[]{1, 2});
        INDArray expOut = Nd4j.create(new int[] {2,2});
        DynamicCustomOp diag = DynamicCustomOp.builder("diag").addInputs(ia).addOutputs(expOut).build();
        Nd4j.getExecutioner().exec(diag);
        SDVariable t = sd.diag(in);

        SDVariable loss = sd.max("loss", t, 0, 1);

        sd.associateArrayWithVariable(ia, in);
        sd.exec();
        INDArray out = t.getArr();

        if(!expOut.equals(out)){log.info("forward failed");}

        try{
            GradCheckUtil.checkGradients(sd);
        } catch (Exception e){
            e.printStackTrace();
        }
    }

    @Test
    public void testTransforms() {
        //Test transforms (non-pairwise)
        Nd4j.getRandom().setSeed(12345);

        List<String> allFailed = new ArrayList<>();
        for (int i = 0; i < 54; i++) {

            SameDiff sd = SameDiff.create();

            int nOut = 4;
            int minibatch = 5;
            SDVariable in = sd.var("in", new int[]{-1, nOut});

            INDArray ia = Nd4j.randn(minibatch, nOut);

            int dim;
            SDVariable t;
            INDArray expOut;
            switch (i) {
                case 0:
                    t = in.add(5.0);
                    expOut = ia.add(5.0);
                    break;
                case 1:
                    t = in.sub(5.0);
                    expOut = ia.sub(5.0);
                    break;
                case 2:
                    t = in.mul(2.5);
                    expOut = ia.mul(2.5);
                    break;
                case 3:
                    t = in.div(4.0);
                    expOut = ia.div(4.0);
                    break;
                case 4:
                    t = in.rsub(5.0);
                    expOut = ia.rsub(5.0);
                    break;
                case 5:
                    t = in.rdiv(1.0);
                    expOut = ia.rdiv(1.0);
                    break;
                case 6:
                    t = sd.pow(in, 2.5);
                    ia = Nd4j.rand(minibatch, nOut);
                    expOut = Transforms.pow(ia, 2.5, true);
                    break;
                case 7:
                    t = sd.sigmoid(in);
                    ia = Nd4j.rand(minibatch, nOut).muli(2).subi(1.0);
                    expOut = Transforms.sigmoid(ia, true);
                    break;
                case 8:
                    t = sd.tanh(in);
                    ia = Nd4j.rand(minibatch, nOut).muli(2).subi(1.0);
                    expOut = Transforms.tanh(ia, true);
                    break;
                case 9:
                    t = sd.tan(in);
                    ia = Nd4j.rand(minibatch, nOut);
                    expOut = Nd4j.getExecutioner().execAndReturn(new Tan(ia.dup()));
                    break;
                case 10:
                    t = sd.cos(in);
                    expOut = Transforms.cos(ia, true);
                    break;
                case 11:
                    t = sd.sin(in);
                    expOut = Transforms.sin(ia, true);
                    break;
                case 12:
                    t = sd.softplus(in);
                    expOut = Transforms.softPlus(ia, true);
                    break;
                case 13:
                    t = sd.log(in);
                    ia = Nd4j.rand(minibatch, nOut);
                    expOut = Transforms.log(ia, true);
                    break;
                case 14:
                    t = sd.neg(in);
                    expOut = ia.neg();
                    break;
                case 15:
                    t = sd.acos(in);
                    ia = Nd4j.rand(minibatch, nOut).muli(1.8).subi(0.9);
                    expOut = Transforms.acos(ia, true);
                    break;
                case 16:
                    t = sd.acosh(in);
                    ia = Nd4j.rand(minibatch, nOut).addi(1.01); //Only defined for x >= 1
                    expOut = Nd4j.getExecutioner().execAndReturn(new ACosh(ia.dup()));
                    break;
                case 17:
                    t = sd.asin(in);
                    ia = Nd4j.rand(minibatch, nOut).muli(1.8).subi(0.9);
                    expOut = Transforms.asin(ia, true);
                    break;
                case 18:
                    t = sd.atan(in);
                    ia = Nd4j.rand(minibatch, nOut).muli(4).subi(2);
                    expOut = Transforms.atan(ia, true);
                    break;
                case 19:
                    t = sd.atanh(in);
                    ia = Nd4j.rand(minibatch, nOut).muli(1.8).subi(0.9);
                    expOut = Transforms.atanh(ia, true);
                    break;
                case 20:
                    t = sd.cosh(in);
                    expOut = Transforms.cosh(ia, true);
                    break;
                case 21:
                    t = sd.cube(in);
                    expOut = Transforms.pow(ia, 3.0, true);
                    break;
                case 22:
                    t = sd.elu(in);
                    expOut = Transforms.elu(ia, true);
                    break;
                case 23:
                    //TODO SHOULDN'T THIS HAVE A DIMENSION ARG???
                    t = sd.softmax(in);
                    ia = Nd4j.rand(minibatch, nOut);
                    expOut = Nd4j.getExecutioner().execAndReturn(new OldSoftMax(ia.dup()));
                    break;
                case 24:
                    t = sd.sqrt(in);
                    ia = Nd4j.rand(minibatch, nOut);
                    expOut = Transforms.sqrt(ia, true);
                    break;
                case 25:
                    t = sd.square(in);
                    expOut = Transforms.pow(ia, 2.0, true);
                    break;
                case 26:
                    t = sd.transpose(in);
                    expOut = ia.transpose().dup();
                    break;
                case 27:
                    t = sd.abs(in);
                    expOut = Transforms.abs(ia, true);
                    break;
                case 28:
                    t = sd.sinh(in);
                    expOut = Transforms.sinh(ia, true);
                    break;
                case 29:
                    t = sd.asinh(in);
                    expOut = Nd4j.getExecutioner().execAndReturn(new ASinh(ia.dup()));
                    break;
                case 30:
                    t = sd.exp(in);
                    expOut = Transforms.exp(ia, true);
                    break;
                case 31:
                    t = sd.floor(in);
                    expOut = Transforms.floor(ia, true);
                    break;
                case 32:
                    t = sd.relu(in, 0.0);
                    ia = Nd4j.rand(minibatch, nOut);
                    expOut = Transforms.relu(ia, true);
                    break;
                case 33:
                    t = sd.hardTanh(in);
                    ia = Nd4j.rand(minibatch, nOut).muli(2).subi(1.0);
                    expOut = Transforms.hardTanh(ia, true);
                    break;
                case 34:
                    t = sd.logSigmoid(in);
                    expOut = Nd4j.getExecutioner().execAndReturn(new LogSigmoid(ia.dup()));
                    break;
                case 35:
                    t = sd.swish(in);
                    expOut = Nd4j.getExecutioner().execAndReturn(new Swish(ia.dup()));
                    break;
                case 36:
                    t = sd.sign(in);
                    expOut = Transforms.sign(ia, true);
                    break;
                case 37:
                    t = sd.softsign(in);
                    expOut = Transforms.softsign(ia, true);
                    break;
                case 38:
                    t = sd.leakyRelu(in, 0.0);
                    ia = Nd4j.rand(minibatch, nOut);
                    expOut = Transforms.leakyRelu(ia, true);
                    break;
                case 39:
                    // TODO DIMENSION ARG???
                    // TODO fix me
                    t = sd.logSoftmax(in);
                    ia = Nd4j.rand(minibatch, nOut);
                    expOut = Transforms.log(Transforms.softmax(ia, true));
                    break;
                case 40:
                    t = sd.selu(in);
                    expOut = Nd4j.getExecutioner().execAndReturn(new SELU(ia.dup()));
                    break;
                case 41:
                    t = sd.gt(in, 1.0);
                    expOut = ia.gt(1.0);
                    break;
                case 42:
                    t = sd.gte(in, 1.0);
                    expOut = ia.gte(1.0);
                    break;
                case 43:
                    t = sd.lt(in, 1.0);
                    expOut = ia.lt(1.0);
                    break;
                case 44:
                    t = sd.lte(in, 1.0);
                    expOut = ia.lte(1.0);
                    break;
                case 45:
                    t = sd.eq(in, 2.0);
                    ia = Nd4j.linspace(1, minibatch * nOut, minibatch * nOut).reshape('c', minibatch, nOut);
                    expOut = ia.eq(2.0);
                    break;
                case 46:
                    t = sd.neq(in, 2.0);
                    ia = Nd4j.linspace(1, minibatch * nOut, minibatch * nOut).reshape('c', minibatch, nOut);
                    expOut = ia.neq(2.0);
                    break;
                case 47:
                    t = sd.ceil(in);
                    expOut = Transforms.ceil(ia, true);
                    break;
                case 48:
                    ia = Nd4j.randn(ia.shape()).muli(2);
                    t = sd.clipByValue(in, -3, 2);
                    expOut = ia.dup();
                    BooleanIndexing.replaceWhere(expOut, -3, Conditions.lessThan(-3));
                    BooleanIndexing.replaceWhere(expOut, 2, Conditions.greaterThan(2));
                    break;
                case 49:
                    //Clip by norm, dimension 0, some below threshold, some above
                    double clip = 2.0;
                    ia = Nd4j.rand(ia.shape());
                    ia.muliRowVector(ia.norm2(0).rdiv(clip));  //Exactly at threshold...
                    System.out.println(ia.norm2(0));
                    ia.muliRowVector(Nd4j.linspace(0.9, 1.1, ia.size(1)));
                    System.out.println(ia.norm2(0));

                    expOut = Nd4j.create(ia.shape());
                    for (int j = 0; j < ia.columns(); j++) {
                        INDArray origCol = ia.getColumn(j);
                        if (origCol.norm2Number().doubleValue() < clip) {
                            expOut.putColumn(j, origCol);
                        } else {
                            expOut.putColumn(j, origCol.mul(clip / origCol.norm2Number().doubleValue()));
                        }
                    }

                    t = sd.clipByNorm(in, clip);
                    break;
                    //TODO clip by norm along other dimensions
                case 50:
                    dim = 1;
                    t = sd.reverse(in, dim);
                    expOut = Nd4j.create(ia.shape());
                    DynamicCustomOp reverse = DynamicCustomOp.builder("reverse")
                            .addIntegerArguments(dim)
                            .addInputs(ia).addOutputs(expOut).build();
                    Nd4j.getExecutioner().exec(reverse);

                    break;
                case 51:
                    dim = 0;
                    boolean exclusive = false;
                    boolean reverseBool = false;
                    t = sd.cumsum(in, exclusive, reverseBool, dim);
                    expOut = Nd4j.create(ia.shape());
                    DynamicCustomOp cumsum = DynamicCustomOp.builder("cumsum")
                            .addIntegerArguments(dim)
                            .addInputs(ia).addOutputs(expOut).build();
                    Nd4j.getExecutioner().exec(cumsum);
                    break;
                case 52:
                    dim = 0;
                    boolean ex = false;
                    boolean revBool = false;
                    t = sd.cumsum(in, ex, revBool, dim);
                    expOut = Nd4j.create(ia.shape());
                    DynamicCustomOp cumprod = DynamicCustomOp.builder("cumprod")
                            .addIntegerArguments(dim)
                            .addInputs(ia).addOutputs(expOut).build();
                    Nd4j.getExecutioner().exec(cumprod);
                case 53:
                    ia = Nd4j.create(new float[] {4,2});
                    in = sd.var("in", new int[]{1, 2});
                    expOut = Nd4j.create(new int[] {2,2});
                    DynamicCustomOp op = DynamicCustomOp.builder("diag").addInputs(ia).addOutputs(expOut).build();
                    Nd4j.getExecutioner().exec(op);
                    t = sd.diag(in);
                    break;
                default:
                    throw new RuntimeException();
            }


            DifferentialFunction[] funcs = sd.functions();
            String name = funcs[0].opName();


            String msg = "test: " + i + " - " + name;
            log.info("*** Starting test: " + msg);

            SDVariable loss = sd.mean("loss", t);


            sd.associateArrayWithVariable(ia, in);
            sd.exec();
            INDArray out = t.getArr();

            if (!expOut.equals(out)) {
                allFailed.add(msg + " - FAILED ON FORWARD");
                continue;
            }

            boolean ok;
            try {
                ok = GradCheckUtil.checkGradients(sd);
            } catch (Exception e) {
                e.printStackTrace();
                msg += " - EXCEPTION";
                ok = false;
            }

            assertTrue(msg, ok);
            if(!ok){

                allFailed.add(msg);
            }
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
        for (int i = 0; i < 21; i++) {

            SameDiff sd = SameDiff.create();

            int nOut = 4;
            int minibatch = 5;
            SDVariable in1 = sd.var("in1", new int[]{-1, nOut});
            SDVariable in2 = sd.var("in2", new int[]{-1, nOut});

            INDArray ia = Nd4j.randn(minibatch, nOut);
            INDArray ib = Nd4j.randn(minibatch, nOut);

            SDVariable t;
            INDArray expOut;
            switch (i) {
                case 0:
                    t = in1.add(in2);
                    expOut = ia.add(ib);
                    break;
                case 1:
                    t = in1.sub(in2);
                    expOut = ia.sub(ib);
                    break;
                case 2:
                    t = in1.mul(in2);
                    expOut = ia.mul(ib);
                    break;
                case 3:
                    t = in1.div(in2);
                    expOut = ia.div(ib);
                    break;
                case 4:
                    t = in1.rsub(in2);
                    expOut = ia.rsub(ib);
                    break;
                case 5:
                    t = in1.rdiv(in2);
                    expOut = ia.rdiv(ib);
                    break;
                case 6:
                    t = sd.eq(in1, in2);
                    expOut = ia.eq(ib);
                    break;
                case 7:
                    t = sd.neq(in1, in2);
                    expOut = ia.neq(ib);
                    break;
                case 8:
                    t = sd.gt(in1, in2);
                    expOut = ia.gt(ib);
                    break;
                case 9:
                    t = sd.lt(in1, in2);
                    expOut = ia.lt(ib);
                    break;
                case 10:
                    t = sd.gte(in1, in2);
                    expOut = ia.dup();
                    Nd4j.getExecutioner().exec(new GreaterThanOrEqual(new INDArray[]{ia, ib}, new INDArray[]{expOut}));
                    break;
                case 11:
                    t = sd.lte(in1, in2);
                    expOut = ia.dup();
                    Nd4j.getExecutioner().exec(new LessThanOrEqual(new INDArray[]{ia, ib}, new INDArray[]{expOut}));
                    break;
                case 12:
                    ia = Nd4j.getExecutioner().exec(new BernoulliDistribution(ia, 0.5));
                    ib = Nd4j.getExecutioner().exec(new BernoulliDistribution(ib, 0.5));
                    t = sd.or(in1, in2);
                    expOut = Transforms.or(ia, ib);
                    break;
                case 13:
                    ib = Nd4j.randn(nOut, nOut);
                    t = sd.mmul(in1, in2);
                    expOut = ia.mmul(ib);
                    break;
                case 14:
                    t = sd.max(in1, in2);
                    expOut = Nd4j.getExecutioner().execAndReturn(new OldMax(ia, ib, ia.dup(), ia.length()));
                    break;
                case 15:
                    t = sd.min(in1, in2);
                    expOut = Nd4j.getExecutioner().execAndReturn(new OldMin(ia, ib, ia.dup(), ia.length()));
                    break;
                case 16:
                    ia = Nd4j.getExecutioner().exec(new BernoulliDistribution(ia, 0.5));
                    ib = Nd4j.getExecutioner().exec(new BernoulliDistribution(ib, 0.5));
                    t = sd.and(in1, in2);
                    expOut = Transforms.and(ia, ib);
                    break;
                case 17:
                    ia = Nd4j.getExecutioner().exec(new BernoulliDistribution(ia, 0.5));
                    ib = Nd4j.getExecutioner().exec(new BernoulliDistribution(ib, 0.5));
                    t = sd.xor(in1, in2);
                    expOut = Transforms.xor(ia, ib);
                    break;
                case 18:
                    t = sd.assign(in1, in2);
                    expOut = ib;
                    break;
                case 19:
                    t = sd.atan2(in1, in2);
                    expOut = Transforms.atan2(ib, ia);    //Note: y,x order for samediff; x,y order for transforms
                    break;
                case 20:
                    t = sd.mergeAdd(in1, in2, in2);
                    expOut = ia.add(ib).add(ib);
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
            sd.exec();
            INDArray out = t.getArr();

            assertEquals(msg, expOut, out);

            boolean ok;
            try {
                ok = GradCheckUtil.checkGradients(sd);
            } catch (Exception e) {
                e.printStackTrace();
                msg += " - EXCEPTION";
                ok = false;
            }

            if (!ok) {
                allFailed.add(msg);
            }
        }

        if (allFailed.size() > 0) {
            log.error("All failed transforms: " + allFailed);
            fail(allFailed.size() + " transforms failed");
        }
    }
}
