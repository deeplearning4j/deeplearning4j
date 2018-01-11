package org.nd4j.autodiff.gradcheck;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.*;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.GreaterThanOrEqual;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.LessThanOrEqual;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.OldMax;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.OldMin;
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution;
import org.nd4j.linalg.factory.Nd4j;
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
    public void testTransforms() {
        //Test transforms (non-pairwise)
        Nd4j.getRandom().setSeed(12345);

        List<String> allFailed = new ArrayList<>();
        for (int i = 0; i < 47; i++) {

            SameDiff sd = SameDiff.create();

            int nOut = 4;
            int minibatch = 5;
            SDVariable in = sd.var("in", new int[]{-1, nOut});

            INDArray ia = Nd4j.randn(minibatch, nOut);

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
                    expOut = ia.div(4);
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
                    ia = Nd4j.rand(minibatch, nOut);
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
                    //TODO DIMENSION ARG???
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
                    ia = Nd4j.linspace(1,minibatch*nOut, minibatch*nOut).reshape('c', minibatch, nOut);
                    expOut = ia.eq(2.0);
                    break;
                case 46:
                    t = sd.neq(in, 2.0);
                    ia = Nd4j.linspace(1,minibatch*nOut, minibatch*nOut).reshape('c', minibatch, nOut);
                    expOut = ia.neq(2.0);
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

            if(!expOut.equals(out)){
                allFailed.add(msg + " - FAILED ON FORWARD");
                continue;
            }

            boolean ok;
            try{
                ok = GradCheckUtil.checkGradients(sd);
            } catch (Exception e){
                e.printStackTrace();
                msg += " - EXCEPTION";
                ok = false;
            }

//            assertTrue(msg, ok);
            if(!ok){
                allFailed.add(msg);
            }
        }

        if(allFailed.size() > 0){
            log.error("All failed transforms: " + allFailed);
            fail(allFailed.size() + " transforms failed");
        }
    }

    @Test
    public void testPairwiseTransforms(){
        /*
        add, sub, mul, div, rsub, rdiv
        eq, neq, gt, lt, gte, lte, or,
        min, max
        mmul
        tensormmul
         */
        //Test transforms (pairwise)
        Nd4j.getRandom().setSeed(12345);

        List<String> allFailed = new ArrayList<>();
        for (int i = 0; i < 16; i++) {
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
            try{
                ok = GradCheckUtil.checkGradients(sd);
            } catch (Exception e){
                e.printStackTrace();
                msg += " - EXCEPTION";
                ok = false;
            }

            if(!ok){
                allFailed.add(msg);
            }
        }

        if(allFailed.size() > 0){
            log.error("All failed transforms: " + allFailed);
            fail(allFailed.size() + " transforms failed");
        }
    }
}
