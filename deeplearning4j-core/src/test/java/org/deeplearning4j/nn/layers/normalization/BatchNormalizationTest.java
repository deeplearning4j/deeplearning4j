package org.deeplearning4j.nn.layers.normalization;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToCnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.BatchNormalizationParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastAddOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastDivOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastSubOp;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.util.ArrayUtil;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.util.Arrays;
import java.util.Random;

import static org.junit.Assert.*;

/**
 */
public class BatchNormalizationTest {

    static {
        //Force Nd4j initialization, then set data type to double:
        Nd4j.zeros(1);
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
    }

    protected INDArray dnnInput = Nd4j.linspace(0, 31, 32).reshape(2, 16);
    protected INDArray dnnEpsilon = Nd4j.linspace(0, 31, 32).reshape(2, 16);

    protected INDArray cnnInput = Nd4j.linspace(0, 63, 64).reshape(2, 2, 4, 4);
    protected INDArray cnnEpsilon = Nd4j.linspace(0, 63, 64).reshape(2, 2, 4, 4);

    @Before
    public void doBefore() {
    }

    protected static Layer getLayer(int nOut) {
        return getLayer(nOut, Nd4j.EPS_THRESHOLD, false, -1, -1);
    }

    protected static Layer getLayer(int nOut, double epsilon, boolean lockGammaBeta, double gamma, double beta) {
        BatchNormalization.Builder b = new BatchNormalization.Builder().nOut(nOut).eps(epsilon);
        if (lockGammaBeta) {
            b.lockGammaBeta(true).gamma(gamma).beta(beta);
        }
        BatchNormalization bN = b.build();
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .iterations(1).layer(bN).build();

        int numParams = conf.getLayer().initializer().numParams(conf, true);
        INDArray params = null;
        if(numParams > 0) {
            params = Nd4j.create(1, numParams);
        }
        Layer layer = conf.getLayer().instantiate(conf, null, 0, params, true);
        if(numParams > 0){
            layer.setBackpropGradientsViewArray(Nd4j.create(1,numParams));
        }
        return layer;
    }


    @Test
    public void testDnnForwardPass() {
        int nOut = 10;
        Layer l = getLayer(nOut, 0.0, false, -1, -1);
        assertEquals(4*nOut, l.numParams());        //Gamma, beta, global mean, global var

        INDArray randInput = Nd4j.rand(100, nOut);
        INDArray output = l.activate(randInput, true);

        INDArray mean = output.mean(0);
        INDArray stdev = output.std(false, 0);

        System.out.println(Arrays.toString(mean.data().asFloat()));

        assertArrayEquals(new float[nOut], mean.data().asFloat(), 1e-6f);
        assertEquals(Nd4j.ones(1, nOut), stdev);

        //If we fix gamma/beta: expect different mean and variance...
        double gamma = 2.0;
        double beta = 3.0;
        l = getLayer(nOut, 0.0, true, gamma, beta);
        assertEquals(2*nOut, l.numParams()); //Should have only global mean/var parameters
        output = l.activate(randInput, true);
        mean = output.mean(0);
        stdev = output.std(false, 0);

        assertEquals(Nd4j.valueArrayOf(mean.shape(), beta), mean);
        assertEquals(Nd4j.valueArrayOf(stdev.shape(), gamma), stdev);
    }

    @Test
    public void testDnnForwardBackward(){
        double eps = 1e-5;
        int nIn = 4;
        int minibatch = 2;
        Nd4j.getRandom().setSeed(12345);
        INDArray input = Nd4j.rand('c',new int[]{minibatch,nIn});

        //TODO: other values for gamma/beta
        INDArray gamma = Nd4j.ones(1,nIn);
        INDArray beta = Nd4j.zeros(1,nIn);

        Layer l = getLayer(nIn, eps, false, -1, -1);

        INDArray mean = input.mean(0);
        INDArray var = input.var(false, 0);
        INDArray xHat = input.subRowVector(mean).divRowVector(Transforms.sqrt(var.add(eps),true));
        INDArray outExpected = xHat.mulRowVector(gamma).addRowVector(beta);

        INDArray out = l.activate(input, true);

        System.out.println(Arrays.toString(outExpected.data().asDouble()));
        System.out.println(Arrays.toString(out.data().asDouble()));

        assertEquals(outExpected, out);

        //-------------------------------------------------------------
        //Check backprop
        INDArray epsilon = Nd4j.rand(minibatch,nIn);    //dL/dy

        INDArray dldgammaExp = epsilon.mul(xHat).sum(0);
        INDArray dldbetaExp = epsilon.sum(0);

        INDArray dldxhat = epsilon.mulRowVector(gamma);
        INDArray dldvar = dldxhat.mul(input.subRowVector(mean)).mul(-0.5)
                .mulRowVector(Transforms.pow(var.add(eps),-3.0/2.0, true)).sum(0);
        INDArray dldmu = dldxhat.mulRowVector(Transforms.pow(var.add(eps),-1.0/2.0, true)).neg().sum(0)
                .add(dldvar.mul(input.subRowVector(mean).mul(-2.0).sum(0).div(minibatch)));
        INDArray dldinExp = dldxhat.mulRowVector(Transforms.pow(var.add(eps),-1.0/2.0, true))
                .add(input.subRowVector(mean).mul(2.0/minibatch).mulRowVector(dldvar))
                .addRowVector(dldmu.mul(1.0/minibatch));

        Pair<Gradient,INDArray> p = l.backpropGradient(epsilon);

        INDArray dldgamma = p.getFirst().getGradientFor("gamma");
        INDArray dldbeta = p.getFirst().getGradientFor("beta");

        assertEquals(dldgammaExp, dldgamma);
        assertEquals(dldbetaExp, dldbeta);

        System.out.println("EPSILONS");
        System.out.println(Arrays.toString(dldinExp.data().asDouble()));
        System.out.println(Arrays.toString(p.getSecond().dup().data().asDouble()));
        assertEquals(dldinExp, p.getSecond());
    }

    @Test
    public void testCnnForwardPass() {
        int nOut = 10;
        Layer l = getLayer(nOut, 0.0, false, -1, -1);
        assertEquals(4*nOut, l.numParams());        //Gamma, beta, global mean, global var
        int hw = 15;

        Nd4j.getRandom().setSeed(12345);
        INDArray randInput = Nd4j.rand(12345, 100, nOut, hw, hw);
        INDArray output = l.activate(randInput, true);

        assertEquals(4, output.rank());

        INDArray mean = output.mean(0, 2, 3);
        INDArray stdev = output.std(false, 0, 2, 3);

        assertArrayEquals(new float[nOut], mean.data().asFloat(), 1e-6f);
        assertArrayEquals(Nd4j.ones(1, nOut).data().asFloat(), stdev.data().asFloat(), 1e-6f);

        //If we fix gamma/beta: expect different mean and variance...
        double gamma = 2.0;
        double beta = 3.0;
        l = getLayer(nOut, 0.0, true, gamma, beta);
        assertEquals(2*nOut, l.numParams()); //Should have only global mean/var parameters
        output = l.activate(randInput, true);
        mean = output.mean(0, 2, 3);
        stdev = output.std(false, 0, 2, 3);

        assertEquals(Nd4j.valueArrayOf(mean.shape(), beta), mean);
        assertEquals(Nd4j.valueArrayOf(stdev.shape(), gamma), stdev);
    }

    @Test
    public void test2dVs4d(){
        //Idea: 2d and 4d should be the same...
        Nd4j.getRandom().setSeed(12345);

        int m = 2;
        int h = 3;
        int w = 3;
        int nOut = 2;

        INDArray in = Nd4j.rand('c',m*h*w,nOut);

        INDArray in4 = in.dup();
        in4 = Shape.newShapeNoCopy(in4,new int[]{m,h,w,nOut},false);
        assertNotNull(in4);
        in4 = in4.permute(0,3,1,2).dup();
        INDArray arr = Nd4j.rand(1,m*h*w*nOut).reshape('f',h,w,m,nOut).permute(2,3,1,0);
        in4 = arr.assign(in4);

        Layer l1 = getLayer(nOut);
        Layer l2 = getLayer(nOut);

        INDArray out2d = l1.activate(in.dup(), true);
        INDArray out4d = l2.activate(in4.dup(), true);

        INDArray out4dAs2 = out4d.permute(0,2,3,1).dup('c');
        out4dAs2 = Shape.newShapeNoCopy(out4dAs2,new int[]{m*h*w,nOut},false);

        assertEquals(out2d, out4dAs2);

        //Test backprop:
        INDArray epsilons2d = Nd4j.rand('c',m*h*w,nOut);
        INDArray epsilons4d = epsilons2d.dup();
        epsilons4d = Shape.newShapeNoCopy(epsilons4d,new int[]{m,h,w,nOut},false);
        assertNotNull(epsilons4d);
        epsilons4d = epsilons4d.permute(0,3,1,2).dup();

        Pair<Gradient,INDArray> b2d = l1.backpropGradient(epsilons2d);
        Pair<Gradient,INDArray> b4d = l2.backpropGradient(epsilons4d);

        INDArray e4dAs2d = b4d.getSecond().permute(0,2,3,1).dup('c');
        e4dAs2d = Shape.newShapeNoCopy(e4dAs2d,new int[]{m*h*w,nOut},false);

        assertEquals(b2d.getSecond(), e4dAs2d);
    }

    @Test
    public void testCnnForwardBackward(){
        double eps = 1e-5;
        int nIn = 4;
        int hw = 3;
        int minibatch = 2;
        Nd4j.getRandom().setSeed(12345);
        INDArray input = Nd4j.rand('c',new int[]{minibatch,nIn,hw,hw});

        //TODO: other values for gamma/beta
        INDArray gamma = Nd4j.ones(1,nIn);
        INDArray beta = Nd4j.zeros(1,nIn);

        Layer l = getLayer(nIn, eps, false, -1, -1);

        INDArray mean = input.mean(0,2,3);
        INDArray var = input.var(false, 0,2,3);
        INDArray xHat = Nd4j.getExecutioner().execAndReturn(
                new BroadcastSubOp(input,mean,input.dup(),1));
        Nd4j.getExecutioner().execAndReturn(
                new BroadcastDivOp(xHat,Transforms.sqrt(var.add(eps),true),xHat,1));

        INDArray outExpected = Nd4j.getExecutioner().execAndReturn(
                new BroadcastMulOp(xHat,gamma,xHat.dup(),1));
        Nd4j.getExecutioner().execAndReturn(new BroadcastAddOp(outExpected,beta,outExpected,1));

        INDArray out = l.activate(input, true);

        System.out.println(Arrays.toString(outExpected.data().asDouble()));
        System.out.println(Arrays.toString(out.data().asDouble()));

        assertEquals(outExpected, out);

        //-------------------------------------------------------------
        //Check backprop
        INDArray epsilon = Nd4j.rand('c',new int[]{minibatch,nIn,hw,hw});    //dL/dy

        int effectiveMinibatch = minibatch * hw * hw;

        INDArray dldgammaExp = epsilon.mul(xHat).sum(0,2,3);
        INDArray dldbetaExp = epsilon.sum(0,2,3);

        INDArray dldxhat = Nd4j.getExecutioner().execAndReturn(
                new BroadcastMulOp(epsilon,gamma,epsilon.dup(),1)); //epsilon.mulRowVector(gamma);

        INDArray inputSubMean = Nd4j.getExecutioner().execAndReturn(
                new BroadcastSubOp(input,mean,input.dup(),1));

        INDArray dldvar = dldxhat.mul(inputSubMean).mul(-0.5);
        dldvar = Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(
                dldvar, Transforms.pow(var.add(eps),-3.0/2.0, true), dldvar.dup(), 1));
        dldvar = dldvar.sum(0,2,3);


        INDArray dldmu = Nd4j.getExecutioner().execAndReturn(
                new BroadcastMulOp(dldxhat, Transforms.pow(var.add(eps),-1.0/2.0, true), dldxhat.dup(), 1))
                .neg().sum(0,2,3);
        dldmu = dldmu.add(
                dldvar.mul(inputSubMean.mul(-2.0).sum(0,2,3).div(effectiveMinibatch)));

        INDArray dldinExp = Nd4j.getExecutioner().execAndReturn(
                new BroadcastMulOp(dldxhat, Transforms.pow(var.add(eps),-1.0/2.0, true), dldxhat.dup(), 1 ));
        dldinExp = dldinExp.add(
                Nd4j.getExecutioner().execAndReturn(
                        new BroadcastMulOp(inputSubMean.mul(2.0/effectiveMinibatch),dldvar, inputSubMean.dup(), 1)));
        dldinExp = Nd4j.getExecutioner().execAndReturn(new BroadcastAddOp(dldinExp, dldmu.mul(1.0/effectiveMinibatch), dldinExp.dup(), 1));

        Pair<Gradient,INDArray> p = l.backpropGradient(epsilon);

        INDArray dldgamma = p.getFirst().getGradientFor("gamma");
        INDArray dldbeta = p.getFirst().getGradientFor("beta");

        assertEquals(dldgammaExp, dldgamma);
        assertEquals(dldbetaExp, dldbeta);

        System.out.println("EPSILONS");
        System.out.println(Arrays.toString(dldinExp.data().asDouble()));
        System.out.println(Arrays.toString(p.getSecond().dup().data().asDouble()));
        assertEquals(dldinExp, p.getSecond());
    }

    @Test
    public void testDnnShapeBatchNormForward() {
        Layer layer = getLayer(16);
        // Confirm param initial shape before override
        assertArrayEquals(new int[]{1, 16}, layer.getParam("gamma").shape());
        assertArrayEquals(new int[]{1, 16}, layer.getParam("beta").shape());
        layer.setParam("gamma", Nd4j.linspace(0, 15, 16));
        layer.setParam("beta", Nd4j.linspace(0, 15, 16));


        INDArray activationsActual = layer.preOutput(dnnInput);
        INDArray activationsExpected = Nd4j.create(new double[]{
                0.00000000e+00, 7.81248399e-11, 1.56249680e-10,
                2.34374298e-10, 3.12499360e-10, 3.90624422e-10,
                4.68748595e-10, 5.46873657e-10, 6.24998719e-10,
                7.03122893e-10, 7.81248843e-10, 8.59373017e-10,
                9.37497191e-10, 1.01562314e-09, 1.09374731e-09,
                1.17187327e-09, 0.00000000e+00, 2.00000000e+00,
                4.00000000e+00, 6.00000000e+00, 8.00000000e+00,
                1.00000000e+01, 1.20000000e+01, 1.40000000e+01,
                1.60000000e+01, 1.80000000e+01, 2.00000000e+01,
                2.20000000e+01, 2.40000000e+01, 2.60000000e+01,
                2.80000000e+01, 3.00000000e+01
        }, new int[]{2, 16});

        System.out.println(Arrays.toString(activationsExpected.data().asFloat()));
        System.out.println(Arrays.toString(activationsActual.data().asFloat()));

        assertEquals(activationsExpected, activationsActual);
        assertArrayEquals(activationsExpected.shape(), activationsActual.shape());
    }


    @Test
    public void testDnnShapeBatchNormBack() {
        Layer layer = getLayer(16);
        layer.setParam("gamma", Nd4j.linspace(0, 15, 16));
        layer.setParam("beta", Nd4j.linspace(0, 15, 16));

        layer.preOutput(dnnInput);
        layer.setBackpropGradientsViewArray(Nd4j.create(1, 64));
        Pair<Gradient, INDArray> actualOut = layer.backpropGradient(dnnEpsilon);

        INDArray dnnExpectedEpsilonOut = Nd4j.create(new double[]{
                0.00000000e+00, -1.56249680e-10, -3.12499360e-10,
                -4.68748595e-10, -6.24998719e-10, -7.81248843e-10,
                -9.37497191e-10, -1.09374731e-09, -1.24999744e-09,
                -1.40624934e-09, -1.56249413e-09, -1.71874603e-09,
                -1.87499438e-09, -2.03124273e-09, -2.18749818e-09,
                -2.34373942e-09, 0.00000000e+00, 1.56249680e-10,
                3.12499804e-10, 4.68748595e-10, 6.24997831e-10,
                7.81248843e-10, 9.37497191e-10, 1.09374731e-09,
                1.24999744e-09, 1.40624579e-09, 1.56249769e-09,
                1.71874603e-09, 1.87499438e-09, 2.03124983e-09,
                2.18749818e-09, 2.34374653e-09
        }, new int[]{2, 16});


        // short calculation expected output
        INDArray dnnExpectedEpsilonOutOther = Nd4j.create(new double[]{
                16., 15., 14., 13., 12., 11., 10., 9., 8., 7., 6.,
                5., 4., 3., 2., 1., -16., -15., -14., -13., -12., -11.,
                -10., -9., -8., -7., -6., -5., -4., -3., -2., -1.
        }, new int[]{2, 16});

        INDArray expectedGGamma = Nd4j.create(new double[]
                {
                        16., 16., 16., 16., 16., 16., 16., 16., 16., 16., 16.,
                        16., 16., 16., 16., 16.
                }, new int[]{1, 16});

        INDArray expectedBeta = Nd4j.create(new double[]
                {
                        16., 18., 20., 22., 24., 26., 28., 30., 32., 34., 36.,
                        38., 40., 42., 44., 46.
                }, new int[]{1, 16});

        System.out.println(Arrays.toString(dnnExpectedEpsilonOut.data().asFloat()));
        System.out.println(Arrays.toString(actualOut.getSecond().data().asFloat()));

        // arrays are the same but assert does not see that
        assertEquals(dnnExpectedEpsilonOut, actualOut.getSecond());
        assertEquals(expectedGGamma, actualOut.getFirst().getGradientFor("gamma"));
        assertEquals(expectedBeta, actualOut.getFirst().getGradientFor("beta"));
    }

    @Test
    public void testCnnShapeBatchNormForward() {
        Layer layer = getLayer(2);
        // Confirm param initial shape before override
        assertArrayEquals(new int[]{1, 2}, layer.getParam("gamma").shape());
        assertArrayEquals(new int[]{1, 2}, layer.getParam("beta").shape());

        layer.setParam("gamma", Nd4j.linspace(2, 3, 2));
        layer.setParam("beta", Nd4j.linspace(2, 3, 2));
        INDArray activationsActual = layer.preOutput(cnnInput);
        INDArray activationsExpected = Nd4j.create(new double[]{
                3.90625310e-11, 3.90625310e-11, 3.90625310e-11,
                3.90625310e-11, 3.90625310e-11, 3.90625310e-11,
                3.90625310e-11, 3.90625310e-11, 3.90625310e-11,
                3.90625310e-11, 3.90625310e-11, 3.90625310e-11,
                3.90625310e-11, 3.90625310e-11, 3.90625310e-11,
                3.90625310e-11, 5.85940185e-11, 5.85940185e-11,
                5.85940185e-11, 5.85940185e-11, 5.85940185e-11,
                5.85940185e-11, 5.85940185e-11, 5.85940185e-11,
                5.85940185e-11, 5.85940185e-11, 5.85940185e-11,
                5.85940185e-11, 5.85940185e-11, 5.85940185e-11,
                5.85940185e-11, 5.85940185e-11, 4.00000000e+00,
                4.00000000e+00, 4.00000000e+00, 4.00000000e+00,
                4.00000000e+00, 4.00000000e+00, 4.00000000e+00,
                4.00000000e+00, 4.00000000e+00, 4.00000000e+00,
                4.00000000e+00, 4.00000000e+00, 4.00000000e+00,
                4.00000000e+00, 4.00000000e+00, 4.00000000e+00,
                6.00000000e+00, 6.00000000e+00, 6.00000000e+00,
                6.00000000e+00, 6.00000000e+00, 6.00000000e+00,
                6.00000000e+00, 6.00000000e+00, 6.00000000e+00,
                6.00000000e+00, 6.00000000e+00, 6.00000000e+00,
                6.00000000e+00, 6.00000000e+00, 6.00000000e+00,
                6.00000000e+00
        }, new int[]{2, 2, 4, 4});

        System.out.println(Arrays.toString(activationsExpected.data().asFloat()));
        System.out.println(Arrays.toString(activationsActual.data().asFloat()));

        assertEquals(activationsExpected, activationsActual);
        assertArrayEquals(activationsExpected.shape(), activationsActual.shape());
    }

    @Test
    public void testCnnShapeBatchNormBack() {
        Layer layer = getLayer(2);
        layer.setParam("gamma", Nd4j.linspace(2, 3, 2));
        layer.setParam("beta", Nd4j.linspace(2, 3, 2));
        layer.preOutput(cnnInput);
        layer.setBackpropGradientsViewArray(Nd4j.create(1, 4));
        Pair<Gradient, INDArray> actualOut = layer.backpropGradient(cnnEpsilon);

        INDArray expectedEpsilonOut = Nd4j.create(new double[]{
                -7.81250620e-11, -7.81250620e-11, -7.81250620e-11,
                -7.81250620e-11, -7.81250620e-11, -7.81250620e-11,
                -7.81250620e-11, -7.81250620e-11, -7.81250620e-11,
                -7.81250620e-11, -7.81250620e-11, -7.81250620e-11,
                -7.81250620e-11, -7.81250620e-11, -7.81250620e-11,
                -7.81250620e-11, -1.17187149e-10, -1.17187149e-10,
                -1.17187149e-10, -1.17187149e-10, -1.17188037e-10,
                -1.17187149e-10, -1.17187149e-10, -1.17187149e-10,
                -1.17187149e-10, -1.17187149e-10, -1.17187149e-10,
                -1.17188037e-10, -1.17186261e-10, -1.17186261e-10,
                -1.17188037e-10, -1.17188037e-10, 7.81250620e-11,
                7.81250620e-11, 7.81246179e-11, 7.81255061e-11,
                7.81250620e-11, 7.81250620e-11, 7.81246179e-11,
                7.81255061e-11, 7.81255061e-11, 7.81250620e-11,
                7.81246179e-11, 7.81246179e-11, 7.81255061e-11,
                7.81255061e-11, 7.81250620e-11, 7.81246179e-11,
                1.17187149e-10, 1.17187149e-10, 1.17188037e-10,
                1.17187149e-10, 1.17188037e-10, 1.17187149e-10,
                1.17187149e-10, 1.17188037e-10, 1.17187149e-10,
                1.17188037e-10, 1.17187149e-10, 1.17186261e-10,
                1.17188037e-10, 1.17188037e-10, 1.17188037e-10,
                1.17186261e-10
        }, new int[]{2, 2, 4, 4});

        INDArray expectedGGamma = Nd4j.create(new double[]
                {
                        512, 512
                }, new int[]{1, 2});

        INDArray expectedBeta = Nd4j.create(new double[]
                {
                        752, 1264
                }, new int[]{1, 2});

        System.out.println(Arrays.toString(expectedEpsilonOut.data().asFloat()));
        System.out.println(Arrays.toString(actualOut.getSecond().data().asFloat()));

        // arrays are the same but assert does not see that
        assertEquals(expectedEpsilonOut, actualOut.getSecond());
        assertEquals(expectedGGamma, actualOut.getFirst().getGradientFor("gamma"));
        assertEquals(expectedBeta, actualOut.getFirst().getGradientFor("beta"));

    }

    @Test
    public void testMultiCNNBNLayer() throws Exception {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .seed(123)
                .list()
                .layer(0, new ConvolutionLayer.Builder().nIn(1).nOut(6).weightInit(WeightInit.XAVIER).activation("relu").build())
                .layer(1, new BatchNormalization.Builder().build())
                .layer(2, new DenseLayer.Builder().nOut(2).build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .nIn(2).nOut(10).build())
                .backprop(true).pretrain(false)
                .setInputType(InputType.convolutionalFlat(28, 28, 1))
                .build();
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        DataSetIterator iter = new MnistDataSetIterator(2, 2);
        DataSet next = iter.next();

        network.setInput(next.getFeatureMatrix());
        INDArray activationsActual = network.preOutput(next.getFeatureMatrix());
        assertEquals(10, activationsActual.shape()[1], 1e-2);

        network.fit(next);
        INDArray actualGammaParam = network.getLayer(1).getParam(BatchNormalizationParamInitializer.GAMMA);
        INDArray actualBetaParam = network.getLayer(1).getParam(BatchNormalizationParamInitializer.BETA);
        assertTrue(actualGammaParam != null);
        assertTrue(actualBetaParam != null);
    }

    @Test
    public void testDBNBNMultiLayer() throws Exception {
        DataSetIterator iter = new MnistDataSetIterator(2, 2);
        DataSet next = iter.next();

        // Run with separate activation layer
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(2)
                .seed(123)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(28 * 28 * 1).nOut(10).weightInit(WeightInit.XAVIER).activation("relu").build())
                .layer(1, new BatchNormalization.Builder().nOut(10).build())
                .layer(2, new ActivationLayer.Builder().activation("relu").build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .nIn(10).nOut(10).build())
                .backprop(true).pretrain(false)
                .build();

        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();

        network.setInput(next.getFeatureMatrix());
        INDArray activationsActual = network.preOutput(next.getFeatureMatrix());
        assertEquals(10, activationsActual.shape()[1], 1e-2);

        network.fit(next);
        INDArray actualGammaParam = network.getLayer(1).getParam(BatchNormalizationParamInitializer.GAMMA);
        INDArray actualBetaParam = network.getLayer(1).getParam(BatchNormalizationParamInitializer.BETA);
        assertTrue(actualGammaParam != null);
        assertTrue(actualBetaParam != null);
    }

    @Ignore
    @Test
    public void testMultiLSTMBNLayer() throws Exception {
        // TODO once BatchNorm setup for RNN, expand this test
        int nChannelsIn = 3;
        int inputSize = 10 * 10 * nChannelsIn;    //10px x 10px x 3 channels
        int miniBatchSize = 4;
        int timeSeriesLength = 10;
        int nClasses = 3;

        Nd4j.getRandom().setSeed(12345);
        INDArray input = Nd4j.rand(new int[]{miniBatchSize, inputSize, timeSeriesLength});
        INDArray labels = Nd4j.zeros(miniBatchSize, nClasses, timeSeriesLength);
        Random r = new Random(12345);
        for (int i = 0; i < miniBatchSize; i++) {
            for (int j = 0; j < timeSeriesLength; j++) {
                int idx = r.nextInt(nClasses);
                labels.putScalar(new int[]{i, idx, j}, 1.0);
            }
        }
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5).nIn(3).nOut(5).stride(1, 1)
                        .activation("identity").weightInit(WeightInit.XAVIER).updater(Updater.NONE).build())    //Out: (10-5)/1+1 = 6 -> 6x6x5
                .layer(1, new BatchNormalization.Builder().build())
                .layer(2, new ActivationLayer.Builder().activation("relu").build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2).stride(1, 1).build())    //Out: (6-2)/1+1 = 5 -> 5x5x5
                .layer(4, new DenseLayer.Builder().nIn(5 * 5 * 5).nOut(4)
                        .updater(Updater.NONE).weightInit(WeightInit.XAVIER).activation("relu")
                        .build())
                .layer(5, new GravesLSTM.Builder().nIn(4).nOut(3)
                        .activation("identity").updater(Updater.NONE).weightInit(WeightInit.XAVIER)
                        .build())
//                .layer(6, new BatchNormalization.Builder().build())
//                .layer(7, new ActivationLayer.Builder().activation("tanh").build())
                .layer(6, new RnnOutputLayer.Builder().nIn(3).nOut(nClasses)
                        .activation("softmax").lossFunction(LossFunctions.LossFunction.MCXENT)
                        .updater(Updater.NONE).build())
                .cnnInputSize(10, 10, 3)
                .pretrain(false).backprop(true)
                .build();

        conf.getInputPreProcessors().put(0, new RnnToCnnPreProcessor(10, 10, 3));

        MultiLayerNetwork mln = new MultiLayerNetwork(conf);
        mln.init();
        mln.setInput(input);
        mln.setLabels(labels);
        mln.fit();

    }

    @Test
    public void testCNNBNActivationCombo() throws Exception {
        DataSetIterator iter = new MnistDataSetIterator(2, 2);
        DataSet next = iter.next();

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(2)
                .seed(123)
                .list()
                .layer(0, new ConvolutionLayer.Builder().nIn(1).nOut(6).weightInit(WeightInit.XAVIER).activation("identity").build())
                .layer(1, new BatchNormalization.Builder().build())
                .layer(2, new ActivationLayer.Builder().activation("relu").build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .nOut(10).build())
                .backprop(true).pretrain(false)
                .setInputType(InputType.convolutionalFlat(28, 28, 1))
                .build();

        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        network.fit(next);

        assertNotEquals(null, network.getLayer(0).getParam("W"));
        assertNotEquals(null, network.getLayer(0).getParam("b"));
    }


    @Test
    public void checkSerialization() throws Exception {
        //Serialize the batch norm network (after training), and make sure we get same activations out as before
        // i.e., make sure state is properly stored

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(2)
                .seed(12345)
                .list()
                .layer(0, new ConvolutionLayer.Builder().nIn(1).nOut(6).weightInit(WeightInit.XAVIER).activation("identity").build())
                .layer(1, new BatchNormalization.Builder().build())
                .layer(2, new ActivationLayer.Builder().activation("leakyrelu").build())
                .layer(3, new DenseLayer.Builder().nOut(10).activation("leakyrelu").build())
                .layer(4, new BatchNormalization.Builder().build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .nOut(10).build())
                .backprop(true).pretrain(false)
                .setInputType(InputType.convolutionalFlat(28, 28, 1))
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        DataSetIterator iter = new MnistDataSetIterator(16,true,12345);
        for( int i=0; i<20; i++ ){
            net.fit(iter.next());
        }

        INDArray in = iter.next().getFeatureMatrix();

        INDArray out = net.output(in, false);
        INDArray out2 = net.output(in, false);

        assertEquals(out, out2);

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ModelSerializer.writeModel(net,baos,true);
        baos.close();
        byte[] bArr = baos.toByteArray();

        ByteArrayInputStream bais = new ByteArrayInputStream(bArr);
        MultiLayerNetwork net2 = ModelSerializer.restoreMultiLayerNetwork(bais, true);

        INDArray outDeser = net2.output(in, false);

        assertEquals(out, outDeser);
    }


    @Ignore
    @Test
    public void testMultiLSTMLayer() throws Exception {
        // TODO use this to test when batch norm implemented for RNN

        int inputSize = 10 * 10;
        int miniBatchSize = 4;
        int timeSeriesLength = 10;
        int nClasses = 3;

        Nd4j.getRandom().setSeed(12345);
        INDArray input = Nd4j.rand(new int[]{miniBatchSize, inputSize, timeSeriesLength});
        INDArray labels = Nd4j.zeros(miniBatchSize, nClasses, timeSeriesLength);
        Random r = new Random(12345);
        for (int i = 0; i < miniBatchSize; i++) {
            for (int j = 0; j < timeSeriesLength; j++) {
                int idx = r.nextInt(nClasses);
                labels.putScalar(new int[]{i, idx, j}, 1.0);
            }
        }
        // Run without separate activation layer
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .list()
                .layer(0, new GravesLSTM.Builder().nIn(100).nOut(3)
                        .activation("tanh").updater(Updater.NONE).weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new RnnOutputLayer.Builder().nIn(3).nOut(nClasses)
                        .activation("softmax").lossFunction(LossFunctions.LossFunction.MCXENT)
                        .updater(Updater.NONE).build())
                .pretrain(false).backprop(true)
                .build();

        MultiLayerNetwork mln = new MultiLayerNetwork(conf);
        mln.init();
        mln.setInput(input);
        mln.setLabels(labels);
        mln.fit();


        // Run with separate activation layer
        MultiLayerConfiguration conf2 = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .list()
                .layer(0, new GravesLSTM.Builder().nIn(100).nOut(3)
                        .activation("identity").updater(Updater.NONE).weightInit(WeightInit.XAVIER)
                        .build())
//                .layer(, new BatchNormalization.Builder().build())
                .layer(1, new ActivationLayer.Builder().activation("tanh").build())
                .layer(2, new RnnOutputLayer.Builder().nIn(3).nOut(nClasses)
                        .activation("softmax").lossFunction(LossFunctions.LossFunction.MCXENT)
                        .updater(Updater.NONE).build())
                .pretrain(false).backprop(true)
                .build();

        conf2.getInputPreProcessors().put(1, new RnnToFeedForwardPreProcessor());
        conf2.getInputPreProcessors().put(2, new FeedForwardToRnnPreProcessor());

        MultiLayerNetwork mln2 = new MultiLayerNetwork(conf2);
        mln2.init();
        mln2.setInput(input);
        mln2.setLabels(labels);
        mln2.fit();

//        assertEquals(mln.getLayer(0).getParam("W"), mln2.getLayer(0).getParam("W"));
//        assertEquals(mln.getLayer(3).getParam("W"), mln2.getLayer(4).getParam("W"));
//        assertEquals(mln.getLayer(0).getParam("b"), mln2.getLayer(0).getParam("b"));
//        assertEquals(mln.getLayer(3).getParam("b"), mln2.getLayer(4).getParam("b"));
    }


    @Ignore
    @Test
    public void testMultiCNNLSTMLayer() throws Exception {
        // TODO use this to test when batch norm implemented for RNN
        int nChannelsIn = 3;
        int inputSize = 10 * 10 * nChannelsIn;    //10px x 10px x 3 channels
        int miniBatchSize = 4;
        int timeSeriesLength = 10;
        int nClasses = 3;

        Nd4j.getRandom().setSeed(12345);
        INDArray input = Nd4j.rand(new int[]{miniBatchSize, inputSize, timeSeriesLength});
        INDArray labels = Nd4j.zeros(miniBatchSize, nClasses, timeSeriesLength);
        Random r = new Random(12345);
        for (int i = 0; i < miniBatchSize; i++) {
            for (int j = 0; j < timeSeriesLength; j++) {
                int idx = r.nextInt(nClasses);
                labels.putScalar(new int[]{i, idx, j}, 1.0);
            }
        }
        // Run without separate activation layer
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5).nIn(3).nOut(5).stride(1, 1)
                        .activation("relu").weightInit(WeightInit.XAVIER).updater(Updater.NONE).build())    //Out: (10-5)/1+1 = 6 -> 6x6x5
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2).stride(1, 1).build())    //Out: (6-2)/1+1 = 5 -> 5x5x5
                .layer(2, new DenseLayer.Builder().nIn(5 * 5 * 5).nOut(4)
                        .updater(Updater.NONE).weightInit(WeightInit.XAVIER).activation("relu")
                        .build())
                .layer(3, new GravesLSTM.Builder().nIn(4).nOut(3)
                        .activation("tanh").updater(Updater.NONE).weightInit(WeightInit.XAVIER)
                        .build())
                .layer(4, new RnnOutputLayer.Builder().nIn(3).nOut(nClasses)
                        .activation("softmax").lossFunction(LossFunctions.LossFunction.MCXENT)
                        .updater(Updater.NONE).build())
                .cnnInputSize(10, 10, 3)
                .pretrain(false).backprop(true)
                .build();

        conf.getInputPreProcessors().put(0, new RnnToCnnPreProcessor(10, 10, 3));

        MultiLayerNetwork mln = new MultiLayerNetwork(conf);
        mln.init();
        mln.setInput(input);
        mln.setLabels(labels);
        mln.fit();


        // Run with separate activation layer
        MultiLayerConfiguration conf2 = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5).nIn(3).nOut(5).stride(1, 1)
                        .activation("identity").weightInit(WeightInit.XAVIER).updater(Updater.NONE).build())    //Out: (10-5)/1+1 = 6 -> 6x6x5
                .layer(1, new BatchNormalization.Builder().build())
                .layer(2, new ActivationLayer.Builder().activation("relu").build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2).stride(1, 1).build())    //Out: (6-2)/1+1 = 5 -> 5x5x5
                .layer(4, new DenseLayer.Builder().nIn(5 * 5 * 5).nOut(4)
                        .updater(Updater.NONE).weightInit(WeightInit.XAVIER).activation("relu")
                        .build())
                .layer(5, new GravesLSTM.Builder().nIn(4).nOut(3)
                        .activation("identity").updater(Updater.NONE).weightInit(WeightInit.XAVIER)
                        .build())
                .layer(6, new BatchNormalization.Builder().build())
                .layer(7, new ActivationLayer.Builder().activation("tanh").build())
                .layer(8, new RnnOutputLayer.Builder().nIn(3).nOut(nClasses)
                        .activation("softmax").lossFunction(LossFunctions.LossFunction.MCXENT)
                        .updater(Updater.NONE).build())
                .cnnInputSize(10, 10, 3)
                .pretrain(false).backprop(true)
                .build();

        conf2.getInputPreProcessors().put(0, new RnnToCnnPreProcessor(10, 10, 3));

        MultiLayerNetwork mln2 = new MultiLayerNetwork(conf2);
        mln2.init();
        mln2.setInput(input);
        mln2.setLabels(labels);
        mln2.fit();
//
//        assertEquals(mln.getLayer(0).getParam("W"), mln2.getLayer(0).getParam("W"));
//        assertEquals(mln.getLayer(3).getParam("W"), mln2.getLayer(4).getParam("W"));
//        assertEquals(mln.getLayer(0).getParam("b"), mln2.getLayer(0).getParam("b"));
//        assertEquals(mln.getLayer(3).getParam("b"), mln2.getLayer(4).getParam("b"));
    }

}
