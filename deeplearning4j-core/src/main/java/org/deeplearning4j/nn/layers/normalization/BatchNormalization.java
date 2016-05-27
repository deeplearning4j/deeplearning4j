package org.deeplearning4j.nn.layers.normalization;

import org.apache.commons.lang3.ArrayUtils;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.params.BatchNormalizationParamInitializer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastAddOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastDivOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastSubOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.util.ArrayUtil;

import java.lang.reflect.Constructor;
import java.util.*;

/**
 * Batch normalization layer.
 * Rerences:
 *  http://arxiv.org/pdf/1502.03167v3.pdf
 *  http://arxiv.org/pdf/1410.7455v8.pdf
 *  https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
 *
 * tried this approach but results did not match https://cthorey.github.io/backpropagation/
 *
 * ideal to apply this between linear and non-linear transformations in layers it follows
 **/

public class BatchNormalization extends BaseLayer<org.deeplearning4j.nn.conf.layers.BatchNormalization> {
    protected int index = 0;
    protected List<IterationListener> listeners = new ArrayList<>();
    protected int[] shape;
    protected INDArray mean;
    protected INDArray var;
    protected INDArray std;
    protected INDArray xMu;
    protected INDArray xHat;
    protected TrainingMode trainingMode;
    protected boolean setMeanVar = true;

    public BatchNormalization(NeuralNetConfiguration conf) {
        super(conf);
    }

    @Override
    public double calcL2() {
        return 0;
    }

    @Override
    public double calcL1() {
        return 0;
    }

    @Override
    public Type type() {
        return Type.NORMALIZATION;
    }

    @Override
    public Gradient error(INDArray input) {
        return null;
    }

    @Override
    public Gradient calcGradient(Gradient layerError, INDArray indArray) {
        return null;
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon) {
        INDArray nextEpsilon;
        shape = getShape(epsilon);
        int batchSize = epsilon.size(0); // number examples in batch
        org.deeplearning4j.nn.conf.layers.BatchNormalization layerConf = layerConf();

        INDArray gamma = (layerConf.isLockGammaBeta())? Nd4j.ones(shape) : getParam(BatchNormalizationParamInitializer.GAMMA);
        Gradient retGradient = new DefaultGradient();

        INDArray dGammaView = gradientViews.get(BatchNormalizationParamInitializer.GAMMA);
        INDArray dBetaView = gradientViews.get(BatchNormalizationParamInitializer.BETA);

        // paper and long calculation
        if (epsilon.rank() == 2) {
            INDArray dGamma = epsilon.mul(xHat).sum(0);
            INDArray dBeta = epsilon.sum(0); // sum over examples in batch
            INDArray dxhat = epsilon.mulRowVector(gamma);
            INDArray dsq = dxhat.mul(xMu).sum(0).mul(0.5).div(Transforms.pow(std, 3)).neg().div(batchSize);

            INDArray dxmu1 = dxhat.divRowVector(std);
            INDArray dxmu2 = xMu.mul(2).mulRowVector(dsq);

            INDArray dx1 = dxmu1.add(dxmu2);
            INDArray dmu = dx1.sum(0).neg();
            INDArray dx2 = dmu.div(batchSize);
            nextEpsilon = Nd4j.getExecutioner().execAndReturn(new BroadcastAddOp(dx1, dx2, dx1.dup(), -1));

            //alternative short calculation - does not match but more normalized epsilon
            INDArray r = xMu.divRowVector(Transforms.pow(std, 2)).mulRowVector(epsilon.mul(xMu).sum(0));
            INDArray otherEp = epsilon.mul(2).subRowVector(dBeta).mulRowVector(gamma.div(std.mul(2))).sub(r);

//            retGradient.setGradientFor(BatchNormalizationParamInitializer.GAMMA, dGamma);
//            retGradient.setGradientFor(BatchNormalizationParamInitializer.BETA, dBeta);

            //TODO rework this to avoid the assign here
            dGammaView.assign(dGamma);
            dBetaView.assign(dBeta);
            retGradient.setGradientFor(BatchNormalizationParamInitializer.GAMMA, dGammaView);
            retGradient.setGradientFor(BatchNormalizationParamInitializer.BETA, dBetaView);

        } else if (epsilon.rank() == 4){
            INDArray dGamma = epsilon.mul(xHat).sum(0,2,3);
            INDArray dBeta = epsilon.sum(0,2,3); // sum over examples in batch
            INDArray dxhat = Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(epsilon, gamma, epsilon.dup(), 1));

            INDArray dsq = dxhat.mul(xMu).sum(0).mul(0.5).div(Transforms.pow(std, 3)).neg().div(batchSize);

            INDArray dxmu1 = Nd4j.getExecutioner().execAndReturn(new BroadcastDivOp(dxhat, std, dxhat, new int[]{1,2,3}));
            INDArray dxmu2 = Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(xMu.mul(2), dsq, xMu.mul(2), new int[]{1,2,3}));

            INDArray dx1 = dxmu1.add(dxmu2);
            INDArray dmu = dx1.sum(0).neg();
            INDArray dx2 = dmu.div(batchSize);
            nextEpsilon = Nd4j.getExecutioner().execAndReturn(new BroadcastAddOp(dx1, dx2, dx1.dup(), new int[]{1,2,3}));
//            retGradient.setGradientFor(BatchNormalizationParamInitializer.GAMMA, dGamma);
//            retGradient.setGradientFor(BatchNormalizationParamInitializer.BETA, dBeta);

            //TODO rework this to avoid the assign here
            dGammaView.assign(dGamma);
            dBetaView.assign(dBeta);
            retGradient.setGradientFor(BatchNormalizationParamInitializer.GAMMA, dGammaView);
            retGradient.setGradientFor(BatchNormalizationParamInitializer.BETA, dBetaView);

        } else {
            // TODO setup BatchNorm for RNN http://arxiv.org/pdf/1510.01378v1.pdf
            throw new IllegalStateException("The layer prior to BatchNorm in the configuration is not currently supported.");
        }

        return new Pair<>(retGradient,nextEpsilon);
    }

    @Override
    public void merge(Layer layer, int batchSize) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void fit(INDArray data) {
    }

    @Override
    public INDArray activate(boolean training) {
        return preOutput(input, training == true? TrainingMode.TRAIN: TrainingMode.TEST);
    }

    @Override
    public Gradient gradient() {
        return gradient;
    }

    @Override
    public INDArray preOutput(INDArray x) {
        return preOutput(x,TrainingMode.TRAIN);
    }

    public INDArray preOutput(INDArray x, TrainingMode training){
        INDArray gamma, beta;
        INDArray activations = null;
        trainingMode = training;
        // TODO add this directly in layer or get the layer prior...
        // batchnorm true but need to clarify if activation before or after

        org.deeplearning4j.nn.conf.layers.BatchNormalization layerConf = layerConf();
        int batchSize = x.size(0); // number examples in batch
        shape = getShape(x);


        // xHat = x-xmean / sqrt(var + epsilon)
        INDArray mean, var;
        if (trainingMode == TrainingMode.TRAIN && layerConf.isUseBatchMean()) {
            // mean and var over samples in batch
            mean = x.mean(0);
            var = x.var(false, 0);
            var.addi(layerConf.getEps());
        } else {
            // cumulative mean and var - primarily used after training
            mean = this.mean;
            var = this.var;
        }
        std = Transforms.sqrt(var);

        if (layerConf.isLockGammaBeta()) {
            gamma = Nd4j.ones(shape);
            beta = Nd4j.zeros(shape);
        } else {
            gamma = getParam(BatchNormalizationParamInitializer.GAMMA);
            beta = getParam(BatchNormalizationParamInitializer.BETA);
        }

        // BN(xk) = gamma*xˆ + β (applying gamma and beta for each activation)
        if (x.rank() == 2) {
            xMu = Nd4j.getExecutioner().execAndReturn(new BroadcastSubOp(x, mean, x.dup(), -1));
            xHat = Nd4j.getExecutioner().execAndReturn(new BroadcastDivOp(xMu, std, xMu.dup(), -1));

            activations = xHat.dup().mulRowVector(gamma).addRowVector(beta);
        } else if (x.rank() == 4) {
            xMu = Nd4j.getExecutioner().execAndReturn(new BroadcastSubOp(x, mean, x.dup(), new int[]{1,2,3}));
            xHat = Nd4j.getExecutioner().execAndReturn(new BroadcastDivOp(xMu, std, xMu.dup(), new int[]{1,2,3}));

            activations = Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(xHat,gamma,xHat.dup(),1));
            activations = Nd4j.getExecutioner().execAndReturn(new BroadcastAddOp(activations,beta,activations,1));
        } else {
            // TODO setup BatchNorm for RNN http://arxiv.org/pdf/1510.01378v1.pdf
            throw new IllegalStateException("The layer prior to BatchNorm in the configuration is not currently supported.");
        }

        // store mean and var if using batch mean while training
        double decay;
        if(training == TrainingMode.TRAIN && layerConf.isUseBatchMean()) {
            // TODO track finetune phase here to update decay for finetune
//          layerConf.setN(layerConf.getN() + 1);
//          decay =  1. / layerConf.getN();

            if (setMeanVar){
                this.mean = this.mean == null? Nd4j.zeros(mean.shape()): this.mean;
                this.var = this.var == null? Nd4j.valueArrayOf(var.shape(), layerConf.getEps()): this.var;
                setMeanVar = false;
            }

            decay = layerConf.getDecay();
            double adjust = batchSize / Math.max(batchSize - 1., 1.);

            this.mean = mean.mul(decay).add(this.mean.mul(1 - decay));
            this.var = var.mul(decay).add(this.var.mul((1 - decay) * adjust));
        }

        return activations;
    }

    @Override
    public INDArray activate(TrainingMode training) {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray activate(INDArray input, TrainingMode training) {
        return preOutput(input,training);
    }

    @Override
    public INDArray preOutput(INDArray x, boolean training) {
        return preOutput(x,training ? TrainingMode.TRAIN : TrainingMode.TEST);
    }

    @Override
    public Layer transpose() {
        throw new UnsupportedOperationException();

    }

    @Override
    public Layer clone() {
        throw new UnsupportedOperationException();

    }

    @Override
    public Collection<IterationListener> getListeners() {
        return listeners;
    }

    @Override
    public void setListeners(IterationListener... listeners) {
        this.listeners = new ArrayList<>(Arrays.asList(listeners));
    }

    @Override
    public void setIndex(int index) {
        this.index = index;
    }

    @Override
    public int getIndex() {
        return index;
    }

    public int[] getShape(INDArray x) {
        if(x.rank() == 2 || x.rank() == 4)
            return new int[] {1, x.size(1)};
        if(x.rank() == 3) {
            int wDim = x.size(1);
            int hdim = x.size(2);
            if(x.size(0) > 1 && wDim * hdim == x.length())
                throw new IllegalArgumentException("Illegal input for batch size");
            return new int[] {1, wDim * hdim};
        }
        else throw new IllegalStateException("Unable to process input of rank " + x.rank());
    }

}
