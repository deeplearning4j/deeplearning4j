package org.deeplearning4j.nn.layers.normalization;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.params.BatchNormalizationParamInitializer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastAddOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastDivOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastSubOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.*;

/**
 * Batch normalization layer.
 * http://arxiv.org/pdf/1410.7455v8.pdf
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
    protected INDArray xHat;
    protected INDArray gGamma;
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
        int batchSize = epsilon.size(0); // number examples in batch
        INDArray reshapeEp = epsilon.dup().reshape(batchSize, shape[1]);

        org.deeplearning4j.nn.conf.layers.BatchNormalization layerConf = layerConf();
        INDArray gBeta = reshapeEp.sum(0); // sum over examples in batch

        if (trainingMode == TrainingMode.TRAIN && layerConf.isUseBatchMean()){
            gGamma = reshapeEp.mul(xHat).sum(0);
        }

        INDArray gamma = (layerConf.isLockGammaBeta())? Nd4j.onesLike(reshapeEp) : getParam(BatchNormalizationParamInitializer.GAMMA);
        INDArray coefficients =  Nd4j.getExecutioner().execAndReturn(new BroadcastDivOp(gamma, std, gamma, -1));

        INDArray tmp = Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(xHat, gGamma, xHat.dup(), -1));
        tmp.addiColumnVector(gBeta).divi(batchSize);
        INDArray gXHat =  Nd4j.getExecutioner().execAndReturn(new BroadcastSubOp(tmp, coefficients, tmp, -1));
        INDArray nextEpsilon = reshapeEp.mul(gXHat).reshape(epsilon.shape());
        Gradient retGradient = new DefaultGradient();
        retGradient.setGradientFor(BatchNormalizationParamInitializer.GAMMA, gGamma);
        retGradient.setGradientFor(BatchNormalizationParamInitializer.BETA, gBeta);
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
        org.deeplearning4j.nn.conf.layers.BatchNormalization layerConf = layerConf();
        int batchSize = x.size(0); // number examples in batch
        shape = getShape(x);
        INDArray reshapeX = (x.rank() > 2)? x.dup().reshape(batchSize, shape[1]): x;

        trainingMode = training;
        if (setMeanVar){
            this.mean = this.mean == null? Nd4j.zeros(shape): this.mean;
            this.var = this.var == null? Nd4j.valueArrayOf(shape, layerConf.getEps()): this.var;
            setMeanVar = false;
        }

        INDArray mean,var;
        if(trainingMode == TrainingMode.TRAIN && layerConf.isUseBatchMean()) {
            // mean and var over samples in batch
            mean = x.mean(0).reshape(shape);
            var = x.var(false, 0).reshape(shape);
            var.addi(layerConf.getEps());
        } else {
            // cumulative mean and var - primarily used after training
            mean = this.mean;
            var = this.var;
        }

        std = Transforms.sqrt(var);
        if(layerConf.isLockGammaBeta()){
            gamma = Nd4j.onesLike(mean);
            beta = Nd4j.zerosLike(mean);
        } else{
            gamma = getParam(BatchNormalizationParamInitializer.GAMMA);
            beta = getParam(BatchNormalizationParamInitializer.BETA);

        }

        // xHat = x-xmean / sqrt(var + epsilon)
        INDArray xMu = Nd4j.getExecutioner().execAndReturn(new BroadcastSubOp(reshapeX, mean, reshapeX, -1));
        xHat = Nd4j.getExecutioner().execAndReturn(new BroadcastDivOp(xMu, std, xMu.dup(),-1));

        // BN(xk) = γkxˆk + βk (applying gamma and beta for each activation/feature)
//        INDArray activations = Nd4j.getExecutioner().execAndReturn(new BroadcastAddOp(gamma, xHat, gamma));
        INDArray activations =  Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(xHat, gamma, xHat.dup(), -1));
        activations.addiColumnVector(beta);

        // update mean and var if using batch mean while training
        double decay;
        if(training == TrainingMode.TRAIN && layerConf.isUseBatchMean()) {
            // TODO figure out how to track finetune phase here to update decay for finetune
//          layerConf.setN(layerConf.getN() + 1);
//          decay =  1. / layerConf.getN();

            decay = layerConf.getDecay();
            double adjust = batchSize / Math.max(batchSize - 1., 1.);
            this.mean = mean.mul(decay).add(this.mean.mul(1-decay));
            this.var = var.mul(decay).add(this.var.mul((1-decay)*adjust));
        }

        return activations.reshape(x.shape());
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
    public void setListeners(Collection<IterationListener> listeners) {
        this.listeners = new ArrayList<>(listeners);
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
        if(x.rank() == 2)
            return new int[] {1, x.size(1)};
        if(x.rank() == 3) {
            int wDim = x.size(1);
            int hdim = x.size(2);
            if(x.size(0) > 1 && wDim * hdim == x.length())
                throw new IllegalArgumentException("Illegal input for batch size");
            return new int[] {1, wDim * hdim};
        }
        else if(x.rank() == 4) {
            int cDim = x.size(1);
            int wDim = x.size(2);
            int hdim = x.size(3);
            if(x.size(0) > 1 && cDim * wDim * hdim == x.length())
                throw new IllegalArgumentException("Illegal input for batch size");
            return new int[] {1, cDim * wDim * hdim};
        }

        else throw new IllegalStateException("Unable to process input of rank " + x.rank());
    }

}
