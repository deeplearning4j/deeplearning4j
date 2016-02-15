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
    protected INDArray std;
    protected int index = 0;
    protected List<IterationListener> listeners = new ArrayList<>();
    protected int[] shape;
    protected INDArray mean;
    protected INDArray var;
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
        int numExamples = epsilon.size(0); // number examples in batch

        org.deeplearning4j.nn.conf.layers.BatchNormalization layerConf = layerConf();
        INDArray gBeta = epsilon.sum(0).reshape(shape); // sum over examples in batch

        if (trainingMode == TrainingMode.TRAIN && layerConf.isUseBatchMean()){
            gGamma = epsilon.mul(xHat).sum(0).reshape(shape);
        }

        INDArray gamma = (layerConf.isLockGammaBeta())? Nd4j.onesLike(gBeta) : getParam(BatchNormalizationParamInitializer.GAMMA);
        INDArray coefficients = gamma.div(std);
        INDArray gXHat, tmpEp;
        INDArray nextEpsilon = Nd4j.zerosLike(epsilon);
        for(int i=0; i < epsilon.size(0); i++) {
            gXHat = (xHat.get(NDArrayIndex.interval(i, i + 1)).reshape(shape).mul(gGamma).add(gBeta)).divi(numExamples);
            tmpEp = epsilon.get(NDArrayIndex.interval(i, i + 1)).reshape(shape).mul(coefficients).sub(gXHat);
            nextEpsilon.putRow(i, tmpEp);
        }

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
        shape = getShape(x);
        return preOutput(x,TrainingMode.TRAIN, shape);
    }

    @Override
    public INDArray preOutput(INDArray x, TrainingMode training) {
        shape = getShape(x);
        return preOutput(x, training, shape);
    }

    public INDArray preOutput(INDArray x, TrainingMode training, int[] shape){
        this.shape = shape; //new int[]{1, x.size(1), x.size(3), x.size(3)};
        INDArray gamma, beta;
        int numExamples  = x.size(0); // number examples in batch
        trainingMode = training;
        org.deeplearning4j.nn.conf.layers.BatchNormalization layerConf = layerConf();
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

        INDArray tmpX, tmpXhat;
        INDArray activations = x.dup();
        xHat = Nd4j.zerosLike(x);
        for(int i=0; i < x.size(0); i++) {
            tmpX = activations.get(NDArrayIndex.interval(i, i + 1)).reshape(shape);
            // xHat = x-xmean / sqrt(var + epsilon)
            tmpXhat = tmpX.subi(mean).divi(std);
            xHat.putRow(i,tmpXhat);
            // BN(xk) = γkxˆk + βk
            tmpX.muli(gamma).addi(beta);
        }

        // TODO below was using broadcast but issue being able to pass in index on 0 - need to add back when fixed for performance improvement
//        INDArray xMu = Nd4j.getExecutioner().execAndReturn(new BroadcastSubOp(x, mean, x, 3));
//        xHat = Nd4j.getExecutioner().execAndReturn(new BroadcastDivOp(xMu,std, xMu.dup(),-1));
//        INDArray out = Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(xHat,gamma,xHat.dup(),-1));
//        out = Nd4j.getExecutioner().execAndReturn(new BroadcastAddOp(out,beta,out,-1));

        // update mean and var if using batch mean while training
        double decay;
        if(training == TrainingMode.TRAIN && layerConf.isUseBatchMean()) {
            // TODO figure out how to track finetune phase here to update decay for finetune
//          layerConf.setN(layerConf.getN() + 1);
//          decay =  1. / layerConf.getN();

            decay = layerConf.getDecay();
            double adjust = numExamples / Math.max(numExamples - 1., 1.);
            this.mean = mean.mul(decay).add(this.mean.mul(1-decay));
            this.var = var.mul(decay).add(this.var.mul((1-decay)*adjust));
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
            int leadDim = x.size(0);
//            int cDim = getParam(BatchNormalizationParamInitializer.GAMMA).length();
            int cDim = x.size(1);
            int rdim = (int) Math.round(((double) x.length() / ((double) leadDim * (double) cDim)));
            if(rdim < 1)
                rdim = 1;
            if(leadDim * cDim * rdim != x.length())
                throw new IllegalArgumentException("Illegal input for batch size");
            return new int[] {1, leadDim*cDim*rdim};
        }
        else if(x.rank() == 4) {
            int leadDim = x.size(0);
            int cDim = x.size(1);
            int wDim = x.size(2);
            int rdim = x.size(3);
            if(rdim < 1)
                rdim = 1;
            if(cDim * wDim * rdim == x.length())
                throw new IllegalArgumentException("Illegal input for batch size");
            return new int[] {1, cDim*wDim*rdim};
        }

        else throw new IllegalStateException("Unable to process input of rank " + x.rank());
    }

}
