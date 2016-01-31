package org.deeplearning4j.nn.layers.normalization;

import org.deeplearning4j.berkeley.Iterators;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.params.BatchNormalizationParamInitializer;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
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

import java.util.*;

/**
 * Batch normalization layer.
 * http://arxiv.org/pdf/1410.7455v8.pdf
 *
 * @author Adam Gibson
 */
public class BatchNormalization extends BaseLayer<org.deeplearning4j.nn.conf.layers.BatchNormalization> {
    protected INDArray std;
    protected int index = 0;
    protected List<IterationListener> listeners = new ArrayList<>();
    protected int[] shape;
    protected Gradient gradient;
    protected INDArray mean;
    protected INDArray var;
    protected INDArray gamma;
    protected INDArray beta;
    protected INDArray gBeta;
    protected INDArray gammaGradient;


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
        epsilon = epsilon.reshape(shape);
        int m = shape[0] * shape[2];
        org.deeplearning4j.nn.conf.layers.BatchNormalization layerConf = layerConf();

        INDArray gBeta = epsilon.sum(0); // sum over examples in batch

//        Nd4j.getExecutioner().execAndReturn(new BroadcastAddOp(gammaGradient, gBeta, gammaGradient, 0));

        // TODO broadcast mult mean to epsilon and then sum across examples OR do a loop for now
        INDArray gammaGradient;
        if(training == TrainingMode.TRAIN && layerConf.isUseBatchMean())
            gammaGradient = epsilon.mul(mean).sum(0);
        else
            gammaGradient = this.gammaGradient == null? Nd4j.zerosLike(gBeta): this.gammaGradient;

//        Nd4j.getExecutioner().execAndReturn(new BroadcastAddOp(gammaGradient,newGamma,gammaGradient,1));

        INDArray coefficients = gamma.div(std);
        // TODO epsilon needs to be broadcast OR loop for now
        INDArray ret = coefficients.mul(epsilon.sub(mean.mul(gammaGradient).add(gBeta.divi(m))));

        // Track gammaGradient
        this.gammaGradient = ?;

        ret = ret.reshape(shape);
        Gradient g = new DefaultGradient();
        this.gradient = g;
        return new Pair<>(g,ret);
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

    @Override
    public INDArray preOutput(INDArray x, TrainingMode training) {
        shape = getShape(x);
        INDArray xFlat = x;
        org.deeplearning4j.nn.conf.layers.BatchNormalization layerConf = layerConf();

        INDArray mean,var;
        if(training == TrainingMode.TRAIN && layerConf.isUseBatchMean()) {
            // mean and var over samples in batch
            mean = x.mean(0).reshape(shape);
            var = x.var(0).reshape(shape); // TODO - this is broken and not giving correct value
            var.addi(layerConf.getEps());
        } else {
            // cumulative mean and var - primarily used after training
            mean = this.mean == null? Nd4j.zeros(1, ArrayUtil.prod(shape)): this.mean;
            var = this.var == null? Nd4j.valueArrayOf(shape, layerConf.getEps()): this.var;
        }

        std = Transforms.sqrt(var);
        INDArray gamma = this.gamma == null? Nd4j.onesLike(mean): this.gamma;
        INDArray beta = this.beta == null? Nd4j.zerosLike(mean): this.beta;

        INDArray out = x.dup();

        for(int i=0; i < x.size(0); i++) {
            INDArray tmp = out.get(NDArrayIndex.interval(i, i + 1));
            // xHat = x-xmean / sqrt(var + epsilon & BN(xk) = γkxˆk + βk
            tmp.subi(mean).divi(std).muli(gamma).addi(beta);
        }

        // TODO below was using broadcast but issue being able to pass in index on 0 - need to add back when fixed for performance improvement
//        INDArray xMu = Nd4j.getExecutioner().execAndReturn(new BroadcastSubOp(x, mean, x, 3));
//        xHat = Nd4j.getExecutioner().execAndReturn(new BroadcastDivOp(xMu,std, xMu.dup(),-1));
//        INDArray out = Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(xHat,gamma,xHat.dup(),-1));
//        out = Nd4j.getExecutioner().execAndReturn(new BroadcastAddOp(out,beta,out,-1));

        // update mean and var if using batch mean
        double decay;
        if(training == TrainingMode.TRAIN && layerConf.isUseBatchMean()) {
            // should take average of all batches at the end - confirm this works
            if(layerConf.isFinetune()) {
                layerConf.setN(layerConf.getN() + 1);
                decay =  1. / layerConf.getN();
            } else
                decay = layerConf.getDecay();
            int m  = shape[0] * shape[2];
            double  adjust = m / Math.max(m - 1., 1.);
            mean.muli(decay);
            this.mean = mean.addi(mean.mul(1-decay));
            var.mul(decay);
            this.var = var.addi(var.mul((1-decay)*adjust));
        }

        // TODO update gamma and beta
        this.gamma = ?;
        this.beta = ?;

        return out.reshape(shape);
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
        if(x.rank() == 3) {
            int leadDim = x.size(0);
//            int cDim = getParam(BatchNormalizationParamInitializer.GAMMA).length();
            int cDim = x.size(1);
            int rdim = (int) Math.round(((double) x.length() / ((double) leadDim * (double) cDim)));
            if(rdim < 1)
                rdim = 1;
            if(leadDim * cDim * rdim != x.length())
                throw new IllegalArgumentException("Illegal input for batch size");
            return new int[] {leadDim,cDim,rdim};
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
            return new int[] {1, cDim,wDim,rdim};
        }

        else throw new IllegalStateException("Unable to process input of rank " + x.rank());
    }

}
