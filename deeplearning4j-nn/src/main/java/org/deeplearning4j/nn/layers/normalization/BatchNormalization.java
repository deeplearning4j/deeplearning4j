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
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

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
    protected static final Logger log = LoggerFactory.getLogger(BatchNormalization.class);

    BatchNormalizationHelper helper = null;
    protected int index = 0;
    protected List<IterationListener> listeners = new ArrayList<>();
    protected int[] shape;
    protected INDArray globalMean;
    protected INDArray globalVar;
    protected INDArray std;
    protected INDArray xMu;
    protected INDArray xHat;
    protected boolean setMeanVar = true;

    public BatchNormalization(NeuralNetConfiguration conf) {
        super(conf);
        initializeHelper();
    }

    void initializeHelper() {
        try {
            helper = Class.forName("org.deeplearning4j.nn.layers.normalization.CudnnBatchNormalizationHelper")
                    .asSubclass(BatchNormalizationHelper.class).newInstance();
            log.debug("CudnnBatchNormalizationHelper successfully loaded");
        } catch (Throwable t) {
            if (!(t instanceof ClassNotFoundException)) {
                log.warn("Could not load CudnnBatchNormalizationHelper", t);
            }
        }
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

//        INDArray gamma = (layerConf.isLockGammaBeta())? Nd4j.ones(shape) : getParam(BatchNormalizationParamInitializer.GAMMA);

        INDArray gamma = null;
        INDArray beta = null;
        if( layerConf.isLockGammaBeta()){

        } else {
            gamma = getParam(BatchNormalizationParamInitializer.GAMMA);
            beta = getParam(BatchNormalizationParamInitializer.BETA);
        }

        Gradient retGradient = new DefaultGradient();

        INDArray dGammaView = gradientViews.get(BatchNormalizationParamInitializer.GAMMA);
        INDArray dBetaView = gradientViews.get(BatchNormalizationParamInitializer.BETA);

        if (helper != null) {
            Pair<Gradient, INDArray> ret = helper.backpropGradient(input, epsilon,
                    shape, gamma, dGammaView, dBetaView, layerConf.getEps());
            if (ret != null) {
                return ret;
            }
        }


        int nOut = layerConf.getNOut();
        if (epsilon.rank() == 2) {
            //TODO: handle fixed beta/gamma case...
            INDArray dGamma = epsilon.mul(xHat).sum(0);     //dL/dGamma = sum_examples dL/dOut .* xHat
            INDArray dBeta = epsilon.sum(0);                //dL/dBeta = sum_examples dL/dOut
            INDArray dxhat = epsilon.mulRowVector(gamma);   //dL/dxHat = dL/dOut . gamma        Shape: [minibatchSize, nOut]
            int[] dxhatShape = dxhat.shape();
            if(!Arrays.equals(new int[]{batchSize,nOut}, dxhatShape)) throw new RuntimeException();

            //dL/dBatchVariance
//            INDArray dLdVar = dxhat.mul(xMu).sum(0).muli(-0.5).muli(Transforms.pow(std, -3.0/2.0, true));   //Shape: [1, miniBatch]
            INDArray dLdVar = dxhat.mul(xMu).mul(-0.5).mulRowVector(Transforms.pow(std, -3.0, true)).sum(0);   //Shape: [1, miniBatch]
            int[] dLdVarShape = dLdVar.shape();
            if(!Arrays.equals(new int[]{1,nOut}, dLdVarShape)) throw new RuntimeException();

//            System.out.println("dLdVar (backprop)");
//            System.out.println(Arrays.toString(dLdVar.data().asDouble()));

            //dL/dmu
            INDArray dxmu1 = dxhat.divRowVector(std).neg().sum(0);    //Should be equivalent to the above, but faster + one less temp array
            if(!Arrays.equals(new int[]{1,nOut}, dxmu1.shape())) throw new RuntimeException();
//            INDArray dxmu1 = dxhat.sum(0).divi(std).negi();    //Should be equivalent to the above, but faster + one less temp array
//            INDArray dxmu2 = dLdVar.mul(-2.0/batchSize).muli(xMu.sum(0));
            INDArray dxmu2 = dLdVar.mul(xMu.mul(-2.0/batchSize).sum(0));

            INDArray dLdmu = dxmu1.add(dxmu2);  //Shape: [1, nOut]
            if(!Arrays.equals(new int[]{1,nOut}, dLdmu.shape())) throw new RuntimeException();
//            System.out.println("dLdMu (backprop)");
//            System.out.println(Arrays.toString(dLdmu.data().asDouble()));

            INDArray dLdx = dxhat.divRowVector(std)
                    .add(xMu.mulRowVector(dLdVar).mul(2.0/batchSize))
                    .addRowVector(dLdmu.mul(1.0/batchSize));
            if(!Arrays.equals(new int[]{batchSize,nOut}, dLdx.shape())) throw new RuntimeException();

//            System.out.println("dLdx (backprop)");
//            System.out.println(Arrays.toString(dLdx.data().asDouble()));

            //TODO rework this to avoid the assign here
            dGammaView.assign(dGamma);
            dBetaView.assign(dBeta);

            retGradient.setGradientFor(BatchNormalizationParamInitializer.GAMMA, dGammaView);
            retGradient.setGradientFor(BatchNormalizationParamInitializer.BETA, dBetaView);

            nextEpsilon = dLdx;

//            INDArray dsq = dxhat.mul(xMu).sum(0).mul(0.5).div(Transforms.pow(std, 3)).neg().div(batchSize);
//
//            INDArray dxmu1 = dxhat.divRowVector(std);
//            INDArray dxmu2 = xMu.mul(2).mulRowVector(dsq);
//
//            INDArray dx1 = dxmu1.add(dxmu2);
//            INDArray dmu = dx1.sum(0).neg();
//            INDArray dx2 = dmu.div(batchSize);
//            nextEpsilon = Nd4j.getExecutioner().execAndReturn(new BroadcastAddOp(dx1, dx2, dx1.dup(), -1));
//
//            //alternative short calculation - does not match but more normalized epsilon
//            INDArray r = xMu.divRowVector(Transforms.pow(std, 2)).mulRowVector(epsilon.mul(xMu).sum(0));
//            INDArray otherEp = epsilon.mul(2).subRowVector(dBeta).mulRowVector(gamma.div(std.mul(2))).sub(r);
//
////            retGradient.setGradientFor(BatchNormalizationParamInitializer.GAMMA, dGamma);
////            retGradient.setGradientFor(BatchNormalizationParamInitializer.BETA, dBeta);
//
//            //TODO rework this to avoid the assign here
//            dGammaView.assign(dGamma);
//            dBetaView.assign(dBeta);
//            retGradient.setGradientFor(BatchNormalizationParamInitializer.GAMMA, dGammaView);
//            retGradient.setGradientFor(BatchNormalizationParamInitializer.BETA, dBetaView);

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
        return preOutput(input, training ? TrainingMode.TRAIN: TrainingMode.TEST);
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
        INDArray activations;
        // TODO add this directly in layer or get the layer prior...
        // batchnorm true but need to clarify if activation before or after

        org.deeplearning4j.nn.conf.layers.BatchNormalization layerConf = layerConf();
        int batchSize = x.size(0); // number examples in batch
        shape = getShape(x);


        // xHat = (x-xmean) / sqrt(var + epsilon)
        //Note that for CNNs, mean and variance are calculated per feature map (i.e., per activation) rather than per activation
        //Pg5 of http://arxiv.org/pdf/1502.03167v3.pdf
        // "For convolutional layers, we additionally want the normalization to obey the convolutional property – so that
        //  different elements of the same feature map, at different locations, are normalized in the same way. To achieve
        //  this, we jointly normalize all the activations in a minibatch, over all locations."
        INDArray mean, var;
        if (training == TrainingMode.TRAIN ) {
            switch(x.rank()){
                case 2:
                    // mean and variance over samples in batch
                    mean = x.mean(0);
                    var = x.var(false, 0);
                    break;
                case 4:
                    // mean and variance over samples AND locations
                    mean = x.mean(0,2,3);
                    var = x.var(false, 0,2,3);
                    break;
                default:
                    throw new IllegalStateException("Batch normalization on activations of rank " + x.rank() + " not supported");
            }


            var.addi(layerConf.getEps());
        } else {
            // Global mean and variance estimate - used after training
            mean = this.globalMean;
            var = this.globalVar;
        }
        std = Transforms.sqrt(var,true);

        INDArray gamma = null;
        INDArray beta = null;
        if (layerConf.isLockGammaBeta()) {
            if(helper != null){
                //TODO: don't create these each iteration, when using cudnn
                int[] gammaBetaShape = new int[]{1,layerConf().getNOut()};
                gamma = Nd4j.valueArrayOf(gammaBetaShape, layerConf().getGamma());
                beta = Nd4j.valueArrayOf(gammaBetaShape, layerConf().getBeta());
            }
        } else {
            gamma = getParam(BatchNormalizationParamInitializer.GAMMA);
            beta = getParam(BatchNormalizationParamInitializer.BETA);
        }

        if (helper != null) {
            double decay = setMeanVar ? 1 : layerConf.getDecay();
            if (setMeanVar){
                this.globalMean = this.globalMean == null? Nd4j.zeros(mean.shape()): this.globalMean;
                this.globalVar = this.globalVar == null? Nd4j.ones(var.shape()): this.globalVar;
                setMeanVar = false;
            }
            INDArray ret = helper.preOutput(x, training == TrainingMode.TRAIN,
                    shape, gamma, beta, this.globalMean, this.globalVar, decay, layerConf.getEps());
            if (ret != null) {
                return ret;
            }
        }

        // BN(xk) = gamma*xˆ + β (applying gamma and beta for each activation)
        if (x.rank() == 2) {
            xMu = x.subRowVector(mean);
            xHat = xMu.divRowVector(std);

            if(layerConf.isLockGammaBeta()){
                //Special case: gamma/beta have fixed values for all outputs
                //Use mul/addi(Number) here to avoid allocating temp arrays of all same value
                double g = layerConf.getGamma();
                double b = layerConf.getBeta();
                if(g != 1.0 && b != 0.0){
                    //Default and most common case: 1.0 and 0.0 for these parameters. No point executing 1 * x + 0 op
                    activations = xHat.mul(g).addi(b);
                } else {
                    activations = xHat;
                }
            } else {
                //Standard case: gamma and beta are learned per parameter
                activations = xHat.mulRowVector(gamma).addiRowVector(beta);
            }
        } else if (x.rank() == 4) {
            xMu = Nd4j.getExecutioner().execAndReturn(new BroadcastSubOp(x, mean, Nd4j.createUninitialized(x.shape(), x.ordering()), 1));
            xHat = Nd4j.getExecutioner().execAndReturn(new BroadcastDivOp(xMu, std, Nd4j.createUninitialized(x.shape(), x.ordering()), 1));

            if(layerConf.isLockGammaBeta()){
                //Special case: gamma/beta have fixed values for all outputs
                //Use mul/addi(Number) here to avoid allocating temp arrays of all same value
                double g = layerConf.getGamma();
                double b = layerConf.getBeta();
                if(g != 1.0 && b != 0.0){
                    //Default and most common case: 1.0 and 0.0 for these parameters. No point executing 1 * x + 0 op
                    activations = xHat.mul(g).addi(b);
                } else {
                    activations = xHat;
                }
            } else {
                //Standard case: gamma and beta are learned per parameter
                activations = Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(xHat,gamma,Nd4j.createUninitialized(x.shape(),x.ordering()),1));
                activations = Nd4j.getExecutioner().execAndReturn(new BroadcastAddOp(activations,beta,activations,1));
            }
        } else {
            // TODO setup BatchNorm for RNN http://arxiv.org/pdf/1510.01378v1.pdf
            throw new IllegalStateException("The layer prior to BatchNorm in the configuration is not currently supported.");
        }

        // store mean and var if using batch mean while training
        double decay;
        if(training == TrainingMode.TRAIN ) {
            // TODO track finetune phase here to update decay for finetune
//          layerConf.setN(layerConf.getN() + 1);
//          decay =  1. / layerConf.getN();

            if (setMeanVar){
                this.globalMean = this.globalMean == null? Nd4j.zeros(mean.shape()): this.globalMean;
                this.globalVar = this.globalVar == null? Nd4j.ones(var.shape()): this.globalVar;
                setMeanVar = false;
            }

            if(layerConf.isMinibatch()){
                //Standard case: Estimate global mean and variance stats by moving average
                //globalMean = decay * globalMean + (1-decay) * minibatchMean
                //globalVar  = decay * globalVar  + (1-decay) * minibatchVar
                //Note that it's safe to do a muli on 'mean' and 'var' variables: can't be the global arrays with training == Trainingmode.TRAIN
                decay = layerConf.getDecay();
                this.globalMean.muli(decay).addi(mean.muli(1-decay));
                this.globalVar.muli(decay).addi(var.muli(1-decay));

            } else {
                //Special case: doing full-batch (entire data set) training (uncommon; only tiny data sets)
                //In this case, minibatch and global stats are identical. Don't want to use a moving average estimate.
                this.globalMean = mean;
                this.globalVar = var;
            }
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
