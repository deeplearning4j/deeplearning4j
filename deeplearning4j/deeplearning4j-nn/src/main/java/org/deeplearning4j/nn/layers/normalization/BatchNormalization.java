package org.deeplearning4j.nn.layers.normalization;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.layers.LayerHelper;
import org.deeplearning4j.nn.params.BatchNormalizationParamInitializer;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastAddOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastDivOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastSubOp;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.util.OneTimeLogger;

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
@Slf4j
public class BatchNormalization extends BaseLayer<org.deeplearning4j.nn.conf.layers.BatchNormalization> {

    BatchNormalizationHelper helper = null;
    protected int index = 0;
    protected List<TrainingListener> listeners = new ArrayList<>();
    protected INDArray std;
    protected INDArray xMu;
    protected INDArray xHat;

    public BatchNormalization(NeuralNetConfiguration conf) {
        super(conf);
        initializeHelper();
    }

    void initializeHelper() {
        String backend = Nd4j.getExecutioner().getEnvironmentInformation().getProperty("backend");
        if("CUDA".equalsIgnoreCase(backend)) {
            try {
                helper = Class.forName("org.deeplearning4j.nn.layers.normalization.CudnnBatchNormalizationHelper")
                        .asSubclass(BatchNormalizationHelper.class).newInstance();
                log.debug("CudnnBatchNormalizationHelper successfully initialized");
                if (!helper.checkSupported(layerConf().getEps())) {
                    helper = null;
                }
            } catch (Throwable t) {
                if (!(t instanceof ClassNotFoundException)) {
                    log.warn("Could not initialize CudnnBatchNormalizationHelper", t);
                } else {
                    OneTimeLogger.info(log, "cuDNN not found: "
                            + "use cuDNN for better GPU performance by including the deeplearning4j-cuda module. "
                            + "For more information, please refer to: https://deeplearning4j.org/cudnn", t);
                }
            }
        }
    }

    @Override
    public double calcL2(boolean backpropParamsOnly) {
        return 0;
    }

    @Override
    public double calcL1(boolean backpropParamsOnly) {
        return 0;
    }

    @Override
    public Type type() {
        return Type.NORMALIZATION;
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);
        INDArray nextEpsilon;
        val shape = getShape(epsilon);
        val batchSize = epsilon.size(0); // number examples in batch
        org.deeplearning4j.nn.conf.layers.BatchNormalization layerConf = layerConf();

        INDArray gamma = null;
        INDArray dGammaView;
        INDArray dBetaView;
        INDArray dGlobalMeanView = gradientViews.get(BatchNormalizationParamInitializer.GLOBAL_MEAN);
        INDArray dGlobalVarView = gradientViews.get(BatchNormalizationParamInitializer.GLOBAL_VAR);
        if (layerConf.isLockGammaBeta()) {
            val tempShape = new long[] {1, shape[1]};
            dGammaView = Nd4j.createUninitialized(tempShape, 'c');
            dBetaView = Nd4j.createUninitialized(tempShape, 'c');
        } else {
            gamma = getParam(BatchNormalizationParamInitializer.GAMMA);
            dGammaView = gradientViews.get(BatchNormalizationParamInitializer.GAMMA);
            dBetaView = gradientViews.get(BatchNormalizationParamInitializer.BETA);
        }

        Gradient retGradient = new DefaultGradient();



        if (helper != null && epsilon.rank() == 4) {
            //Note that cudnn does not support dense (2d) batch norm case as of v5.1
            if (layerConf.isLockGammaBeta()) {
                gamma = Nd4j.valueArrayOf(new long[] {1, shape[1]}, layerConf.getGamma());
            }
            // FIXME: int cast
            Pair<Gradient, INDArray> ret = helper.backpropGradient(input, epsilon, ArrayUtil.toInts(shape), gamma, dGammaView, dBetaView,
                            layerConf.getEps(), workspaceMgr);
            if (ret != null) {
                ret.getFirst().setGradientFor(BatchNormalizationParamInitializer.GLOBAL_MEAN, dGlobalMeanView);
                ret.getFirst().setGradientFor(BatchNormalizationParamInitializer.GLOBAL_VAR, dGlobalVarView);
                return ret;
            }
        }


        if (epsilon.rank() == 2) {
            //TODO: handle fixed beta/gamma case...
            INDArray dBeta = epsilon.sum(0); //dL/dBeta = sum_examples dL/dOut
            INDArray dGamma = epsilon.mul(xHat).sum(0); //dL/dGamma = sum_examples dL/dOut .* xHat
            INDArray dxhat;
            if (layerConf.isLockGammaBeta()) {
                dxhat = epsilon.mul(layerConf.getGamma());
            } else {
                //Standard case
                dxhat = epsilon.mulRowVector(gamma); //dL/dxHat = dL/dOut . gamma        Shape: [minibatchSize, nOut]
            }


            //dL/dVariance
            INDArray dLdVar = dxhat.mul(xMu).sum(0).muli(-0.5).muli(Transforms.pow(std, -3.0, true)); //Shape: [1, miniBatch]

            //dL/dmu
            INDArray dxmu1 = dxhat.sum(0).divi(std).negi();
            INDArray dxmu2 = xMu.sum(0).muli(-2.0 / batchSize).muli(dLdVar);

            INDArray dLdmu = dxmu1.addi(dxmu2); //Shape: [1, nOut]

            //Note the array reuse here: dxhat, xMu, dLdVar, dLdmu - all are invalid after this line (but aren't used later anyway)
            INDArray dLdx = dxhat.diviRowVector(std).addi(xMu.muliRowVector(dLdVar.muli(2.0 / batchSize)))
                            .addiRowVector(dLdmu.muli(1.0 / batchSize));

            //TODO rework this to avoid the assign here
            dGammaView.assign(dGamma);
            dBetaView.assign(dBeta);

            retGradient.setGradientFor(BatchNormalizationParamInitializer.GAMMA, dGammaView);
            retGradient.setGradientFor(BatchNormalizationParamInitializer.BETA, dBetaView);
            //TODO: do this properly
            dGlobalMeanView.assign(0);
            dGlobalVarView.assign(0);
            retGradient.setGradientFor(BatchNormalizationParamInitializer.GLOBAL_MEAN, dGlobalMeanView);
            retGradient.setGradientFor(BatchNormalizationParamInitializer.GLOBAL_VAR, dGlobalVarView);

            nextEpsilon = dLdx;

        } else if (epsilon.rank() == 4) {
            INDArray dBeta = epsilon.sum(0, 2, 3);
            INDArray dGamma = epsilon.mul(xHat).sum(0, 2, 3);
            INDArray dxhat;
            if (layerConf.isLockGammaBeta()) {
                dxhat = epsilon.mul(layerConf.getGamma());
            } else {
                //Standard case
                dxhat = Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(epsilon, gamma,
                                Nd4j.createUninitialized(epsilon.shape(), epsilon.ordering()), 1));
            }

            //dL/dVariance
            INDArray dLdVar = dxhat.mul(xMu).sum(0, 2, 3).muli(-0.5).muli(Transforms.pow(std, -3.0, true));

            //dL/dmu
            val effectiveBatchSize = input.size(0) * input.size(2) * input.size(3);
            INDArray dxmu1 = dxhat.sum(0, 2, 3).divi(std).negi();
            INDArray dxmu2 = xMu.sum(0, 2, 3).muli(-2.0 / effectiveBatchSize).muli(dLdVar);
            INDArray dLdmu = dxmu1.addi(dxmu2);

            INDArray dLdx = Nd4j.getExecutioner().execAndReturn(new BroadcastDivOp(dxhat, std, dxhat, 1))
                            .addi(Nd4j.getExecutioner().execAndReturn(
                                            new BroadcastMulOp(xMu, dLdVar.muli(2.0 / effectiveBatchSize), xMu, 1)));
            Nd4j.getExecutioner()
                            .execAndReturn(new BroadcastAddOp(dLdx, dLdmu.muli(1.0 / effectiveBatchSize), dLdx, 1));

            //TODO rework this to avoid the assign here
            dGammaView.assign(dGamma);
            dBetaView.assign(dBeta);

            retGradient.setGradientFor(BatchNormalizationParamInitializer.GAMMA, dGammaView);
            retGradient.setGradientFor(BatchNormalizationParamInitializer.BETA, dBetaView);
            //TODO: do this properly
            dGlobalMeanView.assign(0);
            dGlobalVarView.assign(0);
            retGradient.setGradientFor(BatchNormalizationParamInitializer.GLOBAL_MEAN, dGlobalMeanView);
            retGradient.setGradientFor(BatchNormalizationParamInitializer.GLOBAL_VAR, dGlobalVarView);

            nextEpsilon = dLdx;
        } else {
            // TODO setup BatchNorm for RNN http://arxiv.org/pdf/1510.01378v1.pdf
            throw new IllegalStateException(
                            "The layer prior to BatchNorm in the configuration is not currently supported. "
                                            + layerId());
        }

        //TODO could optimize this
        nextEpsilon = workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, nextEpsilon);
        return new Pair<>(retGradient, nextEpsilon);
    }

    @Override
    public void fit(INDArray input, LayerWorkspaceMgr workspaceMgr) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(false);
        return preOutput(input, training ? TrainingMode.TRAIN : TrainingMode.TEST, workspaceMgr);
    }

    @Override
    public Gradient gradient() {
        return gradient;
    }

    public INDArray preOutput(INDArray x, TrainingMode training, LayerWorkspaceMgr workspaceMgr) {
        INDArray activations;
        // TODO add this directly in layer or get the layer prior...
        // batchnorm true but need to clarify if activation before or after

        org.deeplearning4j.nn.conf.layers.BatchNormalization layerConf = layerConf();
        val shape = getShape(x);

        INDArray gamma = null;
        INDArray beta = null;
        INDArray globalMeanView = getParam(BatchNormalizationParamInitializer.GLOBAL_MEAN);
        INDArray globalVarView = getParam(BatchNormalizationParamInitializer.GLOBAL_VAR);
        if (layerConf.isLockGammaBeta()) {
            if (helper != null && input.rank() == 4) {
                //TODO: don't create these each iteration, when using cudnn
                val gammaBetaShape = new long[] {1, layerConf().getNOut()};
                gamma = Nd4j.valueArrayOf(gammaBetaShape, layerConf().getGamma());
                beta = Nd4j.valueArrayOf(gammaBetaShape, layerConf().getBeta());
            }
        } else {
            gamma = getParam(BatchNormalizationParamInitializer.GAMMA);
            beta = getParam(BatchNormalizationParamInitializer.BETA);
        }

        if (helper != null && input.rank() == 4) {
            //Note that cudnn does not support dense (2d) batch norm case as of v7.1
            double decay = layerConf.getDecay();

            // FIXME: int cast
            INDArray ret = helper.preOutput(x, training == TrainingMode.TRAIN, ArrayUtil.toInts(shape), gamma, beta, globalMeanView,
                            globalVarView, decay, layerConf.getEps(), workspaceMgr);
            if (ret != null) {
                return ret;
            }
        }

        // xHat = (x-xmean) / sqrt(var + epsilon)
        //Note that for CNNs, mean and variance are calculated per feature map (i.e., per activation) rather than per activation
        //Pg5 of http://arxiv.org/pdf/1502.03167v3.pdf
        // "For convolutional layers, we additionally want the normalization to obey the convolutional property – so that
        //  different elements of the same feature map, at different locations, are normalized in the same way. To achieve
        //  this, we jointly normalize all the activations in a minibatch, over all locations."
        INDArray mean, var;
        if (training == TrainingMode.TRAIN) {
            switch (x.rank()) {
                case 2:
                    // mean and variance over samples in batch
                    mean = x.mean(0);
                    var = x.var(false, 0);
                    break;
                case 4:
                    // mean and variance over samples AND locations
                    mean = x.mean(0, 2, 3);
                    var = x.var(false, 0, 2, 3);
                    break;
                default:
                    throw new IllegalStateException("Batch normalization on activations of rank " + x.rank()
                            + " not supported " + layerId());
            }


            var.addi(layerConf.getEps());
        } else {
            // Global mean and variance estimate - used after training
            mean = getParam(BatchNormalizationParamInitializer.GLOBAL_MEAN);
            var = getParam(BatchNormalizationParamInitializer.GLOBAL_VAR);
        }
        std = Transforms.sqrt(workspaceMgr.dup(ArrayType.INPUT, var), false);

        // BN(xk) = gamma*xˆ + β (applying gamma and beta for each activation)
        if (x.rank() == 2) {
            xMu = workspaceMgr.leverageTo(ArrayType.INPUT, x.subRowVector(mean));
            xHat = workspaceMgr.leverageTo(ArrayType.INPUT, xMu.divRowVector(std));


            if (layerConf.isLockGammaBeta()) {
                //Special case: gamma/beta have fixed values for all outputs
                //Use mul/addi(Number) here to avoid allocating temp arrays of all same value
                double g = layerConf.getGamma();
                double b = layerConf.getBeta();
                if (g != 1.0 && b != 0.0) {
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
            if (!Shape.strideDescendingCAscendingF(x))
                x = x.dup(); //TODO: temp Workaround for broadcast bug. To be removed when fixed
            xMu = workspaceMgr.createUninitialized(ArrayType.INPUT, x.shape(), x.ordering());
            xMu = Nd4j.getExecutioner().execAndReturn(new BroadcastSubOp(x, mean,xMu, 1));
            xHat =  workspaceMgr.createUninitialized(ArrayType.INPUT, x.shape(), x.ordering());
            xHat = Nd4j.getExecutioner().execAndReturn(new BroadcastDivOp(xMu, std,xHat, 1));

            if (layerConf.isLockGammaBeta()) {
                //Special case: gamma/beta have fixed values for all outputs
                //Use mul/addi(Number) here to avoid allocating temp arrays of all same value
                double g = layerConf.getGamma();
                double b = layerConf.getBeta();
                if (g != 1.0 && b != 0.0) {
                    //Default and most common case: 1.0 and 0.0 for these parameters. No point executing 1 * x + 0 op
                    activations = xHat.mul(g).addi(b);
                } else {
                    activations = xHat;
                }
            } else {
                //Standard case: gamma and beta are learned per parameter
                activations = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, x.shape(), x.ordering());
                activations = Nd4j.getExecutioner().execAndReturn(
                                new BroadcastMulOp(xHat, gamma, activations, 1));
                activations = Nd4j.getExecutioner()
                                .execAndReturn(new BroadcastAddOp(activations, beta, activations, 1));
            }
        } else {
            // TODO setup BatchNorm for RNN http://arxiv.org/pdf/1510.01378v1.pdf
            throw new IllegalStateException(
                            "The layer prior to BatchNorm in the configuration is not currently supported. "
                                            + layerId());
        }

        // store mean and var if using batch mean while training
        double decay;
        if (training == TrainingMode.TRAIN) {
            // TODO track finetune phase here to update decay for finetune
            //          layerConf.setN(layerConf.getN() + 1);
            //          decay =  1. / layerConf.getN();

            //            if (setMeanVar){
            //                this.globalMean = this.globalMean == null? Nd4j.zeros(mean.shape()): this.globalMean;
            //                this.globalVar = this.globalVar == null? Nd4j.ones(var.shape()): this.globalVar;
            //                setMeanVar = false;
            //            }

            if (layerConf.isMinibatch()) {
                //Standard case: Estimate global mean and variance stats by moving average
                //globalMean = decay * globalMean + (1-decay) * minibatchMean
                //globalVar  = decay * globalVar  + (1-decay) * minibatchVar
                //Note that it's safe to do a muli on 'mean' and 'var' variables: can't be the global arrays with training == Trainingmode.TRAIN
                decay = layerConf.getDecay();
                globalMeanView.muli(decay).addi(mean.muli(1 - decay));
                globalVarView.muli(decay).addi(var.muli(1 - decay));

            } else {
                //Special case: doing full-batch (entire data set) training (uncommon; only tiny data sets)
                //In this case, minibatch and global stats are identical. Don't want to use a moving average estimate.
                globalMeanView.assign(mean);
                globalVarView.assign(var);
            }
        }

        activations = workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, activations);   //Most of the time this should be a no-op
        return activations;
    }

    @Override
    public Layer transpose() {
        throw new UnsupportedOperationException(layerId());

    }

    @Override
    public Layer clone() {
        throw new UnsupportedOperationException(layerId());

    }

    @Override
    public Collection<TrainingListener> getListeners() {
        return listeners;
    }

    @Override
    public void setListeners(TrainingListener... listeners) {
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

    @Override
    public boolean isPretrainLayer() {
        return false;
    }

    @Override
    public LayerHelper getHelper() {
        return helper;
    }

    public long[] getShape(INDArray x) {
        if (x.rank() == 2 || x.rank() == 4)
            return new long[] {1, x.size(1)};
        if (x.rank() == 3) {
            val wDim = x.size(1);
            val hdim = x.size(2);
            if (x.size(0) > 1 && wDim * hdim == x.length())
                throw new IllegalArgumentException("Illegal input for batch size " + layerId());
            return new long[] {1, wDim * hdim};
        } else
            throw new IllegalStateException("Unable to process input of rank " + x.rank() + " " + layerId());
    }

}
