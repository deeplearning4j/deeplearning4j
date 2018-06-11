package org.deeplearning4j.nn.layers.normalization;

import lombok.val;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.AbstractLayer;
import org.deeplearning4j.nn.layers.LayerHelper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.OldMulOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.nd4j.linalg.primitives.Triple;
import org.nd4j.util.OneTimeLogger;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Properties;

import static org.nd4j.linalg.indexing.NDArrayIndex.interval;

/**
 * Deep neural net normalization approach normalizes activations between layers
 * "brightness normalization"
 * Used for nets like AlexNet
 * <p>
 * For a^i_{x,y} the activity of a neuron computed by applying kernel i
 * at position (x,y) and applying ReLU nonlinearity, the response
 * normalized activation b^i_{x,y} is given by:
 * <p>
 * x^2 = (a^j_{x,y})^2
 * unitScale = (k + alpha * sum_{j=max(0, i - n/2)}^{max(N-1, i + n/2)} (a^j_{x,y})^2 )
 * y = b^i_{x,y} = x * unitScale**-beta
 * <p>
 * gy = epsilon (aka deltas from previous layer)
 * sumPart = sum(a^j_{x,y} * gb^j_{x,y})
 * gx = gy * unitScale**-beta - 2 * alpha * beta * sumPart/unitScale * a^i_{x,y}
 * <p>
 * Reference:
 * http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf
 * https://github.com/vlfeat/matconvnet/issues/10
 * Chainer
 * <p>
 * Created by nyghtowl on 10/29/15.
 */
public class LocalResponseNormalization
                extends AbstractLayer<org.deeplearning4j.nn.conf.layers.LocalResponseNormalization> {
    protected static final Logger log =
                    LoggerFactory.getLogger(org.deeplearning4j.nn.conf.layers.LocalResponseNormalization.class);

    LocalResponseNormalizationHelper helper = null;

    public LocalResponseNormalization(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
        initializeHelper();
    }

    @Override
    public Layer clone() {
        return new LocalResponseNormalization(conf.clone());
    }

    public LocalResponseNormalization(NeuralNetConfiguration conf) {
        super(conf);
        initializeHelper();
    }

    void initializeHelper() {
        String backend = Nd4j.getExecutioner().getEnvironmentInformation().getProperty("backend");
        if("CUDA".equalsIgnoreCase(backend)) {
            try {
                helper = Class.forName("org.deeplearning4j.nn.layers.normalization.CudnnLocalResponseNormalizationHelper")
                        .asSubclass(LocalResponseNormalizationHelper.class).newInstance();
                log.debug("CudnnLocalResponseNormalizationHelper successfully initialized");
                if (!helper.checkSupported(layerConf().getK(), layerConf().getN(), layerConf().getAlpha(),
                        layerConf().getBeta())) {
                    helper = null;
                }
            } catch (Throwable t) {
                if (!(t instanceof ClassNotFoundException)) {
                    log.warn("Could not initialize CudnnLocalResponseNormalizationHelper", t);
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
    public void fit(INDArray input, LayerWorkspaceMgr workspaceMgr) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);

        double k = layerConf().getK();
        double n = layerConf().getN();
        double alpha = layerConf().getAlpha();
        double beta = layerConf().getBeta();
        int halfN = (int) n / 2;

        if (helper != null) {
            Pair<Gradient, INDArray> ret = helper.backpropGradient(input, epsilon, k, n, alpha, beta, workspaceMgr);
            if (ret != null) {
                return ret;
            }
        }

        Triple<INDArray,INDArray,INDArray> triple = activateHelper(true, workspaceMgr, true);
        INDArray activations = triple.getFirst();
        INDArray unitScale = triple.getSecond();
        INDArray scale = triple.getThird();

        val channel = input.size(1);
        INDArray tmp, addVal;
        Gradient retGradient = new DefaultGradient();
        INDArray reverse = activations.mul(epsilon);
        INDArray sumPart = reverse.dup();

        // sumPart = sum(a^j_{x,y} * gb^j_{x,y})
        for (int i = 1; i < halfN + 1; i++) {
            tmp = sumPart.get(NDArrayIndex.all(), interval(i, channel), NDArrayIndex.all(), NDArrayIndex.all());
            addVal = reverse.get(NDArrayIndex.all(), interval(0, channel - i), NDArrayIndex.all(), NDArrayIndex.all());
            sumPart.put(new INDArrayIndex[] {NDArrayIndex.all(), interval(i, channel), NDArrayIndex.all(),
                            NDArrayIndex.all()}, tmp.addi(addVal));

            tmp = sumPart.get(NDArrayIndex.all(), interval(0, channel - i), NDArrayIndex.all(), NDArrayIndex.all());
            addVal = reverse.get(NDArrayIndex.all(), interval(i, channel), NDArrayIndex.all(), NDArrayIndex.all());
            sumPart.put(new INDArrayIndex[] {NDArrayIndex.all(), interval(0, channel - i), NDArrayIndex.all(),
                            NDArrayIndex.all()}, tmp.addi(addVal));
        }

        // gx = gy * unitScale**-beta - 2 * alpha * beta * sumPart/unitScale * a^i_{x,y}    - rearranged for more in-place ops
        INDArray nextEpsilon = workspaceMgr.createUninitialized(ArrayType.ACTIVATION_GRAD, epsilon.shape(), epsilon.ordering());
        Nd4j.getExecutioner().exec(new OldMulOp(epsilon, scale, nextEpsilon));
        nextEpsilon.subi(sumPart.muli(input).divi(unitScale).muli(2 * alpha * beta));
        return new Pair<>(retGradient, nextEpsilon);
    }

    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        return activateHelper(training, workspaceMgr, false).getFirst();
    }

    private Triple<INDArray,INDArray,INDArray> activateHelper(boolean training, LayerWorkspaceMgr workspaceMgr, boolean forBackprop){
        assertInputSet(false);
        double k = layerConf().getK();
        double n = layerConf().getN();
        double alpha = layerConf().getAlpha();
        double beta = layerConf().getBeta();
        int halfN = (int) n / 2;

        if (helper != null) {
            INDArray activations = helper.activate(input, training, k, n, alpha, beta, workspaceMgr);
            if (activations != null) {
                return new Triple<>(activations, null, null);
            }
        }

        val channel = input.size(1);
        INDArray tmp, addVal;
        // x^2 = (a^j_{x,y})^2
        INDArray activitySqr = input.mul(input);
        INDArray sumPart = activitySqr.dup();

        //sum_{j=max(0, i - n/2)}^{max(N-1, i + n/2)} (a^j_{x,y})^2 )
        for (int i = 1; i < halfN + 1; i++) {
            tmp = sumPart.get(NDArrayIndex.all(), interval(i, channel), NDArrayIndex.all(), NDArrayIndex.all());
            addVal = activitySqr.get(NDArrayIndex.all(), interval(0, channel - i), NDArrayIndex.all(),
                            NDArrayIndex.all());
            sumPart.put(new INDArrayIndex[] {NDArrayIndex.all(), interval(i, channel), NDArrayIndex.all(),
                            NDArrayIndex.all()}, tmp.addi(addVal));

            tmp = sumPart.get(NDArrayIndex.all(), interval(0, channel - i), NDArrayIndex.all(), NDArrayIndex.all());
            addVal = activitySqr.get(NDArrayIndex.all(), interval(i, channel), NDArrayIndex.all(), NDArrayIndex.all());
            sumPart.put(new INDArrayIndex[] {NDArrayIndex.all(), interval(0, channel - i), NDArrayIndex.all(),
                            NDArrayIndex.all()}, tmp.addi(addVal));
        }

        INDArray unitScale = null;
        INDArray scale = null;
        INDArray activations = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, input.shape(), input.ordering());
        if(forBackprop) {
            // unitScale = (k + alpha * sum_{j=max(0, i - n/2)}^{max(N-1, i + n/2)} (a^j_{x,y})^2 )
            unitScale = sumPart.mul(alpha).addi(k);
            // y = x * unitScale**-beta
            scale = Transforms.pow(unitScale, -beta, true);
            Nd4j.getExecutioner().exec(new OldMulOp(input, scale, activations));
        } else {
            // unitScale = (k + alpha * sum_{j=max(0, i - n/2)}^{max(N-1, i + n/2)} (a^j_{x,y})^2 )
            sumPart.muli(alpha, activations).addi(k);
            Transforms.pow(activations, -beta, false);
            activations.muli(input);
        }
        if(forBackprop){
            return new Triple<>(activations, unitScale, scale);
        } else {
            return new Triple<>(activations, null, null);
        }
    }

    @Override
    public Layer transpose() {
        throw new UnsupportedOperationException("Not supported " + layerId());
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }

    @Override
    public void clearNoiseWeightParams() {
        //No op
    }

    @Override
    public LayerHelper getHelper() {
        return helper;
    }

    @Override
    public INDArray params() {
        return null;
    }

    @Override
    public INDArray getParam(String param) {
        return params();
    }

    @Override
    public void setParams(INDArray params) {

    }
}
