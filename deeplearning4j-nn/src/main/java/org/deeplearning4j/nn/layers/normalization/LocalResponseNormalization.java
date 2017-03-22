package org.deeplearning4j.nn.layers.normalization;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
                extends BaseLayer<org.deeplearning4j.nn.conf.layers.LocalResponseNormalization> {
    protected static final Logger log =
                    LoggerFactory.getLogger(org.deeplearning4j.nn.conf.layers.LocalResponseNormalization.class);

    LocalResponseNormalizationHelper helper = null;

    private double k;
    private double n;
    private double alpha;
    private double beta;
    private int halfN;
    private INDArray activations, unitScale, scale;

    public LocalResponseNormalization(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
        initializeHelper();
    }

    public LocalResponseNormalization(NeuralNetConfiguration conf) {
        super(conf);
        initializeHelper();
    }

    void initializeHelper() {
        try {
            helper = Class.forName("org.deeplearning4j.nn.layers.normalization.CudnnLocalResponseNormalizationHelper")
                            .asSubclass(LocalResponseNormalizationHelper.class).newInstance();
            log.debug("CudnnLocalResponseNormalizationHelper successfully loaded");
        } catch (Throwable t) {
            if (!(t instanceof ClassNotFoundException)) {
                log.warn("Could not load CudnnLocalResponseNormalizationHelper", t);
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
    public void fit(INDArray input) {}

    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon) {
        if (helper != null) {
            Pair<Gradient, INDArray> ret = helper.backpropGradient(input, epsilon, k, n, alpha, beta);
            if (ret != null) {
                return ret;
            }
        }

        int channel = input.size(1);
        INDArray tmp, addVal;
        Gradient retGradient = new DefaultGradient();
        INDArray reverse = activations.mul(epsilon);
        INDArray sumPart = reverse.dup();

        // sumPart = sum(a^j_{x,y} * gb^j_{x,y})
        for (int i = 1; i < halfN + 1; i++) {
            tmp = sumPart.get(new INDArrayIndex[] {NDArrayIndex.all(), interval(i, channel), NDArrayIndex.all(),
                            NDArrayIndex.all()});
            addVal = reverse.get(new INDArrayIndex[] {NDArrayIndex.all(), interval(0, channel - i), NDArrayIndex.all(),
                            NDArrayIndex.all()});
            sumPart.put(new INDArrayIndex[] {NDArrayIndex.all(), interval(i, channel), NDArrayIndex.all(),
                            NDArrayIndex.all()}, tmp.addi(addVal));

            tmp = sumPart.get(new INDArrayIndex[] {NDArrayIndex.all(), interval(0, channel - i), NDArrayIndex.all(),
                            NDArrayIndex.all()});
            addVal = reverse.get(new INDArrayIndex[] {NDArrayIndex.all(), interval(i, channel), NDArrayIndex.all(),
                            NDArrayIndex.all()});
            sumPart.put(new INDArrayIndex[] {NDArrayIndex.all(), interval(0, channel - i), NDArrayIndex.all(),
                            NDArrayIndex.all()}, tmp.addi(addVal));
        }

        // gx = gy * unitScale**-beta - 2 * alpha * beta * sumPart/unitScale * a^i_{x,y}    - rearranged for more in-place ops
        INDArray nextEpsilon = epsilon.mul(scale).subi(sumPart.muli(input).divi(unitScale).muli(2 * alpha * beta));
        return new Pair<>(retGradient, nextEpsilon);
    }

    @Override
    public INDArray activate(boolean training) {
        k = layerConf().getK();
        n = layerConf().getN();
        alpha = layerConf().getAlpha();
        beta = layerConf().getBeta();
        halfN = (int) n / 2;

        if (helper != null) {
            activations = helper.activate(input, training, k, n, alpha, beta);
            if (activations != null) {
                return activations;
            }
        }

        int channel = input.size(1);
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

        // unitScale = (k + alpha * sum_{j=max(0, i - n/2)}^{max(N-1, i + n/2)} (a^j_{x,y})^2 )
        unitScale = sumPart.mul(alpha).addi(k);
        // y = x * unitScale**-beta
        scale = Transforms.pow(unitScale, -beta);
        activations = input.mul(scale);
        return activations;
    }

    @Override
    public Layer transpose() {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }


    @Override
    public Gradient calcGradient(Gradient layerError, INDArray indArray) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public void merge(Layer layer, int batchSize) {
        throw new UnsupportedOperationException();
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
