package org.deeplearning4j.nn.layers.normalization;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.activations.Activations;
import org.deeplearning4j.nn.api.activations.ActivationsFactory;
import org.deeplearning4j.nn.api.gradients.Gradients;
import org.deeplearning4j.nn.api.gradients.GradientsFactory;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.AbstractLayer;
import org.deeplearning4j.util.OneTimeLogger;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
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
@Slf4j
public class LocalResponseNormalization
                extends AbstractLayer<org.deeplearning4j.nn.conf.layers.LocalResponseNormalization> {

    LocalResponseNormalizationHelper helper = null;

    private double k;
    private double n;
    private double alpha;
    private double beta;
    private int halfN;
    private INDArray activations, unitScale, scale;

    @Override
    public Layer clone() {
        return new LocalResponseNormalization((org.deeplearning4j.nn.conf.layers.LocalResponseNormalization)conf.clone());
    }

    public LocalResponseNormalization(org.deeplearning4j.nn.conf.layers.LocalResponseNormalization conf) {
        super(conf);
        initializeHelper();
    }

    void initializeHelper() {
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
                Properties p = Nd4j.getExecutioner().getEnvironmentInformation();
                if (p.getProperty("backend").equals("CUDA")) {
                    OneTimeLogger.info(log, "cuDNN not found: "
                                    + "use cuDNN for better GPU performance by including the deeplearning4j-cuda module. "
                                    + "For more information, please refer to: https://deeplearning4j.org/cudnn", t);
                }
            }
        }
    }

    public Gradients backpropGradient(Gradients gradients) {
        INDArray epsilon = gradients.get(0);
        if (helper != null) {
            Gradients ret = helper.backpropGradient(input.get(0), epsilon, k, n, alpha, beta);
            if (ret != null) {
                return ret;
            }
        }

        int channel = input.get(0).size(1);
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
        INDArray nextEpsilon = epsilon.mul(scale).subi(sumPart.muli(input.get(0)).divi(unitScale).muli(2 * alpha * beta));
        Gradients g = GradientsFactory.getInstance().create(nextEpsilon, retGradient);
        return backpropPreprocessor(g);
    }

    @Override
    public Activations activate(boolean training) {
        k = layerConf().getK();
        n = layerConf().getN();
        alpha = layerConf().getAlpha();
        beta = layerConf().getBeta();
        halfN = (int) n / 2;

        if (helper != null) {
            activations = helper.activate(input.get(0), training, k, n, alpha, beta);
            if (activations != null) {
                return ActivationsFactory.getInstance().create(activations);
            }
        }

        int channel = input.get(0).size(1);
        INDArray tmp, addVal;
        // x^2 = (a^j_{x,y})^2
        INDArray activitySqr = input.get(0).mul(input.get(0));
        INDArray sumPart = activitySqr.dup();

        //sum_{j=max(0, i - n/2)}^{max(N-1, i + n/2)} (a^j_{x,y})^2 )
        for (int i = 1; i < halfN + 1; i++) {
            tmp = sumPart.get(NDArrayIndex.all(), interval(i, channel), NDArrayIndex.all(), NDArrayIndex.all());
            addVal = activitySqr.get(NDArrayIndex.all(), interval(0, channel - i), NDArrayIndex.all(),
                            NDArrayIndex.all());
            tmp.addi(addVal);

            tmp = sumPart.get(NDArrayIndex.all(), interval(0, channel - i), NDArrayIndex.all(), NDArrayIndex.all());
            addVal = activitySqr.get(NDArrayIndex.all(), interval(i, channel), NDArrayIndex.all(), NDArrayIndex.all());
            tmp.addi(addVal);
        }

        // unitScale = (k + alpha * sum_{j=max(0, i - n/2)}^{max(N-1, i + n/2)} (a^j_{x,y})^2 )
        unitScale = sumPart.mul(alpha).addi(k).leverageTo(ComputationGraph.workspaceExternal);
        // y = x * unitScale**-beta
        scale = Transforms.pow(unitScale, -beta).leverageTo(ComputationGraph.workspaceExternal);
        activations = input.get(0).mul(scale).leverageTo(ComputationGraph.workspaceExternal);
        return ActivationsFactory.getInstance().create(activations);
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
    public InputPreProcessor getPreProcessor() {
        return layerConf().getPreProcessor();
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
