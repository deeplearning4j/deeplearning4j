package org.deeplearning4j.nn.layers.normalization;

import org.deeplearning4j.berkeley.Iterators;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import static org.nd4j.linalg.indexing.NDArrayIndex.interval;

/**
 * Deep neural net normalization approach normalizes activations between layers
 * "brightness normalization"
 *
 * For a^i_{x,y} the activity of a neuron computed by applying kernel i
 *    at position (x,y) and applying ReLU nonlinearity, the response
 *    normalized activation b^i_{x,y} is given by:
 *
 * b^i_{x,y} = a^i_{x,y} /
 * (k + alpha * sum_{j=max(0, i - n/2)}^{max(N-1, i + n/2)} (a^j_{x,y})^2 )**beta
 *
 * Reference: http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf
 * Created by nyghtowl on 10/29/15.
 */
public class LocalResponseNormalization extends BaseLayer<org.deeplearning4j.nn.conf.layers.LocalResponseNormalization>{


    public LocalResponseNormalization(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
    }

    public LocalResponseNormalization(NeuralNetConfiguration conf) {
        super(conf);
    }

    @Override
    public Type type() {
        return Type.NORMALIZATION;
    }

    @Override
    public void fit(INDArray input) {}

    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon) {
        INDArray nextEpsilon = null;
        Gradient retGradient = new DefaultGradient();

        return new Pair<>(retGradient,nextEpsilon);
    }

    @Override
    public INDArray activate(boolean training) {
        double k, alpha, beta, n;
        k = layerConf().getK();
        alpha = layerConf().getAlpha();
        beta = layerConf().getBeta();
        n = layerConf().getN();
        int halfN = (int) n/2;
        int examples = input.shape()[0];
        int channels = input.shape()[1];
        int height = input.shape()[2];
        int width = input.shape()[3];
        INDArray activitySqr = input.mul(input);
        INDArray extraChannels = Nd4j.zeros(new int[] {examples, (channels+2*halfN), height, width});
        INDArray scale = Nd4j.zeros(activitySqr.shape());

        extraChannels.put(new INDArrayIndex[]{
                NDArrayIndex.all(),
                interval(halfN,(halfN+channels)),
                NDArrayIndex.all(),
                NDArrayIndex.all()}
                , activitySqr);

        for (int i = 1; i < n; i++) {
            scale.addi(extraChannels.get(
                    NDArrayIndex.all(),
                    interval(i, (i + channels)),
                    NDArrayIndex.all(),
                    NDArrayIndex.all()));
        }
        scale = Transforms.pow(scale.mul(alpha).add(k),beta);
        return input.div(scale);

    }

    @Override
    public Layer transpose(){
        throw new UnsupportedOperationException("Not yet implemented");
    }


    @Override
    public Gradient calcGradient(Gradient layerError, INDArray indArray) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public void merge(Layer layer, int batchSize) {
        throw new UnsupportedOperationException();
    }


}
