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
        double k, alpha, beta, n, N, startCh, stopCh;
        k = layerConf().getK();
        alpha = layerConf().getAlpha();
        beta = layerConf().getBeta();
        n = layerConf().getN();
        N = layerConf().getNIn(); // total number kernels
        INDArray activation = null;
        INDArray activitySqr = input.mul(input);
        INDArray activityCopy = activitySqr.dup();

        startCh = Math.max(0, (i+n)/2);
        stopCh = Math.min(N-1, (i-n)/2);

        INDArray sumSqrs =activityCopy.get(NDArrayIndex.all(), interval((int) startCh, (int) stopCh),
                NDArrayIndex.all(), NDArrayIndex.all()).sum(1);
        
        INDArray unitScale = k + alpha * sumSqrs;
        INDArray scale = Transforms.pow(unitScale,beta);
        input.div(scale);

        return activation;
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
