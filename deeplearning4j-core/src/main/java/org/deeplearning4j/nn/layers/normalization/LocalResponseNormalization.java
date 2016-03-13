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
 * Used for nets like AlexNet
 *
 * For a^i_{x,y} the activity of a neuron computed by applying kernel i
 *    at position (x,y) and applying ReLU nonlinearity, the response
 *    normalized activation b^i_{x,y} is given by:

 *  x^2 = (a^j_{x,y})^2
 *  unitScale = (k + alpha * sum_{j=max(0, i - n/2)}^{max(N-1, i + n/2)} (a^j_{x,y})^2 )
 *  y = b^i_{x,y} = x * unitScale**-beta
 *
 *  gy = epsilon (aka deltas from previous layer)
 *  sumPart = sum(a^j_{x,y} * gb^j_{x,y})
 *  gx = gy * unitScale**-beta - 2 * alpha * beta * sumPart/unitScale * a^i_{x,y}
 *
 * Reference:
 * http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf
 * https://github.com/vlfeat/matconvnet/issues/10
 * Chainer
 *
 * Created by nyghtowl on 10/29/15.
 */
public class LocalResponseNormalization extends BaseLayer<org.deeplearning4j.nn.conf.layers.LocalResponseNormalization>{

    private double k;
    private double n;
    private double alpha;
    private double beta;
    private int halfN;
    private INDArray activations, unitScale, scale;

    public LocalResponseNormalization(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
    }

    public LocalResponseNormalization(NeuralNetConfiguration conf) {
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
    public void fit(INDArray input) {}

    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon) {
        int channel = input.shape()[1];
        INDArray tmp, addVal;
        Gradient retGradient = new DefaultGradient();
        INDArray reverse = activations.mul(epsilon);
        INDArray sumPart = reverse.dup();

        // sumPart = sum(a^j_{x,y} * gb^j_{x,y})
        for (int i = 1; i < halfN+1; i++){
            tmp = sumPart.get(
                    new INDArrayIndex[]{
                            NDArrayIndex.all(),
                            interval(i, channel),
                            NDArrayIndex.all(),
                            NDArrayIndex.all()});
            addVal = reverse.get(
                    new INDArrayIndex[]{
                            NDArrayIndex.all(),
                            interval(0, channel-i),
                            NDArrayIndex.all(),
                            NDArrayIndex.all()});
            sumPart.put(new INDArrayIndex[]{
                    NDArrayIndex.all(),
                    interval(i, channel),
                    NDArrayIndex.all(),
                    NDArrayIndex.all()}, tmp.addi(addVal));

            tmp = sumPart.get(
                    new INDArrayIndex[]{
                            NDArrayIndex.all(),
                            interval(0, channel-i),
                            NDArrayIndex.all(),
                            NDArrayIndex.all()});
            addVal = reverse.get(
                    new INDArrayIndex[]{
                            NDArrayIndex.all(),
                            interval(i, channel),
                            NDArrayIndex.all(),
                            NDArrayIndex.all()});
            sumPart.put(new INDArrayIndex[]{
                    NDArrayIndex.all(),
                    interval(0, channel-i),
                    NDArrayIndex.all(),
                    NDArrayIndex.all()}, tmp.addi(addVal));
        }

        // gx = gy * unitScale**-beta - 2 * alpha * beta * sumPart/unitScale * a^i_{x,y}
        INDArray nextEpsilon = epsilon.mul(scale).sub(input.mul(2 * alpha * beta).mul(sumPart.div(unitScale)));
        return new Pair<>(retGradient,nextEpsilon);
    }

    @Override
    public INDArray activate(boolean training) {
        k = layerConf().getK();
        n = layerConf().getN();
        alpha = layerConf().getAlpha();
        beta = layerConf().getBeta();
        halfN = (int) n/2;
        int channel = input.shape()[1];
        INDArray tmp, addVal;
        // x^2 = (a^j_{x,y})^2
        INDArray activitySqr = input.mul(input);
        INDArray sumPart = activitySqr.dup();

        //sum_{j=max(0, i - n/2)}^{max(N-1, i + n/2)} (a^j_{x,y})^2 )
        for (int i = 1; i < halfN+1; i++){
            tmp = sumPart.get(
                    new INDArrayIndex[]{
                            NDArrayIndex.all(),
                            interval(i, channel),
                            NDArrayIndex.all(),
                            NDArrayIndex.all()});
            addVal = activitySqr.get(
                    new INDArrayIndex[]{
                            NDArrayIndex.all(),
                            interval(0, channel-i),
                            NDArrayIndex.all(),
                            NDArrayIndex.all()});
            sumPart.put(new INDArrayIndex[]{
                    NDArrayIndex.all(),
                    interval(i, channel),
                    NDArrayIndex.all(),
                    NDArrayIndex.all()}, tmp.addi(addVal));

            tmp = sumPart.get(
                    new INDArrayIndex[]{
                            NDArrayIndex.all(),
                            interval(0, channel-i),
                            NDArrayIndex.all(),
                            NDArrayIndex.all()});
            addVal = activitySqr.get(
                    new INDArrayIndex[]{
                            NDArrayIndex.all(),
                            interval(i, channel),
                            NDArrayIndex.all(),
                            NDArrayIndex.all()});
            sumPart.put(new INDArrayIndex[]{
                    NDArrayIndex.all(),
                    interval(0, channel-i),
                    NDArrayIndex.all(),
                    NDArrayIndex.all()}, tmp.addi(addVal));
        }

        // unitScale = (k + alpha * sum_{j=max(0, i - n/2)}^{max(N-1, i + n/2)} (a^j_{x,y})^2 )
        unitScale = sumPart.mul(alpha).add(k);
        // y = x * unitScale**-beta
        scale = Transforms.pow(unitScale, -beta);
        activations = input.mul(scale);
        return activations;

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
