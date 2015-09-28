package org.deeplearning4j.nn.layers.normalization;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.params.BatchNormalizationParamInitializer;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.*;

/**
 * Batch normalization layer.
 * http://arxiv.org/pdf/1410.7455v8.pdf
 *
 * @author Adam Gibson
 */
public class BatchNormalization implements Layer {
    private INDArray std;
    private NeuralNetConfiguration conf;
    private int index = 0;
    private List<IterationListener> listeners = new ArrayList<>();
    private Map<String,INDArray> params = new LinkedHashMap<>();
    private int[] shape;
    private Gradient gradient;
    private INDArray xHat;

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
        return Type.CONVOLUTIONAL;
    }

    @Override
    public Gradient error(INDArray input) {
        return null;
    }

    @Override
    public INDArray derivativeActivation(INDArray input) {
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
        //  gbeta = gy.sum(axis=(0, 2), keepdims=True)
        INDArray gBeta = epsilon.sum(0,2);
        getParam(BatchNormalizationParamInitializer.GAMMA_GRADIENT).addi(gBeta);
        // ggamma = (gy * self.x_hat).sum(axis=(0, 2), keepdims=True)
        INDArray newGamma = epsilon.mul(xHat).sum(0,2);
        getParam(BatchNormalizationParamInitializer.GAMMA_GRADIENT).addi(newGamma);
        //  coeff = self.gamma / self.std
        INDArray coefficients = getParam(BatchNormalizationParamInitializer.GAMMA).div(std);
        gBeta.divi(m);
        getParam(BatchNormalizationParamInitializer.GAMMA_GRADIENT).divi(m);
        INDArray ret = coefficients.mul(epsilon.sub(xHat).muli(getParam(BatchNormalizationParamInitializer.GAMMA_GRADIENT)).subi(gBeta));
        ret = ret.reshape(shape);
        Gradient g = new DefaultGradient();
        g.setGradientFor(BatchNormalizationParamInitializer.GAMMA_GRADIENT,getParam(BatchNormalizationParamInitializer.GAMMA_GRADIENT));
        g.setGradientFor(BatchNormalizationParamInitializer.BETA_GRADIENT,getParam(BatchNormalizationParamInitializer.BETA_GRADIENT));
        this.gradient = g;
        return new Pair<>(g,ret);
    }

    @Override
    public void merge(Layer layer, int batchSize) {

    }

    @Override
    public INDArray activationMean() {
        return null;
    }

    @Override
    public void update(Gradient gradient) {

    }

    @Override
    public void fit() {

    }

    @Override
    public void update(INDArray gradient, String paramType) {

    }

    @Override
    public double score() {
        return 0;
    }

    @Override
    public void computeGradientAndScore() {

    }

    @Override
    public void accumulateScore(double accum) {

    }

    @Override
    public INDArray params() {
        return null;
    }

    @Override
    public int numParams() {
        return 0;
    }

    @Override
    public void setParams(INDArray params) {

    }

    @Override
    public void fit(INDArray data) {

    }

    @Override
    public void iterate(INDArray input) {

    }

    @Override
    public Gradient gradient() {
        return gradient;
    }

    @Override
    public Pair<Gradient, Double> gradientAndScore() {
        return new Pair<>(gradient(),score());
    }

    @Override
    public int batchSize() {
        return 0;
    }

    @Override
    public NeuralNetConfiguration conf() {
        return conf;
    }

    @Override
    public void setConf(NeuralNetConfiguration conf) {
        this.conf = conf;
    }

    @Override
    public INDArray input() {
        return null;
    }

    @Override
    public void validateInput() {

    }

    @Override
    public ConvexOptimizer getOptimizer() {
        return null;
    }

    @Override
    public INDArray getParam(String param) {
        return params.get(param);
    }

    @Override
    public void initParams() {

    }

    @Override
    public Map<String, INDArray> paramTable() {
        return params;
    }

    @Override
    public void setParamTable(Map<String, INDArray> paramTable) {
        this.params = paramTable;
    }

    @Override
    public void setParam(String key, INDArray val) {
        params.put(key,val);
    }

    @Override
    public void clear() {

    }

    @Override
    public INDArray preOutput(INDArray x) {
        return preOutput(x,TrainingMode.TRAIN);
    }

    @Override
    public INDArray preOutput(INDArray x, TrainingMode training) {
        int[] activationShape = getShape(x);
        org.deeplearning4j.nn.conf.layers.BatchNormalization layerConf = (org.deeplearning4j.nn.conf.layers.BatchNormalization) conf().getLayer();
        //cache the shape
        this.shape = activationShape;
        INDArray mean,var;
        if(training != TrainingMode.TEST && !layerConf.isUseBatchMean()) {
            mean = x.mean(0, 2);
            var = x.var(0, 2);
            var.addi(layerConf.getEps());
        }
        else {
            mean = getParam(BatchNormalizationParamInitializer.AVG_MEAN);
            var = getParam(BatchNormalizationParamInitializer.AVG_VAR);
        }

        std = Transforms.sqrt(var);
        INDArray xMu = x.sub(mean);
        xHat = xMu.div(std);
        INDArray out = getParam(BatchNormalizationParamInitializer.GAMMA).add(xHat).addi(getParam(BatchNormalizationParamInitializer.BETA));
        double decay = 0.0;
        if(training != TrainingMode.TEST && !layerConf.isUseBatchMean()) {
            if(layerConf.isFinetune()) {
                layerConf.setN(layerConf.getN() + 1);
                decay =  1. / layerConf.getN();
            }
            else
                decay = layerConf.getDecay();
            int m  = activationShape[0] * activationShape[2];
            double  adjust = m / Math.max(m - 1., 1.);
            getParam(BatchNormalizationParamInitializer.AVG_MEAN).muli(decay);
            getParam(BatchNormalizationParamInitializer.AVG_MEAN).addi(mean.mul((1 - decay)));
            getParam(BatchNormalizationParamInitializer.AVG_VAR).muli(decay);
            getParam(BatchNormalizationParamInitializer.AVG_VAR).addi(var.mul((1 - decay) * adjust));

        }

        return out.reshape(x.shape());
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
    public INDArray activate(boolean training) {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray activate(INDArray input, boolean training) {
        return preOutput(input,training);
    }

    @Override
    public INDArray activate() {
        throw new UnsupportedOperationException();

    }

    @Override
    public INDArray activate(INDArray input) {
        throw new UnsupportedOperationException();

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

    @Override
    public void setInput(INDArray input) {

    }

    @Override
    public void setInputMiniBatchSize(int size) {

    }

    @Override
    public int getInputMiniBatchSize() {
        return 0;
    }

    public int[] getShape(INDArray x) {
        int leadDim = x.size(0);
        int cDim = getParam(BatchNormalizationParamInitializer.GAMMA).length();
        int rdim = x.length() / (leadDim * cDim);
        if(leadDim * cDim * rdim != x.length())
            throw new IllegalArgumentException("Illegal input for batch size");
        return new int[] {leadDim,cDim,rdim};

    }

}
