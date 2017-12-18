package org.deeplearning4j.nn.layers.wrapper;

import lombok.Data;
import lombok.NonNull;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

import java.util.Collection;
import java.util.Map;

/**
 * Abstract wrapper layer. The idea: this class passes through all methods to the underlying layer.
 * Then, subclasses of BaseWrapperLayer can selectively override specific methods, rather than having
 * to implement every single one of the passthrough methods in each subclass.
 *
 * @author Alex Black
 */
@Data
public abstract class BaseWrapperLayer implements Layer {

    protected Layer underlying;

    public BaseWrapperLayer(@NonNull Layer underlying){
        this.underlying = underlying;
    }

    @Override
    public void setCacheMode(CacheMode mode) {
        underlying.setCacheMode(mode);
    }

    @Override
    public double calcL2(boolean backpropOnlyParams) {
        return underlying.calcL2(backpropOnlyParams);
    }

    @Override
    public double calcL1(boolean backpropOnlyParams) {
        return underlying.calcL1(backpropOnlyParams);
    }

    @Override
    public Type type() {
        return underlying.type();
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon) {
        return underlying.backpropGradient(epsilon);
    }

    @Override
    public INDArray preOutput(INDArray x) {
        return underlying.preOutput(x);
    }

    @Override
    public INDArray preOutput(INDArray x, TrainingMode training) {
        return underlying.preOutput(x, training);
    }

    @Override
    public INDArray activate(TrainingMode training) {
        return underlying.activate(training);
    }

    @Override
    public INDArray activate(INDArray input, TrainingMode training) {
        return underlying.activate(input, training);
    }

    @Override
    public INDArray preOutput(INDArray x, boolean training) {
        return underlying.preOutput(x, training);
    }

    @Override
    public INDArray activate(boolean training) {
        return underlying.activate(training);
    }

    @Override
    public INDArray activate(INDArray input, boolean training) {
        return underlying.activate(input, training);
    }

    @Override
    public INDArray activate() {
        return underlying.activate();
    }

    @Override
    public INDArray activate(INDArray input) {
        return underlying.activate(input);
    }

    @Override
    public Layer transpose() {
        throw new UnsupportedOperationException("Not supported");   //If required, implement in subtype (so traspose is wrapped)
    }

    @Override
    public Layer clone() {
        throw new UnsupportedOperationException("Clone not supported");
    }

    @Override
    public Collection<IterationListener> getListeners() {
        return underlying.getListeners();
    }

    @Override
    public void setListeners(IterationListener... listeners) {
        underlying.setListeners(listeners);
    }

    @Override
    public void addListeners(IterationListener... listener) {
        underlying.addListeners(listener);
    }

    @Override
    public void fit() {
        underlying.fit();
    }

    @Override
    public void update(Gradient gradient) {
        underlying.update(gradient);
    }

    @Override
    public void update(INDArray gradient, String paramType) {
        underlying.update(gradient, paramType);
    }

    @Override
    public double score() {
        return underlying.score();
    }

    @Override
    public void computeGradientAndScore() {
        underlying.computeGradientAndScore();
    }

    @Override
    public void accumulateScore(double accum) {
        underlying.accumulateScore(accum);
    }

    @Override
    public INDArray params() {
        return underlying.params();
    }

    @Override
    public int numParams() {
        return underlying.numParams();
    }

    @Override
    public int numParams(boolean backwards) {
        return underlying.numParams();
    }

    @Override
    public void setParams(INDArray params) {
        underlying.setParams(params);
    }

    @Override
    public void setParamsViewArray(INDArray params) {
        underlying.setParamsViewArray(params);
    }

    @Override
    public INDArray getGradientsViewArray() {
        return underlying.getGradientsViewArray();
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray gradients) {
        underlying.setBackpropGradientsViewArray(gradients);
    }

    @Override
    public void fit(INDArray data) {
        underlying.fit(data);
    }

    @Override
    public void iterate(INDArray input) {
        underlying.iterate(input);
    }

    @Override
    public Gradient gradient() {
        return underlying.gradient();
    }

    @Override
    public Pair<Gradient, Double> gradientAndScore() {
        return underlying.gradientAndScore();
    }

    @Override
    public int batchSize() {
        return underlying.batchSize();
    }

    @Override
    public NeuralNetConfiguration conf() {
        return underlying.conf();
    }

    @Override
    public void setConf(NeuralNetConfiguration conf) {
        underlying.setConf(conf);
    }

    @Override
    public INDArray input() {
        return underlying.input();
    }

    @Override
    public void validateInput() {
        underlying.validateInput();
    }

    @Override
    public ConvexOptimizer getOptimizer() {
        return underlying.getOptimizer();
    }

    @Override
    public INDArray getParam(String param) {
        return underlying.getParam(param);
    }

    @Override
    public void initParams() {
        underlying.initParams();
    }

    @Override
    public Map<String, INDArray> paramTable() {
        return underlying.paramTable();
    }

    @Override
    public Map<String, INDArray> paramTable(boolean backpropParamsOnly) {
        return underlying.paramTable(backpropParamsOnly);
    }

    @Override
    public void setParamTable(Map<String, INDArray> paramTable) {
        underlying.setParamTable(paramTable);
    }

    @Override
    public void setParam(String key, INDArray val) {
        underlying.setParam(key, val);
    }

    @Override
    public void clear() {
        underlying.clear();
    }

    @Override
    public void applyConstraints(int iteration, int epoch) {
        underlying.applyConstraints(iteration, epoch);
    }

    @Override
    public void init() {
        underlying.init();
    }

    @Override
    public void setListeners(Collection<IterationListener> listeners) {
        underlying.setListeners(listeners);
    }

    @Override
    public void setIndex(int index) {
        underlying.setIndex(index);
    }

    @Override
    public int getIndex() {
        return 0;
    }

    @Override
    public int getIterationCount() {
        return underlying.getIterationCount();
    }

    @Override
    public int getEpochCount() {
        return underlying.getEpochCount();
    }

    @Override
    public void setIterationCount(int iterationCount) {
        underlying.setIterationCount(iterationCount);
    }

    @Override
    public void setEpochCount(int epochCount) {
        underlying.setEpochCount(epochCount);
    }

    @Override
    public void setInput(INDArray input) {
        underlying.setInput(input);
    }

    @Override
    public void setInputMiniBatchSize(int size) {
        underlying.setInputMiniBatchSize(size);
    }

    @Override
    public int getInputMiniBatchSize() {
        return underlying.getInputMiniBatchSize();
    }

    @Override
    public void setMaskArray(INDArray maskArray) {
        underlying.setMaskArray(maskArray);
    }

    @Override
    public INDArray getMaskArray() {
        return underlying.getMaskArray();
    }

    @Override
    public boolean isPretrainLayer() {
        return underlying.isPretrainLayer();
    }

    @Override
    public void clearNoiseWeightParams() {
        underlying.clearNoiseWeightParams();
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState, int minibatchSize) {
        return underlying.feedForwardMaskArray(maskArray, currentMaskState, minibatchSize);
    }
}
