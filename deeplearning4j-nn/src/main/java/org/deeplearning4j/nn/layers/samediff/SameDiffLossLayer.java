package org.deeplearning4j.nn.layers.samediff;

import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.api.layers.IOutputLayer;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.samediff.BaseSameDiffLossLayer;
import org.deeplearning4j.nn.conf.layers.samediff.BaseSameDiffOutputLayer;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.primitives.Pair;

import java.util.*;

public class SameDiffLossLayer implements IOutputLayer {

    public static final String LABEL_KEY = "label";

    protected NeuralNetConfiguration conf;
    @Getter @Setter protected int index;
    @Getter @Setter protected INDArray input;
    @Getter @Setter private INDArray labels;
    protected double score;

    @Getter @Setter protected int iterationCount;
    @Getter @Setter protected int epochCount;

    protected SameDiff sameDiff;
    protected SDVariable inVar;
    protected SDVariable labelVar;

    protected Gradient emptyGrad = new DefaultGradient();

    protected double fullNetL1;
    protected double fullNetL2;


    public SameDiffLossLayer(NeuralNetConfiguration conf) {
        this.conf = conf;
    }

    @Override
    public double computeScore(double fullNetworkL1, double fullNetworkL2, boolean training) {
        this.fullNetL1 = fullNetworkL1;
        this.fullNetL2 = fullNetworkL2;
        computeGradientAndScore();
        return score;
    }

    @Override
    public INDArray computeScoreForExamples(double fullNetworkL1, double fullNetworkL2) {
        throw new UnsupportedOperationException("Not yet implemented");
    }


    @Override
    public double score() {
        return score;
    }

    @Override
    public void computeGradientAndScore() {
        Pair<Gradient,INDArray> p = backpropGradient(null);

        sameDiff.associateArrayWithVariable(input, inVar);
        sameDiff.associateArrayWithVariable(labels, labelVar);

        INDArray out = sameDiff.execAndEndResult();
        if(out.length() != 1){
            throw new IllegalStateException("Expected scalar score: got array with shape " + Arrays.toString(out.shape()));
        }

        score = out.getDouble(0);
        score += fullNetL1 + fullNetL2;
//        score /= input.size(0);
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon) {
        if(input == null){
            throw new IllegalStateException("Cannot compute gradient without input (input is null)");
        }
        if(labels == null){
            throw new IllegalStateException("Cannot compute gradient without labels (labels are null)");
        }
        if(sameDiff == null){
            doInit();
        }

        Pair<Map<SDVariable,DifferentialFunction>,List<DifferentialFunction>> p = sameDiff.execBackwards();

        SDVariable inGrad = sameDiff.grad(inVar.getVarName());

        return new Pair<>(emptyGrad, inGrad.getArr());
    }


    protected void doInit(){
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("input", input);
        SDVariable label = sd.var("label", labels);

        BaseSameDiffLossLayer l = (BaseSameDiffLossLayer) conf.getLayer();
        l.defineLayer(sd, in, label);

        this.sameDiff = sd;
        this.inVar = in;
        this.labelVar = label;
    }


    //--------------------------------------------------------------------------------------------------------

    @Override
    public double f1Score(DataSet data) {
        throw new UnsupportedOperationException();
    }

    @Override
    public double f1Score(INDArray examples, INDArray labels) {
        throw new UnsupportedOperationException();
    }

    @Override
    public int numLabels() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void fit(DataSetIterator iter) {
        throw new UnsupportedOperationException();
    }

    @Override
    public int[] predict(INDArray examples) {
        throw new UnsupportedOperationException();
    }

    @Override
    public List<String> predict(DataSet dataSet) {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray labelProbabilities(INDArray examples) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void fit(INDArray examples, INDArray labels) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void fit(DataSet data) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void fit(INDArray examples, int[] labels) {
        throw new UnsupportedOperationException();
    }


    @Override
    public void setCacheMode(CacheMode mode) {
        //No op
    }

    @Override
    public double calcL2(boolean backpropOnlyParams) {
        return 0;   //No params
    }

    @Override
    public double calcL1(boolean backpropOnlyParams) {
        return 0;   //No params
    }

    @Override
    public Type type() {
        return Type.FEED_FORWARD;
    }

    @Override
    public INDArray preOutput(INDArray x) {
        return x;
    }

    @Override
    public INDArray preOutput(INDArray x, TrainingMode training) {
        return x;
    }

    @Override
    public INDArray activate(TrainingMode training) {
        return input;
    }

    @Override
    public INDArray activate(INDArray input, TrainingMode training) {
        return input;
    }

    @Override
    public INDArray preOutput(INDArray x, boolean training) {
        return x;
    }

    @Override
    public INDArray activate(boolean training) {
        return input;
    }

    @Override
    public INDArray activate(INDArray input, boolean training) {
        return input;
    }

    @Override
    public INDArray activate() {
        return input;
    }

    @Override
    public INDArray activate(INDArray input) {
        return input;
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
        return null;
    }

    @Override
    public void setListeners(IterationListener... listeners) {
        //No op
    }

    @Override
    public void addListeners(IterationListener... listener) {
        //No op
    }

    @Override
    public void fit() {
        throw new UnsupportedOperationException("Cannot fit SameDiffLossLayer");
    }

    @Override
    public void update(Gradient gradient) {
        throw new UnsupportedOperationException("Not supported (no parameters)");
    }

    @Override
    public void update(INDArray gradient, String paramType) {
        throw new UnsupportedOperationException("Not supported (no parameters)");
    }

    @Override
    public void accumulateScore(double accum) {
        throw new UnsupportedOperationException("Not supported");
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
    public int numParams(boolean backwards) {
        return 0;
    }

    @Override
    public void setParams(INDArray params) {
        throw new UnsupportedOperationException("Not supported (no parameters)");
    }

    @Override
    public void setParamsViewArray(INDArray params) {
        if(params != null) {
            throw new UnsupportedOperationException("Not supported (no parameters)");
        }
    }

    @Override
    public INDArray getGradientsViewArray() {
        throw new UnsupportedOperationException("Not supported (no parameters)");
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray gradients) {
        throw new UnsupportedOperationException("Not supported (no parameters)");
    }

    @Override
    public void fit(INDArray data) {
        throw new UnsupportedOperationException("Cannot fit SameDiffLossLayer");
    }

    @Override
    public void iterate(INDArray input) {
        throw new UnsupportedOperationException("Cannot fit SameDiffLossLayer");
    }

    @Override
    public Gradient gradient() {
        return null;    //No parameters -> no gradient
    }

    @Override
    public Pair<Gradient, Double> gradientAndScore() {
        return new Pair<>(gradient(), score());
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
        return input;
    }

    @Override
    public void validateInput() {
        //No op
    }

    @Override
    public ConvexOptimizer getOptimizer() {
        throw new UnsupportedOperationException("Not supported (no parameters)");
    }

    @Override
    public INDArray getParam(String param) {
        throw new UnsupportedOperationException("Not supported (no parameters)");
    }

    @Override
    public void initParams() {
        throw new UnsupportedOperationException("Not supported (no parameters)");
    }

    @Override
    public Map<String, INDArray> paramTable() {
        return Collections.emptyMap();
    }

    @Override
    public Map<String, INDArray> paramTable(boolean backpropParamsOnly) {
        return paramTable();
    }

    @Override
    public void setParamTable(Map<String, INDArray> paramTable) {
        if(paramTable != null && paramTable.size() > 0) {
            throw new UnsupportedOperationException("Not supported (no parameters)");
        }
    }

    @Override
    public void setParam(String key, INDArray val) {
        throw new UnsupportedOperationException("Not supported (no parameters)");
    }

    @Override
    public void clear() {
        input = null;
        labels = null;
    }

    @Override
    public void applyConstraints(int iteration, int epoch) {
        //No op
    }

    @Override
    public void init() {
        //No op
    }

    @Override
    public void setListeners(Collection<IterationListener> listeners) {
        //No op
    }

    @Override
    public void setInputMiniBatchSize(int size) {

    }

    @Override
    public int getInputMiniBatchSize() {
        return 0;
    }

    @Override
    public void setMaskArray(INDArray maskArray) {
        if(maskArray != null) {
            throw new UnsupportedOperationException("Mask arrays: not yet supported for SameDiffLossLayer");
        }
    }

    @Override
    public INDArray getMaskArray() {
        return null;
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
    public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState, int minibatchSize) {
        if(maskArray != null){
            throw new UnsupportedOperationException("Mask arrays: not yet supported for SameDiffLossLayer");
        }
        return null;
    }
}
