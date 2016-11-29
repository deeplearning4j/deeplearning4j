package org.deeplearning4j.nn.layers.variational;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.optimize.Solver;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

import static org.deeplearning4j.nn.params.VariationalAutoencoderParamInitializer.BIAS_KEY_SUFFIX;
import static org.deeplearning4j.nn.params.VariationalAutoencoderParamInitializer.WEIGHT_KEY_SUFFIX;

/**
 * Created by Alex on 25/11/2016.
 */
public class VariationalAutoencoder implements Layer {

    protected INDArray input;
    protected INDArray paramsFlattened;
    protected INDArray gradientsFlattened;
    protected Map<String,INDArray> params;
    protected transient Map<String,INDArray> gradientViews;
    protected NeuralNetConfiguration conf;
    protected INDArray dropoutMask;
    protected boolean dropoutApplied = false;
    protected double score = 0.0;
    protected ConvexOptimizer optimizer;
    protected Gradient gradient;
    protected Collection<IterationListener> iterationListeners = new ArrayList<>();
    protected int index = 0;
    protected INDArray maskArray;
    protected Solver solver;

    protected int[] encoderLayerSizes;
    protected int[] decoderLayerSizes;

    public VariationalAutoencoder(NeuralNetConfiguration conf){
        this.conf = conf;

        this.encoderLayerSizes = ((org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder)conf.getLayer()).getEncoderLayerSizes();
        this.decoderLayerSizes = ((org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder)conf.getLayer()).getDecoderLayerSizes();
    }



    @Override
    public void fit() {
        if(input == null){
            throw new IllegalStateException("Cannot fit layer: layer input is null (not set)");
        }

    }

    @Override
    public void update(Gradient gradient) {

    }

    @Override
    public void update(INDArray gradient, String paramType) {

    }

    @Override
    public double score() {
        return score;
    }

    @Override
    public void computeGradientAndScore() {

    }

    @Override
    public void accumulateScore(double accum) {

    }

    @Override
    public INDArray params() {
        return paramsFlattened;
    }

    @Override
    public int numParams() {
        return numParams(false);
    }

    @Override
    public int numParams(boolean backwards) {
        int ret = 0;
        for(Map.Entry<String,INDArray> entry : params.entrySet()){
            if(backwards && (entry.getKey().startsWith("d") || entry.getKey().startsWith("eZXLogStdev2"))) continue;
            ret += entry.getValue().length();
        }
        return ret;
    }

    @Override
    public void setParams(INDArray params) {
        if(params.length() != this.paramsFlattened.length()){
            throw new IllegalArgumentException("Cannot set parameters: expected parameters vector of length " +
                this.paramsFlattened.length() + " but got parameters array of length " + params.length());
        }
        this.paramsFlattened.assign(params);
    }

    @Override
    public void setParamsViewArray(INDArray params) {
        this.paramsFlattened = params;
        //TODO flattening/unflattening...
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray gradients) {
        if(this.params != null && gradients.length() != numParams()){
            throw new IllegalArgumentException("Invalid input: expect gradients array of length " + numParams()
                    + ", got gradient array of length of length " + gradients.length());
        }

        this.gradientsFlattened = gradients;
        this.gradientViews = conf.getLayer().initializer().getGradientsFromFlattened(conf,gradients);
    }

    @Override
    public void applyLearningRateScoreDecay() {

    }

    @Override
    public void fit(INDArray data) {
        this.setInput(data);
        fit();
    }

    @Override
    public void iterate(INDArray input) {
        fit(input);
    }

    @Override
    public Gradient gradient() {
        return null;
    }

    @Override
    public Pair<Gradient, Double> gradientAndScore() {
        return new Pair<>(gradient(), score());
    }

    @Override
    public int batchSize() {
        return input.size(0);
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
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public ConvexOptimizer getOptimizer() {
        return optimizer;
    }

    @Override
    public INDArray getParam(String param) {
        return params.get(param);
    }

    @Override
    public void initParams() {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public Map<String, INDArray> paramTable() {
        return new LinkedHashMap<>(params);
    }

    @Override
    public void setParamTable(Map<String, INDArray> paramTable) {
        this.params = paramTable;
    }

    @Override
    public void setParam(String key, INDArray val) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public void clear() {
        this.input = null;
        this.maskArray = null;
    }

    @Override
    public double calcL2() {
        return 0.0;     //TODO
    }

    @Override
    public double calcL1() {
        return 0.0;     //TODO
    }

    @Override
    public Type type() {
        return Type.FEED_FORWARD;
    }

    @Override
    public Gradient error(INDArray input) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public INDArray derivativeActivation(INDArray input) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public Gradient calcGradient(Gradient layerError, INDArray indArray) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon) {

        String afn = conf().getLayer().getActivationFunction();
        Gradient gradient = new DefaultGradient();

        INDArray[][] fwd = doForward(true, true);
        INDArray finalPreOut = fwd[0][fwd[0].length-1];
        INDArray activationDerivative = Nd4j.getExecutioner().execAndReturn(
                Nd4j.getOpFactory().createTransform(afn, finalPreOut).derivative());

        INDArray currentDelta = epsilon.muli(activationDerivative);

        //Finally, calculate mean value:
        String pzxMeanParamName = "eZXMean" + WEIGHT_KEY_SUFFIX;
        String pzxBiasParamName = "eZXMean" + BIAS_KEY_SUFFIX;
        INDArray mW = params.get(pzxMeanParamName);
        INDArray mB = params.get(pzxBiasParamName);

        INDArray dmW = gradientViews.get(pzxMeanParamName); //f order
        INDArray lastHiddenActivation = fwd[1][fwd[1].length-1];
        Nd4j.gemm(lastHiddenActivation,currentDelta,dmW,true,false,1.0,0.0);
        INDArray dmB = gradientViews.get(pzxBiasParamName);
//        System.out.println("dmB = " + Arrays.toString(dmB.shape()) + "\tcurrdelta = " + Arrays.toString(currentDelta.shape()));
        dmB.assign(currentDelta.sum(0));    //TODO: do this without the assign

        gradient.gradientForVariable().put(pzxMeanParamName, dmW);
        gradient.gradientForVariable().put(pzxBiasParamName, dmB);

        epsilon = mW.mmul(currentDelta.transpose()).transpose();

        int nEncoderLayers = encoderLayerSizes.length;

        INDArray current = input;
        for( int i=nEncoderLayers-1; i>=0; i-- ){
            String wKey = "e" + i + WEIGHT_KEY_SUFFIX;
            String bKey = "e" + i + BIAS_KEY_SUFFIX;

            INDArray weights = params.get(wKey);
            INDArray bias = params.get(bKey);

            INDArray dLdW = gradientViews.get(wKey);
            INDArray dLdB = gradientViews.get(bKey);

            INDArray preOut = fwd[0][i];
            activationDerivative = Nd4j.getExecutioner().execAndReturn(
                    Nd4j.getOpFactory().createTransform(afn, preOut).derivative());

            currentDelta = epsilon.muli(activationDerivative);

            INDArray actInput;
            if(i == 0){
                actInput = input;
            } else {
                actInput = fwd[1][i];
            }
            Nd4j.gemm(actInput,currentDelta,dLdW,true,false,1.0,0.0);
            dLdB.assign(currentDelta.sum(0));    //TODO: do this without the assign

            gradient.gradientForVariable().put(wKey, dLdW);
            gradient.gradientForVariable().put(bKey, dLdB);

            epsilon = weights.mmul(currentDelta.transpose()).transpose();
        }

        return new Pair<>(gradient, epsilon);
    }

    @Override
    public void merge(Layer layer, int batchSize) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public INDArray activationMean() {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public INDArray preOutput(INDArray x) {
        return preOutput(x, TrainingMode.TEST);
    }

    @Override
    public INDArray preOutput(INDArray x, TrainingMode training) {
        return preOutput(x, training == TrainingMode.TRAIN);
    }

    @Override
    public INDArray preOutput(INDArray x, boolean training) {
        setInput(x);
        return preOutput(training);
    }

    public INDArray preOutput(boolean training) {
        INDArray[][] arr = doForward(training, false);
        return arr[0][arr[0].length-1];
    }


    private INDArray[][] doForward(boolean training, boolean forBackprop){
        if(input == null){
            throw new IllegalStateException("Cannot do forward pass with null input");
        }

        //TODO input validation

        int nEncoderLayers = encoderLayerSizes.length;

        INDArray[] preOuts = new INDArray[encoderLayerSizes.length+1];
        INDArray[] activations = new INDArray[encoderLayerSizes.length];
        INDArray current = input;
        for( int i=0; i<nEncoderLayers; i++ ){
            String wKey = "e" + i + WEIGHT_KEY_SUFFIX;
            String bKey = "e" + i + BIAS_KEY_SUFFIX;

            INDArray weights = params.get(wKey);
            INDArray bias = params.get(bKey);

            current = current.mmul(weights).addiRowVector(bias);
            if(forBackprop){
                preOuts[i] = current.dup();
            }
            Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(
                    conf.getLayer().getActivationFunction(), current, conf.getExtraArgs() ));
            activations[i] = current;
        }

        //Finally, calculate mean value:
        String pzxMeanParamName = "eZXMean" + WEIGHT_KEY_SUFFIX;
        String pzxBiasParamName = "eZXMean" + BIAS_KEY_SUFFIX;
        INDArray mW = params.get(pzxMeanParamName);
        INDArray mB = params.get(pzxBiasParamName);

        current = current.mmul(mW).addiRowVector(mB);
        preOuts[preOuts.length-1] = current;

        return new INDArray[][]{preOuts, activations};
    }

    @Override
    public INDArray activate(TrainingMode training) {
        return activate(training == TrainingMode.TRAIN);
    }

    @Override
    public INDArray activate(INDArray input, TrainingMode training) {
        return null;
    }

    @Override
    public INDArray activate(boolean training) {
        INDArray output = preOutput(training);  //Mean values for p(z|x)

        if(!"identity".equals(conf.getLayer().getActivationFunction())){
            Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(
                    conf.getLayer().getActivationFunction(), output, conf.getExtraArgs() ));
        }
        return output;
    }

    @Override
    public INDArray activate(INDArray input, boolean training) {
        setInput(input);
        return activate(training);
    }

    @Override
    public INDArray activate() {
        return activate(false);
    }

    @Override
    public INDArray activate(INDArray input) {
        setInput(input);
        return activate();
    }

    @Override
    public Layer transpose() {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public Layer clone() {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public Collection<IterationListener> getListeners() {
        if(iterationListeners == null) return null;
        return new ArrayList<>(iterationListeners);
    }

    @Override
    public void setListeners(IterationListener... listeners) {
        setListeners(Arrays.<IterationListener>asList(listeners));
    }

    @Override
    public void setListeners(Collection<IterationListener> listeners) {
        if(iterationListeners == null) iterationListeners = new ArrayList<>();
        else iterationListeners.clear();

        iterationListeners.addAll(listeners);
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
        this.input = input;
    }

    @Override
    public void setInputMiniBatchSize(int size) {

    }

    @Override
    public int getInputMiniBatchSize() {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public void setMaskArray(INDArray maskArray) {
        this.maskArray = maskArray;
    }

    @Override
    public INDArray getMaskArray() {
        return maskArray;
    }
}
