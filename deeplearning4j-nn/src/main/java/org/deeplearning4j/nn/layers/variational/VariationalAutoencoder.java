package org.deeplearning4j.nn.layers.variational;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.variational.ReconstructionDistribution;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.params.VariationalAutoencoderParamInitializer;
import org.deeplearning4j.optimize.Solver;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.*;

import static org.deeplearning4j.nn.params.VariationalAutoencoderParamInitializer.BIAS_KEY_SUFFIX;
import static org.deeplearning4j.nn.params.VariationalAutoencoderParamInitializer.WEIGHT_KEY_SUFFIX;

/**
 * Created by Alex on 25/11/2016.
 */
@Data
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
    protected ReconstructionDistribution reconstructionDistribution;

    public VariationalAutoencoder(NeuralNetConfiguration conf){
        this.conf = conf;

        this.encoderLayerSizes = ((org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder)conf.getLayer()).getEncoderLayerSizes();
        this.decoderLayerSizes = ((org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder)conf.getLayer()).getDecoderLayerSizes();
        this.reconstructionDistribution = ((org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder)conf.getLayer()).getOutputDistribution();
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
        //First: do the full forward pass, through the network (including the random sampling, etc)
        //TODO handle multiple samples as an option...
        VAEFwdHelper fwd = doForward(true, true);

        INDArray pzxLogStdev2W = params.get("eZXLogStdev2" + WEIGHT_KEY_SUFFIX);
        INDArray pzxLogStdev2b = params.get("eZXLogStdev2" + BIAS_KEY_SUFFIX);

        INDArray pzxLogStdev2 = fwd.encoderActivations[fwd.encoderActivations.length-1].mmul(pzxLogStdev2W).addiRowVector(pzxLogStdev2b);

        INDArray pzxSigma = Transforms.exp(pzxLogStdev2,true);
        Transforms.sqrt(pzxSigma,false);

        int minibatch = input.size(0);
        int size = fwd.pzxPreOut.size(1);

        //TODO test mode for backprop...
        INDArray e = Nd4j.rand(minibatch, size);

        //TODO NEED ACTIVATION FUNCTION HERE
        INDArray z = fwd.pzxPreOut.add(pzxSigma.mul(e));      //z = mu + sigma * e, with e ~ N(0,1)

        //Next: need to do forward pass through decoder...

        int nDecoderLayers = decoderLayerSizes.length;
        INDArray current = z;
        INDArray[] decoderPreOut = new INDArray[nDecoderLayers];
        INDArray[] decoderActivations = new INDArray[nDecoderLayers];
        for( int i=0; i<nDecoderLayers; i++ ){
            String wKey = "d" + i + WEIGHT_KEY_SUFFIX;
            String bKey = "d" + i + BIAS_KEY_SUFFIX;

            INDArray weights = params.get(wKey);
            INDArray bias = params.get(bKey);

            current = current.mmul(weights).addiRowVector(bias);
            decoderPreOut[i] = current.dup();
            Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(
                    conf.getLayer().getActivationFunction(), current, conf.getExtraArgs() ));
            decoderActivations[i] = current;
        }

        INDArray xzw = params.get("dXZW");
        INDArray xzb = params.get("dXZb");


        INDArray pxzDistributionParams = current.mmul(xzw).addiRowVector(xzb);
        this.score = reconstructionDistribution.logProbability(input, pxzDistributionParams, true);
        //Need to add other component of score:
        INDArray temp = fwd.pzxPreOut.mul(fwd.pzxPreOut).addi(pzxSigma.mul(pzxSigma)).negi();       //TODO ACTIVATION FUNCTION
        temp.addi(pzxLogStdev2).addi(1.0);
        double scorePt1 = 0.5 / temp.size(0) * temp.sumNumber().doubleValue();
        this.score += scorePt1;

        INDArray dpdpxz = reconstructionDistribution.gradient(input, pxzDistributionParams);


        /////////////////////////////////////////////////////////

        INDArray dLdxzw = gradientViews.get("dXZW");
        INDArray dLdxzb = gradientViews.get("dXZb");
        INDArray lastDecActivations = decoderActivations[decoderActivations.length-1];
        Nd4j.gemm(lastDecActivations,dpdpxz,dLdxzw,true,false,1.0,0.0);
        dLdxzb.assign(dpdpxz.sum(0));    //TODO: do this without the assign

        //Do backprop for output probability distribution -> final decoder layer
        INDArray epsilon = xzw.mmul(dpdpxz.transpose()).transpose();

        //Next: we chain derivatives backwards...
        String afn = conf().getLayer().getActivationFunction();

        Gradient gradient = new DefaultGradient(gradientsFlattened);
        for( int i=nDecoderLayers-1; i>=0; i-- ){
            String wKey = "d" + i + WEIGHT_KEY_SUFFIX;
            String bKey = "d" + i + BIAS_KEY_SUFFIX;

            INDArray sigmaPrimeZ = Nd4j.getExecutioner().execAndReturn(
                Nd4j.getOpFactory().createTransform(afn, decoderPreOut[i]).derivative());

            INDArray currentDelta = epsilon.muli(sigmaPrimeZ);

            INDArray weights = params.get(wKey);
            INDArray dLdW = gradientViews.get(wKey);
            INDArray dLdB = gradientViews.get(bKey);

            INDArray actInput;
            if (i == 0) {
                actInput = z;
            } else {
                actInput = decoderActivations[i-1];
            }

            Nd4j.gemm(actInput,currentDelta,dLdW,true,false,1.0,0.0);
            dLdB.assign(currentDelta.sum(0));    //TODO: do this without the assign

            gradient.gradientForVariable().put(wKey, dLdW);
            gradient.gradientForVariable().put(bKey, dLdB);

            epsilon = weights.mmul(currentDelta.transpose()).transpose();
        }

        //Backprop through p(z|x)
        INDArray eZXMeanW = params.get("eZXMeanW");
        INDArray eZXMeanb = params.get("eZXMeanb");
        INDArray eZXLogStdev2W = params.get("eZXLogStdev2W");
        INDArray eZXLogStdev2b = params.get("eZXLogStdev2b");

        INDArray dLdz = epsilon;
        INDArray dLdmu = dLdz.sub(fwd.pzxPreOut);       //TODO ACTIVATION FUNCTION

        INDArray dLdLogSigma2 = dLdz.mul(e).muli(pzxSigma)
                .subi(pzxSigma.mul(pzxSigma)).addi(1).muli(0.5);

        INDArray dLdZXMeanW = gradientViews.get("eZXMeanW");
        INDArray dLdZXLogStdev2W = gradientViews.get("eZXLogStdev2W");
        INDArray dLdZXMeanb = gradientViews.get("eZXMeanb");
        INDArray dLdZXLogStdev2b = gradientViews.get("eZXLogStdev2b");

        INDArray lastEncoderActivation = fwd.encoderActivations[fwd.encoderActivations.length-1];
        Nd4j.gemm(lastEncoderActivation, dLdmu, dLdZXMeanW, true, false, 1.0, 0.0);
        Nd4j.gemm(lastEncoderActivation, dLdLogSigma2, dLdZXLogStdev2W, true, false, 1.0, 0.0);
//        dLdZXMeanb.assign(dLdmu.sub(fwd.getPzxPreOut()).sum(0));        //TODO activation function
        dLdZXMeanb.assign(dLdmu.sub(fwd.getPzxPreOut()).sum(0));        //TODO activation function
        dLdZXLogStdev2b.assign(dLdLogSigma2.sum(0));

        //TODO check order
        gradient.gradientForVariable().put("eZXMeanW", dLdZXMeanW);
        gradient.gradientForVariable().put("eZXMeanb", dLdZXMeanb);
        gradient.gradientForVariable().put("eZXLogStdev2W", dLdZXLogStdev2W);
        gradient.gradientForVariable().put("eZXLogStdev2b", dLdZXLogStdev2b);

        epsilon = eZXMeanW.mmul(dLdmu.transpose()).transpose();
        epsilon.addi(eZXLogStdev2W.mmul(dLdLogSigma2.transpose()).transpose());


        //Backprop through encoder:
        //TODO code reuse with non-pretrain backprop
        int nEncoderLayers = encoderLayerSizes.length;
        for( int i=nEncoderLayers-1; i>=0; i-- ){
            String wKey = "e" + i + WEIGHT_KEY_SUFFIX;
            String bKey = "e" + i + BIAS_KEY_SUFFIX;

            INDArray weights = params.get(wKey);
            INDArray bias = params.get(bKey);

            INDArray dLdW = gradientViews.get(wKey);
            INDArray dLdB = gradientViews.get(bKey);

            INDArray preOut = fwd.encoderPreOuts[i];
            INDArray activationDerivative = Nd4j.getExecutioner().execAndReturn(
                    Nd4j.getOpFactory().createTransform(afn, preOut).derivative());

            INDArray currentDelta = epsilon.muli(activationDerivative);

            INDArray actInput;
            if(i == 0){
                actInput = input;
            } else {
                actInput = fwd.encoderActivations[i];
            }
            Nd4j.gemm(actInput,currentDelta,dLdW,true,false,1.0,0.0);
            dLdB.assign(currentDelta.sum(0));    //TODO: do this without the assign

            gradient.gradientForVariable().put(wKey, dLdW);
            gradient.gradientForVariable().put(bKey, dLdB);

            epsilon = weights.mmul(currentDelta.transpose()).transpose();
        }

        this.gradient = gradient;
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
        return gradient;
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

    private boolean isPretrainParam(String param){
        if(param.startsWith("d") || param.startsWith("eZXLogStdev2")) return true;      //TODO don't hardcode
        return false;
    }

    @Override
    public double calcL2(boolean backpropParamsOnly) {
        if(!conf.isUseRegularization() || conf.getLayer().getL2() <= 0.0 ) return 0.0;

        double l2Sum = 0.0;
        for(Map.Entry<String,INDArray> e : paramTable().entrySet()){
            double l2 = conf().getL2ByParam(e.getKey());
            if(l2 <= 0.0 || (backpropParamsOnly && isPretrainParam(e.getKey()))){
                continue;
            }

            double l2Norm = e.getValue().norm2Number().doubleValue();
            l2Sum += 0.5 * l2 * l2Norm * l2Norm;
        }

        return l2Sum;
    }

    @Override
    public double calcL1(boolean backpropParamsOnly) {
        if(!conf.isUseRegularization() || conf.getLayer().getL1() <= 0.0 ) return 0.0;

        double l1Sum = 0.0;
        for(Map.Entry<String,INDArray> e : paramTable().entrySet()){
            double l1 = conf().getL1ByParam(e.getKey());
            if(l1 <= 0.0 || (backpropParamsOnly && isPretrainParam(e.getKey()))){
                continue;
            }

            l1Sum += l1 * e.getValue().norm1Number().doubleValue();
        }
        return l1Sum;
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

        VAEFwdHelper fwd = doForward(true, true);
        INDArray finalEncoderPreOut = fwd.encoderPreOuts[fwd.encoderPreOuts.length-1];
        INDArray activationDerivative = Nd4j.getExecutioner().execAndReturn(
                Nd4j.getOpFactory().createTransform(afn, finalEncoderPreOut).derivative());

        INDArray currentDelta = epsilon.muli(activationDerivative);

        //Finally, calculate mean value:
        String pzxMeanParamName = "eZXMean" + WEIGHT_KEY_SUFFIX;
        String pzxBiasParamName = "eZXMean" + BIAS_KEY_SUFFIX;
        INDArray mW = params.get(pzxMeanParamName);
        INDArray mB = params.get(pzxBiasParamName);

        INDArray dmW = gradientViews.get(pzxMeanParamName); //f order
        INDArray lastEncoderActivation = fwd.encoderActivations[fwd.encoderActivations.length-1];
        Nd4j.gemm(lastEncoderActivation,currentDelta,dmW,true,false,1.0,0.0);
        INDArray dmB = gradientViews.get(pzxBiasParamName);
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

            INDArray preOut = fwd.encoderPreOuts[i];
            activationDerivative = Nd4j.getExecutioner().execAndReturn(
                    Nd4j.getOpFactory().createTransform(afn, preOut).derivative());

            currentDelta = epsilon.muli(activationDerivative);

            INDArray actInput;
            if(i == 0){
                actInput = input;
            } else {
                actInput = fwd.encoderActivations[i];
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
        VAEFwdHelper f = doForward(training, false);
        return f.pzxPreOut;
    }

    @AllArgsConstructor @Data
    private static class VAEFwdHelper {
        private INDArray[] encoderPreOuts;
        private INDArray pzxPreOut;
        private INDArray[] encoderActivations;
    }


    private VAEFwdHelper doForward(boolean training, boolean forBackprop){
        if(input == null){
            throw new IllegalStateException("Cannot do forward pass with null input");
        }

        //TODO input validation

        int nEncoderLayers = encoderLayerSizes.length;

        INDArray[] encoderPreOuts = new INDArray[encoderLayerSizes.length];
        INDArray[] encoderActivations = new INDArray[encoderLayerSizes.length];
        INDArray current = input;
        for( int i=0; i<nEncoderLayers; i++ ){
            String wKey = "e" + i + WEIGHT_KEY_SUFFIX;
            String bKey = "e" + i + BIAS_KEY_SUFFIX;

            INDArray weights = params.get(wKey);
            INDArray bias = params.get(bKey);

            current = current.mmul(weights).addiRowVector(bias);
            if(forBackprop){
                encoderPreOuts[i] = current.dup();
            }
            Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(
                    conf.getLayer().getActivationFunction(), current, conf.getExtraArgs() ));
            encoderActivations[i] = current;
        }

        //Finally, calculate mean value:
        String pzxMeanParamName = "eZXMean" + WEIGHT_KEY_SUFFIX;
        String pzxBiasParamName = "eZXMean" + BIAS_KEY_SUFFIX;
        INDArray mW = params.get(pzxMeanParamName);
        INDArray mB = params.get(pzxBiasParamName);

        INDArray pzxMean = current.mmul(mW).addiRowVector(mB);


        return new VAEFwdHelper(encoderPreOuts, pzxMean, encoderActivations);
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


    @Override
    public void fit() {
        if(input == null){
            throw new IllegalStateException("Cannot fit layer: layer input is null (not set)");
        }

        if(solver == null){
            solver = new Solver.Builder()
                    .model(this).configure(conf()).listeners(getListeners())
                    .build();
            //Set the updater state view array. For MLN and CG, this is done by MultiLayerUpdater and ComputationGraphUpdater respectively
            Updater updater = solver.getOptimizer().getUpdater();
            int updaterStateSize = updater.stateSizeForLayer(this);
            if(updaterStateSize > 0) updater.setStateViewArray(this, Nd4j.createUninitialized(new int[]{1,updaterStateSize},Nd4j.order()), true);
        }
        this.optimizer = solver.getOptimizer();
        solver.optimize();
    }
}
