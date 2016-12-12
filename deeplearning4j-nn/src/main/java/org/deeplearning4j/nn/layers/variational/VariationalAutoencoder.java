package org.deeplearning4j.nn.layers.variational;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.Getter;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.variational.LossFunctionWrapper;
import org.deeplearning4j.nn.conf.layers.variational.ReconstructionDistribution;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.params.VariationalAutoencoderParamInitializer;
import org.deeplearning4j.optimize.Solver;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.blas.Level1;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.*;

import static org.deeplearning4j.nn.params.VariationalAutoencoderParamInitializer.BIAS_KEY_SUFFIX;
import static org.deeplearning4j.nn.params.VariationalAutoencoderParamInitializer.WEIGHT_KEY_SUFFIX;

/**
 * Variational Autoencoder layer
 * <p>
 * See: Kingma & Welling, 2013: Auto-Encoding Variational Bayes - https://arxiv.org/abs/1312.6114
 * <p>
 * This implementation allows multiple encoder and decoder layers, the number and sizes of which can be set independently.
 * <p>
 * A note on scores during pretraining: This implementation minimizes the negative of the variational lower bound objective
 * as described in Kingma & Welling; the mathematics in that paper is based on maximization of the variational lower bound instead.
 * Thus, scores reported during pretraining in DL4J are the negative of the variational lower bound equation in the paper.
 * The backpropagation and learning procedure is otherwise as described there.
 *
 * @author Alex Black
 */
public class VariationalAutoencoder implements Layer {

    protected INDArray input;
    protected INDArray paramsFlattened;
    protected INDArray gradientsFlattened;
    protected Map<String, INDArray> params;
    @Getter
    protected transient Map<String, INDArray> gradientViews;
    protected NeuralNetConfiguration conf;
    protected double score = 0.0;
    protected ConvexOptimizer optimizer;
    protected Gradient gradient;
    protected Collection<IterationListener> iterationListeners = new ArrayList<>();
    protected Collection<TrainingListener> trainingListeners = null;
    protected int index = 0;
    protected INDArray maskArray;
    protected Solver solver;

    protected int[] encoderLayerSizes;
    protected int[] decoderLayerSizes;
    protected ReconstructionDistribution reconstructionDistribution;
    protected String pzxActivationFn;
    protected int numSamples;

    protected boolean zeroedPretrainParamGradients = false;

    public VariationalAutoencoder(NeuralNetConfiguration conf) {
        this.conf = conf;

        this.encoderLayerSizes = ((org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder) conf.getLayer()).getEncoderLayerSizes();
        this.decoderLayerSizes = ((org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder) conf.getLayer()).getDecoderLayerSizes();
        this.reconstructionDistribution = ((org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder) conf.getLayer()).getOutputDistribution();
        this.pzxActivationFn = ((org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder) conf.getLayer()).getPzxActivationFunction();
        this.numSamples = ((org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder) conf.getLayer()).getNumSamples();
    }


    @Override
    public void update(Gradient gradient) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public void update(INDArray gradient, String paramType) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public double score() {
        return score;
    }

    @Override
    public void computeGradientAndScore() {
        //Forward pass through the encoder and mean for P(Z|X)
        VAEFwdHelper fwd = doForward(true, true);
        String afn = conf().getLayer().getActivationFunction();

        //Forward pass through logStd^2 for P(Z|X)
        INDArray pzxLogStd2W = params.get(VariationalAutoencoderParamInitializer.PZX_LOGSTD2_W);
        INDArray pzxLogStd2b = params.get(VariationalAutoencoderParamInitializer.PZX_LOGSTD2_B);

        INDArray pzxLogStd2Pre = fwd.encoderActivations[fwd.encoderActivations.length - 1].mmul(pzxLogStd2W).addiRowVector(pzxLogStd2b);

        INDArray meanZ = fwd.pzxMeanPreOut.dup();
        INDArray logStdev2Z = pzxLogStd2Pre.dup();
        if (!"identity".equals(pzxActivationFn)) {
            Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(pzxActivationFn, meanZ));
            Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(pzxActivationFn, logStdev2Z));
        }

        INDArray pzxSigmaSquared = Transforms.exp(logStdev2Z, true);
        INDArray pzxSigma = Transforms.sqrt(pzxSigmaSquared, true);

        int minibatch = input.size(0);
        int size = fwd.pzxMeanPreOut.size(1);


        Map<String, INDArray> gradientMap = new HashMap<>();
        double scaleFactor = 1.0 / numSamples;
        Level1 blasL1 = Nd4j.getBlasWrapper().level1();
        INDArray[] encoderActivationDerivs = (numSamples > 1 ? new INDArray[encoderLayerSizes.length] : null);
        for( int l=0; l<numSamples; l++ ) { //Default (and in most cases) numSamples == 1
            double gemmCConstant = (l == 0 ? 0.0 : 1.0);        //0 for first one (to get rid of previous buffer data), otherwise 1 (for adding)

            INDArray e = Nd4j.randn(minibatch, size);
            INDArray z = pzxSigma.mul(e).addi(meanZ);      //z = mu + sigma * e, with e ~ N(0,1)


            //Need to do forward pass through decoder layers
            int nDecoderLayers = decoderLayerSizes.length;
            INDArray current = z;
            INDArray[] decoderPreOut = new INDArray[nDecoderLayers];        //Need pre-out for backprop later
            INDArray[] decoderActivations = new INDArray[nDecoderLayers];
            for (int i = 0; i < nDecoderLayers; i++) {
                String wKey = "d" + i + WEIGHT_KEY_SUFFIX;
                String bKey = "d" + i + BIAS_KEY_SUFFIX;

                INDArray weights = params.get(wKey);
                INDArray bias = params.get(bKey);

                current = current.mmul(weights).addiRowVector(bias);
                decoderPreOut[i] = current.dup();
                Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(afn, current));
                decoderActivations[i] = current;
            }

            INDArray pxzw = params.get(VariationalAutoencoderParamInitializer.PXZ_W);
            INDArray pxzb = params.get(VariationalAutoencoderParamInitializer.PXZ_B);

            if(l == 0){
                //Need to add other component of score, in addition to negative log probability
                //Note the negative here vs. the equation in Kingma & Welling: this is because we are minimizing the negative of
                // variational lower bound, rather than maximizing the variational lower bound
                //Unlike log probability (which is averaged over samples) this should be calculated just once
                INDArray temp = meanZ.mul(meanZ).addi(pzxSigmaSquared).negi();
                temp.addi(logStdev2Z).addi(1.0);
                double scorePt1 = -0.5 / minibatch  * temp.sumNumber().doubleValue();
                this.score = scorePt1 + (calcL1(false) + calcL2(false)) / minibatch;
            }

            INDArray pxzDistributionPreOut = current.mmul(pxzw).addiRowVector(pxzb);
            double logPTheta = reconstructionDistribution.negLogProbability(input, pxzDistributionPreOut, true);
            this.score += logPTheta / numSamples;

            //If we have any training listeners (for example, for UI StatsListener - pass on activations)
            if(trainingListeners != null && trainingListeners.size() > 0 && l == 0){        //Note: only doing this on the *first* sample
                Map<String,INDArray> activations = new LinkedHashMap<>();
                for( int i=0; i<fwd.encoderActivations.length; i++ ){
                    activations.put("e" + i, fwd.encoderActivations[i]);
                }
                activations.put(VariationalAutoencoderParamInitializer.PZX_PREFIX,z);
                for( int i=0; i<decoderActivations.length; i++ ){
                    activations.put("d" + i, decoderActivations[i]);
                }
                activations.put(VariationalAutoencoderParamInitializer.PXZ_PREFIX,reconstructionDistribution.generateAtMean(pxzDistributionPreOut));
                for(TrainingListener tl : trainingListeners){
                    tl.onForwardPass(this, activations);
                }
            }

            /////////////////////////////////////////////////////////
            //Backprop

            //First: calculate the gradients at the input to the reconstruction distribution
            INDArray dpdpxz = reconstructionDistribution.gradient(input, pxzDistributionPreOut);

            //Do backprop for output reconstruction distribution -> final decoder layer
            INDArray dLdxzw = gradientViews.get(VariationalAutoencoderParamInitializer.PXZ_W);
            INDArray dLdxzb = gradientViews.get(VariationalAutoencoderParamInitializer.PXZ_B);
            INDArray lastDecActivations = decoderActivations[decoderActivations.length - 1];
            Nd4j.gemm(lastDecActivations, dpdpxz, dLdxzw, true, false, scaleFactor, gemmCConstant);
            if(l == 0){
                dLdxzb.assign(dpdpxz.sum(0));    //TODO: do this without the assign
                if(numSamples > 1){
                    dLdxzb.muli(scaleFactor);
                }
            } else {
                blasL1.axpy(dLdxzb.length(), scaleFactor, dpdpxz.sum(0), dLdxzb);
            }

            gradientMap.put(VariationalAutoencoderParamInitializer.PXZ_W, dLdxzw);
            gradientMap.put(VariationalAutoencoderParamInitializer.PXZ_B, dLdxzb);

            INDArray epsilon = pxzw.mmul(dpdpxz.transpose()).transpose();

            //Next: chain derivatives backwards through the decoder layers
            for (int i = nDecoderLayers - 1; i >= 0; i--) {
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
                    actInput = decoderActivations[i - 1];
                }

                Nd4j.gemm(actInput, currentDelta, dLdW, true, false, scaleFactor, gemmCConstant);

                if(l == 0){
                    dLdB.assign(currentDelta.sum(0));    //TODO: do this without the assign
                    if(numSamples > 1){
                        dLdB.muli(scaleFactor);
                    }
                } else {
                    blasL1.axpy(dLdB.length(), scaleFactor, currentDelta.sum(0), dLdB);
                }

                gradientMap.put(wKey, dLdW);
                gradientMap.put(bKey, dLdB);

                epsilon = weights.mmul(currentDelta.transpose()).transpose();
            }

            //Do backprop through p(z|x)
            INDArray eZXMeanW = params.get(VariationalAutoencoderParamInitializer.PZX_MEAN_W);
            INDArray eZXLogStdev2W = params.get(VariationalAutoencoderParamInitializer.PZX_LOGSTD2_W);

            INDArray dLdz = epsilon;
            //If we were maximizing the equation in Kinga and Welling, this would be a .sub(meanZ). Here: we are minimizing the negative instead
            INDArray dLdmu = dLdz.add(meanZ);

            INDArray dLdLogSigma2 = dLdz.mul(e).muli(pzxSigma)
                    .addi(pzxSigmaSquared).subi(1).muli(0.5);


            INDArray dLdPreMu = dLdmu.mul(Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(
                    pzxActivationFn, fwd.getPzxMeanPreOut().dup()).derivative()));

            INDArray dLdPreLogSigma2 = dLdLogSigma2.mul(Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(
                    pzxActivationFn, pzxLogStd2Pre.dup()).derivative()));

            //Weight gradients for weights feeding into p(z|x)
            INDArray lastEncoderActivation = fwd.encoderActivations[fwd.encoderActivations.length - 1];
            INDArray dLdZXMeanW = gradientViews.get(VariationalAutoencoderParamInitializer.PZX_MEAN_W);
            INDArray dLdZXLogStdev2W = gradientViews.get(VariationalAutoencoderParamInitializer.PZX_LOGSTD2_W);
            Nd4j.gemm(lastEncoderActivation, dLdPreMu, dLdZXMeanW, true, false, scaleFactor, gemmCConstant);
            Nd4j.gemm(lastEncoderActivation, dLdPreLogSigma2, dLdZXLogStdev2W, true, false, scaleFactor, gemmCConstant);

            //Bias gradients for p(z|x)
            INDArray sigmaPrimePreMu = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(
                    pzxActivationFn, fwd.getPzxMeanPreOut().dup()).derivative());
            INDArray dLdZXMeanb = gradientViews.get(VariationalAutoencoderParamInitializer.PZX_MEAN_B);
            INDArray dLdZXLogStdev2b = gradientViews.get(VariationalAutoencoderParamInitializer.PZX_LOGSTD2_B);
            //If we were maximizing the equation in Kinga and Welling, this would be a .sub(meanZ). Here: we are minimizing the negative instead
            if(l == 0){
                dLdZXMeanb.assign(dLdz.add(meanZ).mul(sigmaPrimePreMu).sum(0));
                dLdZXLogStdev2b.assign(dLdPreLogSigma2.sum(0));
                if(numSamples > 1){
                    dLdZXMeanb.muli(scaleFactor);
                    dLdZXLogStdev2b.muli(scaleFactor);
                }
            } else {
                blasL1.axpy(dLdZXMeanb.length(), scaleFactor, dLdz.add(meanZ).mul(sigmaPrimePreMu).sum(0), dLdZXMeanb);
                blasL1.axpy(dLdZXLogStdev2b.length(), scaleFactor, dLdPreLogSigma2.sum(0), dLdZXLogStdev2b);
            }

            gradientMap.put(VariationalAutoencoderParamInitializer.PZX_MEAN_W, dLdZXMeanW);
            gradientMap.put(VariationalAutoencoderParamInitializer.PZX_MEAN_B, dLdZXMeanb);
            gradientMap.put(VariationalAutoencoderParamInitializer.PZX_LOGSTD2_W, dLdZXLogStdev2W);
            gradientMap.put(VariationalAutoencoderParamInitializer.PZX_LOGSTD2_B, dLdZXLogStdev2b);

            //Epsilon (dL/dActivation) at output of the last encoder layer:
            epsilon = Nd4j.gemm(dLdPreMu,eZXMeanW,false,true);      //Equivalent to: epsilon = eZXMeanW.mmul(dLdPreMu.transpose()).transpose(); using   (AxB^T)^T = BxA^T
            //Next line: equivalent to epsilon.addi(eZXLogStdev2W.mmul(dLdPreLogSigma2.transpose()).transpose());       using: (AxB^T)^T = BxA^T
            Nd4j.gemm(dLdPreLogSigma2,eZXLogStdev2W,epsilon,false,true,1.0,1.0);

            //Backprop through encoder:
            int nEncoderLayers = encoderLayerSizes.length;
            for (int i = nEncoderLayers - 1; i >= 0; i--) {
                String wKey = "e" + i + WEIGHT_KEY_SUFFIX;
                String bKey = "e" + i + BIAS_KEY_SUFFIX;

                INDArray weights = params.get(wKey);

                INDArray dLdW = gradientViews.get(wKey);
                INDArray dLdB = gradientViews.get(bKey);

                INDArray preOut = fwd.encoderPreOuts[i];

                INDArray activationDerivative;
                if(numSamples > 1){
                    //Re-use sigma-prime values for the encoder - these don't change based on multiple samples,
                    // only the errors do
                    if(l == 0){
                        encoderActivationDerivs[i] = Nd4j.getExecutioner().execAndReturn(
                        Nd4j.getOpFactory().createTransform(afn, fwd.encoderPreOuts[i]).derivative());
                    }
                    activationDerivative = encoderActivationDerivs[i];
                } else {
                    activationDerivative = Nd4j.getExecutioner().execAndReturn(
                        Nd4j.getOpFactory().createTransform(afn, preOut).derivative());
                }

                INDArray currentDelta = epsilon.muli(activationDerivative);

                INDArray actInput;
                if (i == 0) {
                    actInput = input;
                } else {
                    actInput = fwd.encoderActivations[i - 1];
                }
                Nd4j.gemm(actInput, currentDelta, dLdW, true, false, scaleFactor, gemmCConstant);
                if(l == 0){
                    dLdB.assign(currentDelta.sum(0));    //TODO: do this without the assign
                    if(numSamples > 1){
                        dLdB.muli(scaleFactor);
                    }
                } else {
                    blasL1.axpy(dLdB.length(), scaleFactor, currentDelta.sum(0), dLdB);
                }

                gradientMap.put(wKey, dLdW);
                gradientMap.put(bKey, dLdB);

                epsilon = weights.mmul(currentDelta.transpose()).transpose();
            }
        }

        //Insert the gradients into the Gradient map in the correct order, in case we need to flatten the gradient later
        // to match the parameters iteration order
        Gradient gradient = new DefaultGradient(gradientsFlattened);
        Map<String, INDArray> g = gradient.gradientForVariable();
        for (int i = 0; i < encoderLayerSizes.length; i++) {
            String w = "e" + i + VariationalAutoencoderParamInitializer.WEIGHT_KEY_SUFFIX;
            g.put(w, gradientMap.get(w));
            String b = "e" + i + VariationalAutoencoderParamInitializer.BIAS_KEY_SUFFIX;
            g.put(b, gradientMap.get(b));
        }
        g.put(VariationalAutoencoderParamInitializer.PZX_MEAN_W, gradientMap.get(VariationalAutoencoderParamInitializer.PZX_MEAN_W));
        g.put(VariationalAutoencoderParamInitializer.PZX_MEAN_B, gradientMap.get(VariationalAutoencoderParamInitializer.PZX_MEAN_B));
        g.put(VariationalAutoencoderParamInitializer.PZX_LOGSTD2_W, gradientMap.get(VariationalAutoencoderParamInitializer.PZX_LOGSTD2_W));
        g.put(VariationalAutoencoderParamInitializer.PZX_LOGSTD2_B, gradientMap.get(VariationalAutoencoderParamInitializer.PZX_LOGSTD2_B));
        for (int i = 0; i < decoderLayerSizes.length; i++) {
            String w = "d" + i + VariationalAutoencoderParamInitializer.WEIGHT_KEY_SUFFIX;
            g.put(w, gradientMap.get(w));
            String b = "d" + i + VariationalAutoencoderParamInitializer.BIAS_KEY_SUFFIX;
            g.put(b, gradientMap.get(b));
        }
        g.put(VariationalAutoencoderParamInitializer.PXZ_W, gradientMap.get(VariationalAutoencoderParamInitializer.PXZ_W));
        g.put(VariationalAutoencoderParamInitializer.PXZ_B, gradientMap.get(VariationalAutoencoderParamInitializer.PXZ_B));

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
        for (Map.Entry<String, INDArray> entry : params.entrySet()) {
            if (backwards && isPretrainParam(entry.getKey())) continue;
            ret += entry.getValue().length();
        }
        return ret;
    }

    @Override
    public void setParams(INDArray params) {
        if (params.length() != this.paramsFlattened.length()) {
            throw new IllegalArgumentException("Cannot set parameters: expected parameters vector of length " +
                    this.paramsFlattened.length() + " but got parameters array of length " + params.length());
        }
        this.paramsFlattened.assign(params);
    }

    @Override
    public void setParamsViewArray(INDArray params) {
        if (this.params != null && params.length() != numParams())
            throw new IllegalArgumentException("Invalid input: expect params of length " + numParams()
                    + ", got params of length " + params.length());
        this.paramsFlattened = params;
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray gradients) {
        if (this.params != null && gradients.length() != numParams()) {
            throw new IllegalArgumentException("Invalid input: expect gradients array of length " + numParams()
                    + ", got gradient array of length of length " + gradients.length());
        }

        this.gradientsFlattened = gradients;
        this.gradientViews = conf.getLayer().initializer().getGradientsFromFlattened(conf, gradients);
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
    public Map<String,INDArray> paramTable(boolean backpropParamsOnly){
        Map<String,INDArray> map = new LinkedHashMap<>();
        for(Map.Entry<String,INDArray> e : params.entrySet()){
            if (!backpropParamsOnly || !isPretrainParam(e.getKey())) {
                    map.put(e.getKey(), e.getValue());
            }
        }
        return map;
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

    public boolean isPretrainParam(String param) {
        return !(param.startsWith("e") || param.startsWith(VariationalAutoencoderParamInitializer.PZX_MEAN_PREFIX));
    }

    @Override
    public double calcL2(boolean backpropParamsOnly) {
        if (!conf.isUseRegularization() || conf.getLayer().getL2() <= 0.0) return 0.0;

        double l2Sum = 0.0;
        for (Map.Entry<String, INDArray> e : paramTable().entrySet()) {
            double l2 = conf().getL2ByParam(e.getKey());
            if (l2 <= 0.0 || (backpropParamsOnly && isPretrainParam(e.getKey()))) {
                continue;
            }

            double l2Norm = e.getValue().norm2Number().doubleValue();
            l2Sum += 0.5 * l2 * l2Norm * l2Norm;
        }

        return l2Sum;
    }

    @Override
    public double calcL1(boolean backpropParamsOnly) {
        if (!conf.isUseRegularization() || conf.getLayer().getL1() <= 0.0) return 0.0;

        double l1Sum = 0.0;
        for (Map.Entry<String, INDArray> e : paramTable().entrySet()) {
            double l1 = conf().getL1ByParam(e.getKey());
            if (l1 <= 0.0 || (backpropParamsOnly && isPretrainParam(e.getKey()))) {
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
        if (!zeroedPretrainParamGradients) {
            for (Map.Entry<String, INDArray> entry : gradientViews.entrySet()) {
                if (isPretrainParam(entry.getKey())) {
                    entry.getValue().assign(0);
                }
            }
            zeroedPretrainParamGradients = true;
        }

        Gradient gradient = new DefaultGradient();

        VAEFwdHelper fwd = doForward(true, true);
        INDArray activationDerivative = Nd4j.getExecutioner().execAndReturn(
                Nd4j.getOpFactory().createTransform(pzxActivationFn, fwd.pzxMeanPreOut).derivative());

        INDArray currentDelta = activationDerivative.muli(epsilon);

        //Finally, calculate mean value:
        INDArray meanW = params.get(VariationalAutoencoderParamInitializer.PZX_MEAN_W);
        INDArray dLdMeanW = gradientViews.get(VariationalAutoencoderParamInitializer.PZX_MEAN_W); //f order
        INDArray lastEncoderActivation = fwd.encoderActivations[fwd.encoderActivations.length - 1];
        Nd4j.gemm(lastEncoderActivation, currentDelta, dLdMeanW, true, false, 1.0, 0.0);
        INDArray dLdMeanB = gradientViews.get(VariationalAutoencoderParamInitializer.PZX_MEAN_B);
        dLdMeanB.assign(currentDelta.sum(0));    //TODO: do this without the assign

        gradient.gradientForVariable().put(VariationalAutoencoderParamInitializer.PZX_MEAN_W, dLdMeanW);
        gradient.gradientForVariable().put(VariationalAutoencoderParamInitializer.PZX_MEAN_B, dLdMeanB);

        epsilon = meanW.mmul(currentDelta.transpose()).transpose();

        int nEncoderLayers = encoderLayerSizes.length;

        String afn = conf().getLayer().getActivationFunction();
        for (int i = nEncoderLayers - 1; i >= 0; i--) {
            String wKey = "e" + i + WEIGHT_KEY_SUFFIX;
            String bKey = "e" + i + BIAS_KEY_SUFFIX;

            INDArray weights = params.get(wKey);

            INDArray dLdW = gradientViews.get(wKey);
            INDArray dLdB = gradientViews.get(bKey);

            INDArray preOut = fwd.encoderPreOuts[i];
            activationDerivative = Nd4j.getExecutioner().execAndReturn(
                    Nd4j.getOpFactory().createTransform(afn, preOut).derivative());

            currentDelta = epsilon.muli(activationDerivative);

            INDArray actInput;
            if (i == 0) {
                actInput = input;
            } else {
                actInput = fwd.encoderActivations[i - 1];
            }
            Nd4j.gemm(actInput, currentDelta, dLdW, true, false, 1.0, 0.0);
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
        return f.pzxMeanPreOut;
    }

    @AllArgsConstructor
    @Data
    private static class VAEFwdHelper {
        private INDArray[] encoderPreOuts;
        private INDArray pzxMeanPreOut;
        private INDArray[] encoderActivations;
    }


    private VAEFwdHelper doForward(boolean training, boolean forBackprop) {
        if (input == null) {
            throw new IllegalStateException("Cannot do forward pass with null input");
        }

        //TODO input validation

        int nEncoderLayers = encoderLayerSizes.length;

        INDArray[] encoderPreOuts = new INDArray[encoderLayerSizes.length];
        INDArray[] encoderActivations = new INDArray[encoderLayerSizes.length];
        INDArray current = input;
        for (int i = 0; i < nEncoderLayers; i++) {
            String wKey = "e" + i + WEIGHT_KEY_SUFFIX;
            String bKey = "e" + i + BIAS_KEY_SUFFIX;

            INDArray weights = params.get(wKey);
            INDArray bias = params.get(bKey);

            current = current.mmul(weights).addiRowVector(bias);
            if (forBackprop) {
                encoderPreOuts[i] = current.dup();
            }
            Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getLayer().getActivationFunction(), current));
            encoderActivations[i] = current;
        }

        //Finally, calculate mean value:
        INDArray mW = params.get(VariationalAutoencoderParamInitializer.PZX_MEAN_W);
        INDArray mB = params.get(VariationalAutoencoderParamInitializer.PZX_MEAN_B);

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

        if (!"identity".equals(pzxActivationFn)) {
            Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(pzxActivationFn, output));
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
        if (iterationListeners == null) return null;
        return new ArrayList<>(iterationListeners);
    }

    @Override
    public void setListeners(IterationListener... listeners) {
        setListeners(Arrays.<IterationListener>asList(listeners));
    }

    @Override
    public void setListeners(Collection<IterationListener> listeners) {
        if (iterationListeners == null) iterationListeners = new ArrayList<>();
        else iterationListeners.clear();
        if (trainingListeners == null) trainingListeners = new ArrayList<>();
        else trainingListeners.clear();

        if(listeners != null && iterationListeners.size() > 0) {
            iterationListeners.addAll(listeners);
            for (IterationListener il : listeners) {
                if (il instanceof TrainingListener) {
                    trainingListeners.add((TrainingListener) il);
                }
            }
        }
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
        return input.size(0);
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
    public boolean isPretrainLayer() {
        return true;
    }


    @Override
    public void fit() {
        if (input == null) {
            throw new IllegalStateException("Cannot fit layer: layer input is null (not set)");
        }

        if (solver == null) {
            solver = new Solver.Builder()
                    .model(this).configure(conf()).listeners(getListeners())
                    .build();
            //Set the updater state view array. For MLN and CG, this is done by MultiLayerUpdater and ComputationGraphUpdater respectively
            Updater updater = solver.getOptimizer().getUpdater();
            int updaterStateSize = updater.stateSizeForLayer(this);
            if (updaterStateSize > 0) {
                updater.setStateViewArray(this, Nd4j.createUninitialized(new int[]{1, updaterStateSize}, Nd4j.order()), true);
            }
        }
        this.optimizer = solver.getOptimizer();
        solver.optimize();
    }

    /**
     * Calculate the reconstruction probability, as described in An & Cho, 2015 - "Variational Autoencoder based
     * Anomaly Detection using Reconstruction Probability" (Algorithm 4)<br>
     * The authors describe it as follows: "This is essentially the probability of the data being generated from a given
     * latent variable drawn from the approximate posterior distribution."<br>
     * <br>
     * Specifically, for each example x in the input, calculate p(x). Note however that p(x) is a stochastic (Monte-Carlo)
     * estimate of the true p(x), based on the specified number of samples. More samples will produce a more accurate
     * (lower variance) estimate of the true p(x) for the current model parameters.<br>
     * <br>
     * Internally uses {@link #reconstructionLogProbability(INDArray, int)} for the actual implementation.
     * That method may be more numerically stable in some cases.<br>
     * <br>
     * The returned array is a column vector of reconstruction probabilities, for each example. Thus, reconstruction probabilities
     * can (and should, for efficiency) be calculated in a batched manner.
     *
     * @param data       The data to calculate the reconstruction probability for
     * @param numSamples Number of samples with which to base the reconstruction probability on.
     * @return Column vector of reconstruction probabilities for each example (shape: [numExamples,1])
     */
    public INDArray reconstructionProbability(INDArray data, int numSamples) {
        INDArray reconstructionLogProb = reconstructionLogProbability(data, numSamples);
        return Transforms.exp(reconstructionLogProb, false);
    }

    /**
     * Return the log reconstruction probability given the specified number of samples.<br>
     * See {@link #reconstructionLogProbability(INDArray, int)} for more details
     *
     * @param data       The data to calculate the log reconstruction probability
     * @param numSamples Number of samples with which to base the reconstruction probability on.
     * @return Column vector of reconstruction log probabilities for each example (shape: [numExamples,1])
     */
    public INDArray reconstructionLogProbability(INDArray data, int numSamples) {
        if(numSamples <= 0){
            throw new IllegalArgumentException("Invalid input: numSamples must be > 0. Got: " + numSamples);
        }
        if(reconstructionDistribution instanceof LossFunctionWrapper){
            throw new UnsupportedOperationException("Cannot calculate reconstruction log probability when using "
                    + "a LossFunction (via LossFunctionWrapper) instead of a ReconstructionDistribution: ILossFunction "
                    + "instances are not in general probabilistic, hence it is not possible to calculate reconstruction probability");
        }

        //Forward pass through the encoder and mean for P(Z|X)
        setInput(data);
        VAEFwdHelper fwd = doForward(true, true);
        String afn = conf().getLayer().getActivationFunction();

        //Forward pass through logStd^2 for P(Z|X)
        INDArray pzxLogStd2W = params.get(VariationalAutoencoderParamInitializer.PZX_LOGSTD2_W);
        INDArray pzxLogStd2b = params.get(VariationalAutoencoderParamInitializer.PZX_LOGSTD2_B);

        INDArray meanZ = fwd.pzxMeanPreOut;
        INDArray logStdev2Z = fwd.encoderActivations[fwd.encoderActivations.length - 1].mmul(pzxLogStd2W).addiRowVector(pzxLogStd2b);
        if (!"identity".equals(pzxActivationFn)) {
            Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(pzxActivationFn, meanZ));
            Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(pzxActivationFn, logStdev2Z));
        }

        INDArray pzxSigma = Transforms.exp(logStdev2Z, false);
        Transforms.sqrt(pzxSigma, false);

        int minibatch = input.size(0);
        int size = fwd.pzxMeanPreOut.size(1);

        INDArray pxzw = params.get(VariationalAutoencoderParamInitializer.PXZ_W);
        INDArray pxzb = params.get(VariationalAutoencoderParamInitializer.PXZ_B);

        INDArray[] decoderWeights = new INDArray[decoderLayerSizes.length];
        INDArray[] decoderBiases = new INDArray[decoderLayerSizes.length];

        for (int i = 0; i < decoderLayerSizes.length; i++) {
            String wKey = "d" + i + WEIGHT_KEY_SUFFIX;
            String bKey = "d" + i + BIAS_KEY_SUFFIX;
            decoderWeights[i] = params.get(wKey);
            decoderBiases[i] = params.get(bKey);
        }

        INDArray sumReconstructionNegLogProbability = null;
        for( int i=0; i<numSamples; i++ ) {
            INDArray e = Nd4j.randn(minibatch, size);
            INDArray z = e.muli(pzxSigma).addi(meanZ);      //z = mu + sigma * e, with e ~ N(0,1)

            //Do forward pass through decoder
            int nDecoderLayers = decoderLayerSizes.length;
            INDArray currentActivations = z;
            for (int j = 0; j < nDecoderLayers; j++) {
                currentActivations = currentActivations.mmul(decoderWeights[j]).addiRowVector(decoderBiases[j]);
                Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(afn, currentActivations));
            }

            //And calculate reconstruction distribution preOut
            INDArray pxzDistributionPreOut = currentActivations.mmul(pxzw).addiRowVector(pxzb);

            if(i == 0){
                sumReconstructionNegLogProbability = reconstructionDistribution.exampleNegLogProbability(data, pxzDistributionPreOut);
            } else {
                sumReconstructionNegLogProbability.addi(reconstructionDistribution.exampleNegLogProbability(data, pxzDistributionPreOut));
            }
        }

        setInput(null);
        return sumReconstructionNegLogProbability.divi(-numSamples);
    }

    /**
     * Given a specified values for the latent space as input (latent space being z in p(z|data)), generate output
     * from P(x|z), where x = E[P(x|z)]<br>
     * i.e., return the mean value for the distribution P(x|z)
     *
     * @param latentSpaceValues    Values for the latent space. size(1) must equal nOut configuration parameter
     * @return Sample of data: E[P(x|z)]
     */
    public INDArray generateAtMeanGivenZ(INDArray latentSpaceValues){
        INDArray pxzDistributionPreOut = decodeGivenLatentSpaceValues(latentSpaceValues);
        return reconstructionDistribution.generateAtMean(pxzDistributionPreOut);
    }

    /**
     * Given a specified values for the latent space as input (latent space being z in p(z|data)), randomly generate output
     * x, where x ~ P(x|z)
     *
     * @param latentSpaceValues    Values for the latent space. size(1) must equal nOut configuration parameter
     * @return Sample of data: x ~ P(x|z)
     */
    public INDArray generateRandomGivenZ(INDArray latentSpaceValues){
        INDArray pxzDistributionPreOut = decodeGivenLatentSpaceValues(latentSpaceValues);
        return reconstructionDistribution.generateRandom(latentSpaceValues);
    }

    private INDArray decodeGivenLatentSpaceValues(INDArray latentSpaceValues){
        if(latentSpaceValues.size(1) != params.get(VariationalAutoencoderParamInitializer.PZX_MEAN_W).size(1)){
            throw new IllegalArgumentException("Invalid latent space values: expected size " + params.get(VariationalAutoencoderParamInitializer.PZX_MEAN_W).size(1)
                    + ", got size (dimension 1) = " + latentSpaceValues.size(1));
        }

        //Do forward pass through decoder

        int nDecoderLayers = decoderLayerSizes.length;
        INDArray currentActivations = latentSpaceValues;
        String afn = conf().getLayer().getActivationFunction();

        for (int i = 0; i < nDecoderLayers; i++) {
            String wKey = "d" + i + WEIGHT_KEY_SUFFIX;
            String bKey = "d" + i + BIAS_KEY_SUFFIX;
            INDArray w = params.get(wKey);
            INDArray b = params.get(bKey);
            currentActivations = currentActivations.mmul(w).addiRowVector(b);
            Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(afn, currentActivations));
        }

        INDArray pxzw = params.get(VariationalAutoencoderParamInitializer.PXZ_W);
        INDArray pxzb = params.get(VariationalAutoencoderParamInitializer.PXZ_B);
        return currentActivations.mmul(pxzw).addiRowVector(pxzb);
    }
}
