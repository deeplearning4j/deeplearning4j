package org.deeplearning4j.nn.conf.layers.samediff;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.params.SameDiffParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.IUpdater;

import java.util.Collection;
import java.util.Map;

@Slf4j
@Data
@EqualsAndHashCode(callSuper = true)
public abstract class AbstractSameDiffLayer extends Layer {

    protected double l1;
    protected double l2;
    protected double l1Bias;
    protected double l2Bias;
    protected IUpdater updater;
    protected IUpdater biasUpdater;
    protected GradientNormalization gradientNormalization;
    protected double gradientNormalizationThreshold = Double.NaN;

    private SDLayerParams layerParams;

    protected AbstractSameDiffLayer(Builder builder){
        super(builder);
        this.l1 = builder.l1;
        this.l2 = builder.l2;
        this.l1Bias = builder.l1Bias;
        this.l2Bias = builder.l2Bias;
        this.updater = builder.updater;
        this.biasUpdater = builder.biasUpdater;

        //Check that this class has a no-arg constructor for JSON: better to detect this now provide useful information
        // to pre-empt a failure later for users, which will have a more difficult to understand message
        try{
            getClass().getDeclaredConstructor();
        } catch (NoSuchMethodException e){
            log.warn("***SameDiff layer {} does not have a zero argument (no-arg) constructor.***\nA no-arg constructor " +
                    "is required for JSON deserialization, which is used for both model saving and distributed (Spark) " +
                    "training.\nA no-arg constructor (private, protected or public) as well as setters (or simply a " +
                    "Lombok @Data annotation) should be added to avoid JSON errors later.", getClass().getName());
        } catch (SecurityException e){
            //Ignore
        }
    }

    protected AbstractSameDiffLayer(){
        //No op constructor for Jackson
    }

    public SDLayerParams getLayerParams(){
        if(layerParams == null){
            layerParams = new SDLayerParams();
            defineParameters(layerParams);
        }
        return layerParams;
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        //Default implementation: no-op
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        //Default implementation: no-op
        return null;
    }


    public void applyGlobalConfigToLayer(NeuralNetConfiguration.Builder globalConfig) {
        //Default implementation: no op
    }

    /**
     * Define the parameters for the network. Use {@link SDLayerParams#addWeightParam(String, long...)} and
     * {@link SDLayerParams#addBiasParam(String, long...)}
     * @param params Object used to set parameters for this layer
     */
    public abstract void defineParameters(SDLayerParams params);

    /**
     * Set the initial parameter values for this layer, if required
     * @param params Parameter arrays that may be initialized
     */
    public abstract void initializeParameters(Map<String,INDArray> params);

    @Override
    public abstract org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners,
                                                                int layerIndex, INDArray layerParamsView, boolean initializeParams);

    //==================================================================================================================

    @Override
    public ParamInitializer initializer() {
        return SameDiffParamInitializer.getInstance();
    }

    @Override
    public double getL1ByParam(String paramName) {
        return (initializer().isWeightParam(this, paramName) ? l1 :  l1Bias);
    }

    @Override
    public double getL2ByParam(String paramName) {
        return (initializer().isWeightParam(this, paramName) ? l2 :  l2Bias);
    }

    @Override
    public IUpdater getUpdaterByParam(String paramName){
        if(biasUpdater != null && initializer().isBiasParam(this, paramName)){
            return biasUpdater;
        } else if(initializer().isBiasParam(this, paramName) || initializer().isWeightParam(this, paramName)){
            return updater;
        }
        throw new IllegalStateException("Unknown parameter key: " + paramName);
    }

    @Override
    public boolean isPretrainParam(String paramName) {
        return false;
    }

    @Override
    public boolean isPretrain() {
        return false;
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        return new LayerMemoryReport(); //TODO
    }

    /**
     * Returns the memory layout ('c' or 'f' order - i.e., row/column major) of the parameters. In most cases,
     * this can/should be left
     * @param param Name of the parameter
     * @return Memory layout ('c' or 'f') of the parameter
     */
    public char paramReshapeOrder(String param){
        return 'c';
    }

    protected void initWeights(int fanIn, int fanOut, WeightInit weightInit, INDArray array){
        WeightInitUtil.initWeights(fanIn, fanOut, array.shape(), weightInit, null, paramReshapeOrder(null), array);
    }

    public void applyGlobalConfig(NeuralNetConfiguration.Builder b){
        if(Double.isNaN(l1)){
            l1 = b.getL1();
        }
        if(Double.isNaN(l2)){
            l2 = b.getL2();
        }
        if(Double.isNaN(l1Bias)){
            l1Bias = b.getL1Bias();
        }
        if(Double.isNaN(l2Bias)){
            l2Bias = b.getL2Bias();
        }
        if(updater == null){
            updater = b.getIUpdater();
        }
        if(biasUpdater == null){
            biasUpdater = b.getBiasUpdater();
        }
        if(gradientNormalization == null){
            gradientNormalization = b.getGradientNormalization();
        }
        if(Double.isNaN(gradientNormalizationThreshold)){
            gradientNormalizationThreshold = b.getGradientNormalizationThreshold();
        }

        applyGlobalConfigToLayer(b);
    }

    public static abstract class Builder<T extends Builder<T>> extends Layer.Builder<T> {

        protected double l1 = Double.NaN;
        protected double l2 = Double.NaN;
        protected double l1Bias = Double.NaN;
        protected double l2Bias = Double.NaN;
        protected IUpdater updater = null;
        protected IUpdater biasUpdater = null;

        /**
         * L1 regularization coefficient (weights only). Use {@link #l1Bias(double)} to configure the l1 regularization
         * coefficient for the bias.
         */
        public T l1(double l1) {
            this.l1 = l1;
            return (T) this;
        }

        /**
         * L2 regularization coefficient (weights only). Use {@link #l2Bias(double)} to configure the l2 regularization
         * coefficient for the bias.
         */
        public T l2(double l2) {
            this.l2 = l2;
            return (T) this;
        }

        /**
         * L1 regularization coefficient for the bias. Default: 0. See also {@link #l1(double)}
         */
        public T l1Bias(double l1Bias) {
            this.l1Bias = l1Bias;
            return (T) this;
        }

        /**
         * L2 regularization coefficient for the bias. Default: 0. See also {@link #l2(double)}
         */
        public T l2Bias(double l2Bias) {
            this.l2Bias = l2Bias;
            return (T) this;
        }

        /**
         * Gradient updater. For example, {@link org.nd4j.linalg.learning.config.Adam}
         * or {@link org.nd4j.linalg.learning.config.Nesterovs}
         *
         * @param updater Updater to use
         */
        public T updater(IUpdater updater) {
            this.updater = updater;
            return (T) this;
        }

        /**
         * Gradient updater configuration, for the biases only. If not set, biases will use the updater as
         * set by {@link #updater(IUpdater)}
         *
         * @param biasUpdater Updater to use for bias parameters
         */
        public T biasUpdater(IUpdater biasUpdater){
            this.biasUpdater = biasUpdater;
            return (T) this;
        }
    }
}
