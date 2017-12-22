package org.deeplearning4j.nn.conf.layers.samediff;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.layers.samediff.SameDiffLayer;
import org.deeplearning4j.nn.params.SameDiffParamInitializer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.IUpdater;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;

@Data
@EqualsAndHashCode(callSuper = true)
public abstract class BaseSameDiffLayer extends Layer {

    protected double l1;
    protected double l2;
    protected double l1Bias;
    protected double l2Bias;
    protected IUpdater updater;
    protected IUpdater biasUpdater;


    private List<String> paramKeys;

    protected BaseSameDiffLayer(Builder builder){
        super(builder);
        this.l1 = builder.l1;
        this.l2 = builder.l2;
        this.l1Bias = builder.l1Bias;
        this.l2Bias = builder.l2Bias;
        this.updater = builder.updater;
        this.biasUpdater = builder.biasUpdater;
    }

    protected BaseSameDiffLayer(){
        //No op constructor for Jackson
    }

    @Override
    public abstract InputType getOutputType(int layerIndex, InputType inputType);

    @Override
    public abstract void setNIn(InputType inputType, boolean override);

    @Override
    public abstract InputPreProcessor getPreProcessorForInputType(InputType inputType);

    public abstract List<String> weightKeys();

    public abstract List<String> biasKeys();

    public abstract Map<String,int[]> paramShapes();

    public abstract List<String> defineLayer(SameDiff sameDiff, SDVariable layerInput, Map<String,SDVariable> paramTable);

    public abstract void applyGlobalConfigToLayer(NeuralNetConfiguration.Builder globalConfig);

    //==================================================================================================================

    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf, Collection<IterationListener> iterationListeners,
                                                       int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        SameDiffLayer ret = new SameDiffLayer(conf);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        return ret;
    }

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
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        return new LayerMemoryReport(); //TODO
    }

    public List<String> paramKeys(){
        if(paramKeys == null){
            List<String> pk = new ArrayList<>();
            pk.addAll(weightKeys());
            pk.addAll(biasKeys());
            paramKeys = pk;
        }
        return paramKeys;
    }

    public char paramReshapeOrder(String param){
        return 'f';
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
