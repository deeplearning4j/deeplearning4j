package org.deeplearning4j.nn.conf.layers.samediff;

import lombok.Data;
import org.deeplearning4j.nn.api.TrainingConfig;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.samediff.SameDiffGraphVertex;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Map;

@Data
public abstract class SameDiffVertex extends GraphVertex implements TrainingConfig {

    private SDVertexParams vertexParams;
    private String name;

    protected double l1 = Double.NaN;
    protected double l2 = Double.NaN;
    protected double l1Bias = Double.NaN;
    protected double l2Bias = Double.NaN;
    protected IUpdater updater;
    protected IUpdater biasUpdater;
    protected GradientNormalization gradientNormalization;
    protected double gradientNormalizationThreshold = Double.NaN;

    public abstract SDVariable defineVertex(SameDiff sameDiff, Map<String,SDVariable> layerInput, Map<String,SDVariable> paramTable);

    /**
     * Define the parameters - and inputs - for the network.
     * Use {@link SDVertexParams#addWeightParam(String, long...)} and
     * {@link SDVertexParams#addBiasParam(String, long...)}.
     * Note also you must define (and optionally name) the inputs to the vertex. This is required so that
     * DL4J knows how many inputs exists for the vertex.
     * @param params Object used to set parameters for this layer
     */
    public abstract void defineParametersAndInputs(SDVertexParams params);

    /**
     * Set the initial parameter values for this layer, if required
     * @param params Parameter arrays that may be initialized
     */
    public abstract void initializeParameters(Map<String,INDArray> params);

    public SDVertexParams getVertexParams(){
        if(vertexParams == null){
            vertexParams = new SDVertexParams();
            defineParametersAndInputs(vertexParams);
        }
        return vertexParams;
    }

    @Override
    public GraphVertex clone() {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public int numParams(boolean backprop) {
        SDLayerParams params = getVertexParams();
        long count = 0;
        for(long[] l : params.getParamShapes().values()){
            count += ArrayUtil.prodLong(l);
        }
        return (int)count;
    }

    @Override
    public int minVertexInputs() {
        return 1;
    }

    @Override
    public int maxVertexInputs() {
        return -1;
    }

    @Override
    public org.deeplearning4j.nn.graph.vertex.GraphVertex instantiate(ComputationGraph graph, String name, int idx,
                                                                      INDArray paramsView, boolean initializeParams) {
        this.name = name;
        return new SameDiffGraphVertex(this, graph, name, idx, paramsView, initializeParams );
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType... vertexInputs) throws InvalidInputTypeException {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public MemoryReport getMemoryReport(InputType... inputTypes) {
        return null;
    }


    public char paramReshapeOrder(String paramName){
        return 'c';
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

    public void applyGlobalConfigToLayer(NeuralNetConfiguration.Builder globalConfig) {
        //Default implementation: no op
    }


    @Override
    public String getLayerName() {
        return name;
    }

    @Override
    public boolean isPretrain() {
        return false;
    }

    @Override
    public double getL1ByParam(String paramName) {
        if(l1 == 0.0 && l1Bias == 0.0 ){
            return 0.0;
        }
        if(getVertexParams().isWeightParam(paramName)){
            return l1;
        }
        if(getVertexParams().isBiasParam(paramName)){
            return l1Bias;
        }
        throw new IllegalStateException("Unknown parameter name: " + paramName + " - not in weights (" + getVertexParams().getWeightParameterKeys()
                + ") or biases (" + getVertexParams().getBiasParameterKeys() + ")");
    }

    @Override
    public double getL2ByParam(String paramName) {
        if(l2 == 0.0 && l2Bias == 0.0 ){
            return 0.0;
        }
        if(getVertexParams().isWeightParam(paramName)){
            return l2;
        }
        if(getVertexParams().isBiasParam(paramName)){
            return l2Bias;
        }
        throw new IllegalStateException("Unknown parameter name: " + paramName + " - not in weights (" + getVertexParams().getWeightParameterKeys()
                + ") or biases (" + getVertexParams().getBiasParameterKeys() + ")");
    }

    @Override
    public boolean isPretrainParam(String paramName) {
        return false;
    }

    @Override
    public IUpdater getUpdaterByParam(String paramName) {
        if(getVertexParams().isWeightParam(paramName)){
            return updater;
        }
        if(getVertexParams().isBiasParam(paramName)){
            if(biasUpdater == null){
                return updater;
            }
            return biasUpdater;
        }
        throw new IllegalStateException("Unknown parameter name: " + paramName + " - not in weights (" + getVertexParams().getWeightParameterKeys()
                + ") or biases (" + getVertexParams().getBiasParameterKeys() + ")");
    }

    @Override
    public GradientNormalization getGradientNormalization() {
        return gradientNormalization;
    }

    @Override
    public double getGradientNormalizationThreshold() {
        return gradientNormalizationThreshold;
    }

    @Override
    public void setPretrain(boolean pretrain) {
        //No op
    }
}
