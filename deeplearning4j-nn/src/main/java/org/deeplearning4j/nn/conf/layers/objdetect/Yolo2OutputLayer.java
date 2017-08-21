package org.deeplearning4j.nn.conf.layers.objdetect;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BaseOutputLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.params.EmptyParamInitializer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.impl.LossL2;

import java.util.Collection;

public class Yolo2OutputLayer extends org.deeplearning4j.nn.conf.layers.Layer {

    private double lambdaCoord;
    private double lambdaNoObj;
    private ILossFunction lossPositionScale;
    private ILossFunction lossConfidence;
    private ILossFunction lossClassPredictions;

    private Yolo2OutputLayer(Builder builder){
        super(builder);
        this.lambdaCoord = builder.lambdaCoord;
        this.lambdaNoObj = builder.lambdaNoObj;
        this.lossPositionScale = builder.lossPositionScale;
        this.lossConfidence = builder.lossConfidence;
        this.lossClassPredictions = builder.lossClassPredictions;
    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<IterationListener> iterationListeners, int layerIndex, INDArray layerParamsView, boolean initializeParams) {

    }

    @Override
    public ParamInitializer initializer() {
        return EmptyParamInitializer.getInstance();
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if()
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        //No op
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        return null;
    }

    @Override
    public double getL1ByParam(String paramName) {
        return 0;
    }

    @Override
    public double getL2ByParam(String paramName) {
        return 0;
    }

    @Override
    public double getLearningRateByParam(String paramName) {
        return 0;
    }

    @Override
    public boolean isPretrainParam(String paramName) {
        return false;
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        return null;
    }

    public static class Builder extends org.deeplearning4j.nn.conf.layers.Layer.Builder<Builder> {

        private double lambdaCoord = 5;
        private double lambdaNoObj = 0.5;
        private ILossFunction lossPositionScale = new LossL2();
        private ILossFunction lossConfidence = new LossL2();
        private ILossFunction lossClassPredictions = new LossL2();

        public Builder lambdaCoord(double lambdaCoord){
            this.lambdaCoord = lambdaCoord;
            return this;
        }

        public Builder lambbaNoObj(double lambdaNoObj){
            this.lambdaNoObj = lambdaNoObj;
            return this;
        }

        public Builder lossPositionScale(ILossFunction lossPositionScale){
            this.lossPositionScale = lossPositionScale;
            return this;
        }

        public Builder lossConfidence(ILossFunction lossConfidence){
            this.lossConfidence = lossConfidence;
            return this;
        }

        public Builder lossClassPredictions(ILossFunction lossClassPredictions){
            this.lossClassPredictions = lossClassPredictions;
            return this;
        }

        @Override
        public Yolo2OutputLayer build() {
            return new Yolo2OutputLayer(this);
        }
    }
}
