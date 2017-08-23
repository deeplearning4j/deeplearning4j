package org.deeplearning4j.nn.conf.layers.objdetect;

import lombok.Getter;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BaseOutputLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.params.EmptyParamInitializer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.impl.LossL2;
import org.nd4j.linalg.primitives.Pair;

import java.util.Collection;
import java.util.List;
import java.util.Map;

@Getter
public class Yolo2OutputLayer extends org.deeplearning4j.nn.conf.layers.Layer {

    private double lambdaCoord;
    private double lambdaNoObj;
    private ILossFunction lossPositionScale;
    private ILossFunction lossConfidence;
    private ILossFunction lossClassPredictions;
    private INDArray boundingBoxes;

    private Yolo2OutputLayer(Builder builder){
        super(builder);
        this.lambdaCoord = builder.lambdaCoord;
        this.lambdaNoObj = builder.lambdaNoObj;
        this.lossPositionScale = builder.lossPositionScale;
        this.lossConfidence = builder.lossConfidence;
        this.lossClassPredictions = builder.lossClassPredictions;
        this.boundingBoxes = builder.boundingBoxes;
    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<IterationListener> iterationListeners, int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer ret = new org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer(conf);
        ret.setListeners(iterationListeners);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        return ret;
    }

    @Override
    public ParamInitializer initializer() {
        return EmptyParamInitializer.getInstance();
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        return inputType;
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        //No op
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        switch (inputType.getType()){
            case FF:
            case RNN:
                throw new UnsupportedOperationException("Cannot use FF or RNN input types");
            case CNN:
                return null;
            case CNNFlat:
                InputType.InputTypeConvolutionalFlat cf = (InputType.InputTypeConvolutionalFlat)inputType;
                return new FeedForwardToCnnPreProcessor(cf.getHeight(), cf.getWidth(), cf.getDepth());
            default:
                return null;
        }
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
        //TODO
        return null;
    }

    public static class Builder extends org.deeplearning4j.nn.conf.layers.Layer.Builder<Builder> {

        private double lambdaCoord = 5;
        private double lambdaNoObj = 0.5;
        private ILossFunction lossPositionScale = new LossL2();
        private ILossFunction lossConfidence = new LossL2();
        private ILossFunction lossClassPredictions = new LossL2();
        private INDArray boundingBoxes;

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

        /**
         * Bounding box priors dimensions [height, width] - *as a fraction of  the total image*
         *
         * @param boundingBoxes
         * @return
         */
        public Builder boundingBoxes(INDArray boundingBoxes){
            this.boundingBoxes = boundingBoxes;
            return this;
        }

        @Override
        public Yolo2OutputLayer build() {
            if(boundingBoxes == null){
                throw new IllegalStateException();
            }

            if(boundingBoxes.rank() != 2 || boundingBoxes.size(1) != 2){
                throw new IllegalStateException();
            }

            return new Yolo2OutputLayer(this);
        }
    }
}
