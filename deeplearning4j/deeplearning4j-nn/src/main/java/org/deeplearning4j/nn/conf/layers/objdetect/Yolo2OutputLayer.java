package org.deeplearning4j.nn.conf.layers.objdetect;

import lombok.Data;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.params.EmptyParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.impl.LossL2;
import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;
import org.nd4j.shade.jackson.databind.annotation.JsonSerialize;
import org.nd4j.shade.serde.jackson.VectorDeSerializer;
import org.nd4j.shade.serde.jackson.VectorSerializer;

import java.util.Arrays;
import java.util.Collection;
import java.util.Map;

/**
 * Output (loss) layer for YOLOv2 object detection model, based on the papers:
 * YOLO9000: Better, Faster, Stronger - Redmon & Farhadi (2016) - https://arxiv.org/abs/1612.08242<br>
 * and<br>
 * You Only Look Once: Unified, Real-Time Object Detection - Redmon et al. (2016) -
 * http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf<br>
 *
 * This loss function implementation is based on the YOLOv2 version of the paper. However, note that it doesn't
 * currently support simultaneous training on both detection and classification datasets as described in the
 * YOlO9000 paper.
 *
 * Note: Input activations to the Yolo2OutputLayer should have shape: [minibatch, b*(5+c), H, W], where:<br>
 * b = number of bounding boxes (determined by config)<br>
 * c = number of classes<br>
 * H = output/label height<br>
 * W = output/label width<br>
 *
 * @author Alex Black
 */
@Data
public class Yolo2OutputLayer extends org.deeplearning4j.nn.conf.layers.Layer {

    private double lambdaCoord;
    private double lambdaNoObj;
    private ILossFunction lossPositionScale;
    private ILossFunction lossClassPredictions;
    @JsonSerialize(using = VectorSerializer.class)
    @JsonDeserialize(using = VectorDeSerializer.class)
    private INDArray boundingBoxes;

    private Yolo2OutputLayer(){
        //No-arg costructor for Jackson JSON
    }

    private Yolo2OutputLayer(Builder builder){
        super(builder);
        this.lambdaCoord = builder.lambdaCoord;
        this.lambdaNoObj = builder.lambdaNoObj;
        this.lossPositionScale = builder.lossPositionScale;
        this.lossClassPredictions = builder.lossClassPredictions;
        this.boundingBoxes = builder.boundingBoxes;
    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners, int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer ret = new org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer(conf);
        ret.setListeners(trainingListeners);
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
        return inputType;   //Same shape output as input
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
    public boolean isPretrain() {
        return false;
    }

    @Override
    public double getL1ByParam(String paramName) {
        return 0;   //No params
    }

    @Override
    public double getL2ByParam(String paramName) {
        return 0;   //No params
    }

    @Override
    public boolean isPretrainParam(String paramName) {
        return false;   //No params
    }

    @Override
    public GradientNormalization getGradientNormalization() {
        return GradientNormalization.None;
    }

    @Override
    public double getGradientNormalizationThreshold() {
        return 1.0;
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        long numValues = inputType.arrayElementsPerExample();

        //This is a VERY rough estimate...
        return new LayerMemoryReport.Builder(layerName, Yolo2OutputLayer.class, inputType, inputType)
                .standardMemory(0, 0)   //No params
                .workingMemory(0, numValues, 0, 6 * numValues)
                .cacheMemory(0, 0)  //No cache
                .build();
    }

    public static class Builder extends org.deeplearning4j.nn.conf.layers.Layer.Builder<Builder> {

        private double lambdaCoord = 5;
        private double lambdaNoObj = 0.5;
        private ILossFunction lossPositionScale = new LossL2();
        private ILossFunction lossClassPredictions = new LossL2();
        private INDArray boundingBoxes;

        /**
         * Loss function coefficient for position and size/scale components of the loss function.
         * Default (as per paper): 5
         *
         * @param lambdaCoord Lambda value for size/scale component of loss function
         */
        public Builder lambdaCoord(double lambdaCoord){
            this.lambdaCoord = lambdaCoord;
            return this;
        }

        /**
         * Loss function coefficient for the "no object confidence" components of the loss function.
         * Default (as per paper): 0.5
         *
         * @param lambdaNoObj Lambda value for no-object (confidence) component of the loss function
         */
        public Builder lambbaNoObj(double lambdaNoObj){
            this.lambdaNoObj = lambdaNoObj;
            return this;
        }

        /**
         * Loss function for position/scale component of the loss function
         *
         * @param lossPositionScale Loss function for position/scale
         */
        public Builder lossPositionScale(ILossFunction lossPositionScale){
            this.lossPositionScale = lossPositionScale;
            return this;
        }

        /**
         * Loss function for the class predictions - defaults to L2 loss (i.e., sum of squared errors, as per the
         * paper), however Loss MCXENT could also be used (which is more common for classification).
         *
         * @param lossClassPredictions Loss function for the class prediction error component of the YOLO loss function
         */
        public Builder lossClassPredictions(ILossFunction lossClassPredictions){
            this.lossClassPredictions = lossClassPredictions;
            return this;
        }

        /**
         * Bounding box priors dimensions [width, height]. For N bounding boxes, input has shape [rows, columns] = [N, 2]
         * Note that dimensions should be specified as fraction of grid size. For example, a network with 13x13 output,
         * a value of 1.0 would correspond to one grid cell; a value of 13 would correspond to the entire image.
         *
         * @param boundingBoxes Bounding box prior dimensions (width, height)
         */
        public Builder boundingBoxPriors(INDArray boundingBoxes){
            this.boundingBoxes = boundingBoxes;
            return this;
        }

        @Override
        public Yolo2OutputLayer build() {
            if(boundingBoxes == null){
                throw new IllegalStateException("Bounding boxes have not been set");
            }

            if(boundingBoxes.rank() != 2 || boundingBoxes.size(1) != 2){
                throw new IllegalStateException("Bounding box priors must have shape [nBoxes, 2]. Has shape: "
                        + Arrays.toString(boundingBoxes.shape()));
            }

            return new Yolo2OutputLayer(this);
        }
    }
}
