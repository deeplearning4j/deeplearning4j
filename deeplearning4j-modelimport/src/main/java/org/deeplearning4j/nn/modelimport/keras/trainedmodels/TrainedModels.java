package org.deeplearning4j.nn.modelimport.keras.trainedmodels;


import org.deeplearning4j.nn.modelimport.keras.trainedmodels.Utils.ImageNetLabels;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;

/**
 * Support for popular trained image-classification models
 * @author susaneraly
 * @deprecated Please use the new module deeplearning4j-zoo and instantiate pretrained models from the zoo directly.
 */
public enum TrainedModels {

    VGG16, VGG16NOTOP;

    /**
     * Name of the sub dir in the local cache associated with the model.
     * Local cache dir ~/.dl4j/trainedmodels
     */
    protected String getModelDir() {
        switch (this) {
            case VGG16:
                return "vgg16";
            case VGG16NOTOP:
                return "vgg16notop";
            default:
                throw new UnsupportedOperationException("Unknown or not supported trained model " + this);
        }
    }

    protected String getJSONURL() {
        switch (this) {
            case VGG16:
                return "https://raw.githubusercontent.com/deeplearning4j/dl4j-examples/f9da30063c1636e1de515f2ac514e9a45c1b32cd/dl4j-examples/src/main/resources/trainedModels/VGG16.json";
            case VGG16NOTOP:
                //FIXME
                return "https://raw.githubusercontent.com/deeplearning4j/dl4j-examples/de0087d3b16357d4bc1edbdb6b16f55d2c3da8c9/dl4j-examples/src/main/resources/trainedModels/VGG16NoTop.json";
            default:
                throw new UnsupportedOperationException("Unknown or not supported trained model " + this);
        }
    }

    protected String getH5URL() {
        switch (this) {
            case VGG16:
                return "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels.h5";
            case VGG16NOTOP:
                return "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels_notop.h5";
            default:
                throw new UnsupportedOperationException("Unknown or not supported trained model " + this);
        }
    }

    protected String getH5FileName() {
        switch (this) {
            case VGG16:
                return "vgg16_weights_th_dim_ordering_th_kernels.h5";
            case VGG16NOTOP:
                return "vgg16_weights_th_dim_ordering_th_kernels_notop.h5";
            default:
                throw new UnsupportedOperationException("Unknown or not supported trained model " + this);
        }
    }

    protected String getJSONFileName() {
        switch (this) {
            case VGG16:
                return "VGG16.json";
            case VGG16NOTOP:
                //FIXME
                return "VGG16NoTop.json";
            default:
                throw new UnsupportedOperationException("Unknown or not supported trained model " + this);
        }
    }

    /**
     *
     * @return DataSetPreProcessor required for use with the model
     */
    public DataSetPreProcessor getPreProcessor() {
        switch (this) {
            case VGG16:
            case VGG16NOTOP:
                return new VGG16ImagePreProcessor();
            default:
                throw new UnsupportedOperationException("Unknown or not supported trained model " + this);
        }
    }

    /**
     * Shape of the input to the net, for a minibatch size of 1
     * @return
     */

    public int[] getInputShape() {
        switch (this) {
            case VGG16:
            case VGG16NOTOP:
                return new int[] {1, 3, 224, 224};
            default:
                throw new UnsupportedOperationException("Unknown or not supported trained model " + this);
        }
    }

    /**
     * Shape of the output NDArray from the net, for a minibatch size of 1
     * @return
     */
    public int[] getOuputShape() {
        switch (this) {
            case VGG16:
                return new int[] {1, 1000};
            case VGG16NOTOP:
                return new int[] {1, 512, 7, 7};
            default:
                throw new UnsupportedOperationException("Unknown or not supported trained model " + this);
        }
    }

    /*
        Given predictions from the trained model this method will return a string
        listing the top five matches and the respective probabilities
     */
    public String decodePredictions(INDArray predictions) {
        ArrayList<String> labels;
        String predictionDescription = "";
        int[] top5 = new int[5];
        float[] top5Prob = new float[5];
        switch (this) {
            case VGG16:
                labels = ImageNetLabels.getLabels();
                break;
            case VGG16NOTOP:
            default:
                throw new UnsupportedOperationException("Unknown or not supported on trained model " + this);
        }
        //brute force collect top 5
        int i = 0;
        for (int batch = 0; batch < predictions.size(0); batch++) {
            predictionDescription += "Predictions for batch ";
            if (predictions.size(0) > 1) {
                predictionDescription += String.valueOf(batch);
            }
            predictionDescription += " :";
            INDArray currentBatch = predictions.getRow(batch).dup();
            while (i < 5) {
                top5[i] = Nd4j.argMax(currentBatch, 1).getInt(0, 0);
                top5Prob[i] = currentBatch.getFloat(batch, top5[i]);
                currentBatch.putScalar(0, top5[i], 0);
                predictionDescription += "\n\t" + String.format("%3f", top5Prob[i] * 100) + "%, " + labels.get(top5[i]);
                i++;
            }
        }
        return predictionDescription;
    }
}
