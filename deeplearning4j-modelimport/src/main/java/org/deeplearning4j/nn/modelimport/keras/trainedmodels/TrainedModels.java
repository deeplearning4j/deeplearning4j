package org.deeplearning4j.nn.modelimport.keras.trainedmodels;


import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;

/**
 * Support for popular trained image-classification models
 * @author susaneraly
 */
public enum TrainedModels {

    VGG16,
    VGG16NOTOP;

    /**
     * Name of the sub dir in the local cache associated with the model.
     * Local cache dir ~/.dl4j/trainedmodels
     * @return
     */
    public String getModelDir() {
        switch (this) {
            case VGG16:
                return "vgg16";
            case VGG16NOTOP:
                return "vgg16notop";
            default:
                throw new UnsupportedOperationException("Unknown or not supported trained model " + this);
        }
    }

    public String getJSONURL() {
        switch (this) {
            case VGG16:
                return "https://raw.githubusercontent.com/deeplearning4j/dl4j-examples/f9da30063c1636e1de515f2ac514e9a45c1b32cd/dl4j-examples/src/main/resources/trainedModels/VGG16.json";
            case VGG16NOTOP:
                //FIXME
                return "https://raw.githubusercontent.com/deeplearning4j/dl4j-examples/f9da30063c1636e1de515f2ac514e9a45c1b32cd/dl4j-examples/src/main/resources/trainedModels/VGG16.json";
            default:
                throw new UnsupportedOperationException("Unknown or not supported trained model " + this);
        }
    }

    public String getH5URL() {
        switch (this) {
            case VGG16:
                return "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels.h5";
            case VGG16NOTOP:
                return "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels_notop.h5";
            default:
                throw new UnsupportedOperationException("Unknown or not supported trained model " + this);
        }
    }

    public String getH5FileName() {
        switch (this) {
            case VGG16:
                return "vgg16_weights_th_dim_ordering_th_kernels.h5";
            case VGG16NOTOP:
                return "vgg16_weights_th_dim_ordering_th_kernels_notop.h5";
            default:
                throw new UnsupportedOperationException("Unknown or not supported trained model " + this);
        }
    }

    public String getJSONFileName() {
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
                return new VGG16ImagePreProcessor();
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
        switch(this) {
            case VGG16:
                return new int[]{1,3,224,224};
            case VGG16NOTOP:
                return new int[] {1,3,224,224};
            default:
                throw new UnsupportedOperationException("Unknown or not supported trained model " + this);
        }
    }

    /**
     * Shape of the output NDArray from the net, for a minibatch size of 1
     * @return
     */
    public int[] getOuputShape() {
        switch(this) {
            case VGG16:
                return new int[]{1,1000};
            case VGG16NOTOP:
                return new int[] {1,512,7,7};
            default:
                throw new UnsupportedOperationException("Unknown or not supported trained model " + this);
        }
    }
}
