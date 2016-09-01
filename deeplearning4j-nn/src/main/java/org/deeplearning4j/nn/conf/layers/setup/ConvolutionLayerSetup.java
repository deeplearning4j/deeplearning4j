package org.deeplearning4j.nn.conf.layers.setup;


import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;

/**
 * @deprecated Use {@link org.deeplearning4j.nn.conf.MultiLayerConfiguration.Builder#setInputType(InputType)} to set nIns
 * and add preprocessors as required. This can be done using {@code builder.setInputType(InputType.convolutional(height, width, channels))}
 */
@Deprecated
public class ConvolutionLayerSetup {

    /**
     * Take in the configuration
     * @param builder the configuration builder
     * @param height initial height of the data
     * @param width initial width of the data
     * @param channels initial number of channels in the data
     * @deprecated Use {@link org.deeplearning4j.nn.conf.MultiLayerConfiguration.Builder#setInputType(InputType)} to set nIns
     * and add preprocessors as required. This can be done using {@code builder.setInputType(InputType.convolutional(height, width, channels))}
     */
    @Deprecated
    public ConvolutionLayerSetup(MultiLayerConfiguration.Builder builder,int height,int width,int channels) {
        builder.setInputType(InputType.convolutional(height, width, channels));
    }
}