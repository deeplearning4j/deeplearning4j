package org.deeplearning4j.zoo.model.helper;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.SeparableConvolution2D;
import org.deeplearning4j.zoo.model.NASNet;
import org.nd4j.linalg.activations.Activation;

/**
 * Layer helpers {@link NASNet}.
 *
 * @author Jsutin Long (crockpotveggies)
 */
public class NASNetHelper {


    private String sepConvBlock(ComputationGraphConfiguration.GraphBuilder graphBuilder, int filters, int kernelSize, int stride, String blockId, String input) {
        String prefix = "sepConvBlock"+blockId;

        graphBuilder
                .addLayer(prefix+"_act", new ActivationLayer(Activation.RELU), input)
                .addLayer(prefix+"_sepconv1", new SeparableConvolution2D.Builder(kernelSize, kernelSize).stride(stride, stride).nOut(filters).hasBias(false)
                        .convolutionMode(ConvolutionMode.Same).build(), prefix+"_act")
                .addLayer(prefix+"_conv1_bn", new BatchNormalization.Builder().eps(1e-3).build(), prefix+"_sepconv1")
                .addLayer(prefix+"_act2", new ActivationLayer(Activation.RELU), prefix+"_conv1_bn")
                .addLayer(prefix+"_sepconv2", new SeparableConvolution2D.Builder(kernelSize, kernelSize).stride(stride, stride).nOut(filters).hasBias(false)
                        .convolutionMode(ConvolutionMode.Same).build(), prefix+"_act2")
                .addLayer(prefix+"_conv2_bn", new BatchNormalization.Builder().eps(1e-3).build(), prefix+"_sepconv2");

        return prefix;
    }

    private String adjustBlock(ComputationGraphConfiguration.GraphBuilder graphBuilder, int filters, String blockId, String input) {

    }

    private String reductionCell(ComputationGraphConfiguration.GraphBuilder graphBuilder, int fireId, int squeeze, int expand, String input) {

    }

}
