package org.deeplearning4j.zoo.model.helper;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.convolutional.Cropping2D;
import org.deeplearning4j.zoo.model.NASNet;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.primitives.Pair;

import java.util.Map;

/**
 * Layer helpers {@link NASNet}.
 *
 * @author Justin Long (crockpotveggies)
 */
public class NASNetHelper {


    public static String sepConvBlock(ComputationGraphConfiguration.GraphBuilder graphBuilder, int filters, int kernelSize, int stride, String blockId, String input) {
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

        return prefix+"_conv2_bn";
    }

    public static String adjustBlock(ComputationGraphConfiguration.GraphBuilder graphBuilder, int filters, String blockId, String input) {
        return adjustBlock(graphBuilder, filters, blockId, input, null);
    }

    public static String adjustBlock(ComputationGraphConfiguration.GraphBuilder graphBuilder, int filters, String blockId, String input, String inputToMatch) {
        String prefix = "adjustBlock"+blockId;
        String outputName = input;

        if(inputToMatch == null) {
            inputToMatch = input;
        }
        Map<String, InputType> layerActivationTypes = graphBuilder.getLayerActivationTypes();
        int[] shapeToMatch = layerActivationTypes.get(inputToMatch).getShape();
        int[] inputShape = layerActivationTypes.get(input).getShape();

        if(shapeToMatch[1] != inputShape[1]) {
            graphBuilder
                    .addLayer(prefix+"_relu1", new ActivationLayer(Activation.RELU), input)
                    // tower 1
                    .addLayer(prefix+"_avgpool1", new SubsamplingLayer.Builder(PoolingType.AVG).kernelSize(2,2).stride(2,2)
                            .build(), prefix+"_relu1")
                    .addLayer(prefix+"_conv1", new ConvolutionLayer.Builder(1,1).nOut((int) Math.floor(filters / 2))
                            .convolutionMode(ConvolutionMode.Same).build(), prefix+"_avg_pool_1")
                    // tower 2
                    .addLayer(prefix+"_zeropad1", new ZeroPaddingLayer(0,1), prefix+"_relu1")
                    .addLayer(prefix+"_crop1", new Cropping2D(1,0), prefix+"_zeropad_1")
                    .addLayer(prefix+"_avgpool2", new SubsamplingLayer.Builder(PoolingType.AVG).kernelSize(1,1).stride(2,2)
                            .build(), prefix+"_crop1")
                    .addLayer(prefix+"_conv2", new ConvolutionLayer.Builder(1,1).nOut((int) Math.floor(filters / 2))
                            .convolutionMode(ConvolutionMode.Same).build(), prefix+"_avgpool2")

                    .addVertex(prefix+"_concat1", new MergeVertex(), prefix+"_conv1", prefix+"_conv2")
                    .addLayer(prefix+"_bn1", new BatchNormalization.Builder().eps(1e-3).gamma(0.9997)
                            .build(), prefix+"_concat1");

            outputName = prefix+"_bn1";
        }

        if(inputShape[3] != filters) {
            graphBuilder
                    .addLayer(prefix+"_projection_relu", new ActivationLayer(Activation.RELU), outputName)
                    .addLayer(prefix+"_projection_conv", new ConvolutionLayer.Builder(1,1).stride(1,1).nOut(filters)
                            .convolutionMode(ConvolutionMode.Same).build(), prefix+"_projection_relu")
                    .addLayer(prefix+"_projection_bn", new BatchNormalization.Builder().eps(1e-3).gamma(0.9997)
                            .build(), prefix+"_projection_conv");
            outputName = prefix+"_projection_bn";
        }

        return outputName;
    }

    public static Pair<String, String> normalA(ComputationGraphConfiguration.GraphBuilder graphBuilder, int filters, String blockId, String inputX, String inputP) {
        String prefix = "normalA"+blockId;

        String topAdjust = adjustBlock(graphBuilder, filters, prefix, inputP, inputX);

        // top block
        graphBuilder
                .addLayer(prefix+"_relu1", new ActivationLayer(Activation.RELU), topAdjust)
                .addLayer(prefix+"_conv1", new ConvolutionLayer.Builder(1,1).stride(1,1).nOut(filters).hasBias(false)
                        .convolutionMode(ConvolutionMode.Same).build(), prefix+"_relu1")
                .addLayer(prefix+"_bn1", new BatchNormalization.Builder().eps(1e-3).gamma(0.9997)
                        .build(), prefix+"_conv1");

        // block 1
        String left1 = sepConvBlock(graphBuilder, filters, 5, 1, prefix+"_left1", prefix+"_bn1");
        String right1 = sepConvBlock(graphBuilder, filters, 3, 1, prefix+"_right1", topAdjust);
        graphBuilder.addVertex(prefix+"_add1", new ElementWiseVertex(ElementWiseVertex.Op.Add), left1, right1);

        // block 2
        String left2 = sepConvBlock(graphBuilder, filters, 5, 1, prefix+"_left2", topAdjust);
        String right2 = sepConvBlock(graphBuilder, filters, 3, 1, prefix+"_right2", topAdjust);
        graphBuilder.addVertex(prefix+"_add2", new ElementWiseVertex(ElementWiseVertex.Op.Add), left2, right2);

        // block 3
        graphBuilder
                .addLayer(prefix+"_left3", new SubsamplingLayer.Builder(PoolingType.AVG).kernelSize(3,3).stride(1,1)
                        .convolutionMode(ConvolutionMode.Same).build(), prefix+"_bn1")
                .addVertex(prefix+"_add3", new ElementWiseVertex(ElementWiseVertex.Op.Add), prefix+"_left3", topAdjust);

        // block 4
        graphBuilder
                .addLayer(prefix+"_left4", new SubsamplingLayer.Builder(PoolingType.AVG).kernelSize(3,3).stride(1,1)
                        .convolutionMode(ConvolutionMode.Same).build(), topAdjust)
                .addLayer(prefix+"_right4", new SubsamplingLayer.Builder(PoolingType.AVG).kernelSize(3,3).stride(1,1)
                        .convolutionMode(ConvolutionMode.Same).build(), topAdjust)
                .addVertex(prefix+"_add4", new ElementWiseVertex(ElementWiseVertex.Op.Add), prefix+"_left4", prefix+"_right4");

        // block 5
        String left5 = sepConvBlock(graphBuilder, filters, 3, 1, prefix+"_left5", topAdjust);
        graphBuilder.addVertex(prefix+"_add5", new ElementWiseVertex(ElementWiseVertex.Op.Add), prefix+"_left5", prefix+"_bn1");

        // output
        graphBuilder.addVertex(prefix, new MergeVertex(),
                topAdjust, prefix+"_add1", prefix+"_add2", prefix+"_add3", prefix+"_add4", prefix+"_add5");

        return new Pair<>(prefix, inputX);

    }

    public static Pair<String, String> reductionA(ComputationGraphConfiguration.GraphBuilder graphBuilder, int filters, String blockId, String inputX, String inputP) {
        String prefix = "reductionA"+blockId;

        String topAdjust = adjustBlock(graphBuilder, filters, prefix, inputP, inputX);

        // top block
        graphBuilder
                .addLayer(prefix+"_relu1", new ActivationLayer(Activation.RELU), topAdjust)
                .addLayer(prefix+"_conv1", new ConvolutionLayer.Builder(1,1).stride(1,1).nOut(filters).hasBias(false)
                        .convolutionMode(ConvolutionMode.Same).build(), prefix+"_relu1")
                .addLayer(prefix+"_bn1", new BatchNormalization.Builder().eps(1e-3).gamma(0.9997)
                        .build(), prefix+"_conv1");

        // block 1
        String left1 = sepConvBlock(graphBuilder, filters, 5, 2, prefix+"_left1", prefix+"_bn1");
        String right1 = sepConvBlock(graphBuilder, filters, 7, 2, prefix+"_right1", topAdjust);
        graphBuilder.addVertex(prefix+"_add1", new ElementWiseVertex(ElementWiseVertex.Op.Add), left1, right1);

        // block 2
        graphBuilder.addLayer(prefix+"_left2", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(3,3).stride(2,2)
                .convolutionMode(ConvolutionMode.Same).build(), prefix+"_bn1");
        String right2 = sepConvBlock(graphBuilder, filters, 3, 1, prefix+"_right2", topAdjust);
        graphBuilder.addVertex(prefix+"_add2", new ElementWiseVertex(ElementWiseVertex.Op.Add), prefix+"_left2", right2);

        // block 3
        graphBuilder.addLayer(prefix+"_left3", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG).kernelSize(3,3).stride(2,2)
                .convolutionMode(ConvolutionMode.Same).build(), prefix+"_bn1");
        String right3 = sepConvBlock(graphBuilder, filters, 5, 2, prefix+"_right3", topAdjust);
        graphBuilder.addVertex(prefix+"_add3", new ElementWiseVertex(ElementWiseVertex.Op.Add), prefix+"_left3", right3);

        // block 4
        graphBuilder
                .addLayer(prefix+"_left4", new SubsamplingLayer.Builder(PoolingType.AVG).kernelSize(3,3).stride(1,1)
                        .convolutionMode(ConvolutionMode.Same).build(), prefix+"_add1")
                .addVertex(prefix+"_add4", new ElementWiseVertex(ElementWiseVertex.Op.Add), prefix+"_add2", prefix+"_left4");

        // block 5
        String left5 = sepConvBlock(graphBuilder, filters, 3, 2, prefix+"_left5", prefix+"_add1");
        graphBuilder
                .addLayer(prefix+"_right5", new SubsamplingLayer.Builder(PoolingType.MAX).kernelSize(3,3).stride(2,2)
                        .convolutionMode(ConvolutionMode.Same).build(), prefix+"_bn1")
                .addVertex(prefix+"_add5", new ElementWiseVertex(ElementWiseVertex.Op.Add), left5, prefix+"_right5");

        // output
        graphBuilder.addVertex(prefix, new MergeVertex(),
                prefix+"_add2", prefix+"_add3", prefix+"_add4", prefix+"_add5");

        return new Pair<>(prefix, inputX);


    }

}
