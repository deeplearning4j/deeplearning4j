package org.deeplearning4j;

import lombok.val;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class Debug {

    @Test
    public void test() {

        int height = 8;
        int width = 8;
        int depth = 3;

        INDArray img = Nd4j.ones(1, depth, height, width);

        for (int i = 0; i < depth; i++) {
            for (int j = 0; j < height; j++) {
                for (int k = 0; k < width; k++) {
                    img.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.point(i), NDArrayIndex.point(j), NDArrayIndex.point(k)}, j + k);
                }
            }
        }

        val builder_pooling = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .name("max_pooling")
                .kernelSize(2, 2)
                .stride(1, 1)
                .build();



        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.ONES)
                .list()
                .layer(0, new ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).padding(0,0).nIn(depth).nOut(1).biasInit(0.0).activation(Activation.RELU).build())
                .layer(1, builder_pooling)
                .setInputType(InputType.convolutional(height, width, depth)).build();


        val net = new MultiLayerNetwork(conf);
        net.init();

        val output = net.output(img);

        System.out.println(output);
    }

    @Test
    public void test2(){

        INDArray x = Nd4j.create(new double[]{2,2,2});
        INDArray y = Nd4j.create(new double[]{4,6,8});
        INDArray result = Nd4j.createUninitialized(1,3);

        CustomOp op = DynamicCustomOp.builder("Div")
                .addInputs(x,y)
                .addOutputs(result)
                .callInplace(false)
                .build();

        Nd4j.getExecutioner().exec(op);

        System.out.println(result);
    }

}
