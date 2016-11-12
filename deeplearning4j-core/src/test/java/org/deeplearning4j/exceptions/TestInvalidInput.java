package org.deeplearning4j.exceptions;

import org.deeplearning4j.exception.DL4JException;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.fail;

/**
 * A set of
 */
@Ignore
public class TestInvalidInput {

    @Test
    public void testInputNinMismatchDense(){
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list()
                .layer(0, new DenseLayer.Builder().nIn(10).nOut(10).build())
                .layer(1, new OutputLayer.Builder().nIn(10).nOut(10).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        try{
            net.feedForward(Nd4j.create(1,20));
            fail("Expected DL4JException");
        }catch (DL4JException e){
            System.out.println("testInputNinMismatchDense(): " + e.getMessage());
        }catch (Exception e){
            e.printStackTrace();
            fail("Expected DL4JException");
        }
    }

    @Test
    public void testInputNinMismatchOutputLayer(){
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list()
                .layer(0, new DenseLayer.Builder().nIn(10).nOut(20).build())
                .layer(1, new OutputLayer.Builder().nIn(10).nOut(10).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        try{
            net.feedForward(Nd4j.create(1,10));
            fail("Expected DL4JException");
        }catch (DL4JException e){
            System.out.println("testInputNinMismatchOutputLayer(): " + e.getMessage());
        }catch (Exception e){
            e.printStackTrace();
            fail("Expected DL4JException");
        }
    }

    @Test
    public void testLabelsNOutMismatchOutputLayer(){
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list()
                .layer(0, new DenseLayer.Builder().nIn(10).nOut(10).build())
                .layer(1, new OutputLayer.Builder().nIn(10).nOut(10).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        try{
            net.fit(Nd4j.create(1,10), Nd4j.create(1,20));
            fail("Expected DL4JException");
        }catch (DL4JException e){
            System.out.println("testLabelsNOutMismatchOutputLayer(): " + e.getMessage());
        }catch (Exception e){
            e.printStackTrace();
            fail("Expected DL4JException");
        }
    }

    @Test
    public void testInputNinMismatchConvolutional(){
        //Rank 4 input, but input depth does not match nIn depth

        int h = 16;
        int w = 16;
        int d = 3;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list()
                .layer(0, new ConvolutionLayer.Builder().nIn(d).nOut(5).build())
                .layer(1, new OutputLayer.Builder().nOut(10).build())
                .setInputType(InputType.convolutional(h,w,d))
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        try{
            net.feedForward(Nd4j.create(1,5,h,w));
            fail("Expected DL4JException");
        }catch (DL4JException e){
            System.out.println("testInputNinMismatchConvolutional(): " + e.getMessage());
        }catch (Exception e){
            e.printStackTrace();
            fail("Expected DL4JException");
        }
    }

    @Test
    public void testInputNinRank2Convolutional(){
        //Rank 2 input, instead of rank 4 input. For example, forgetting the

        int h = 16;
        int w = 16;
        int d = 3;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list()
                .layer(0, new ConvolutionLayer.Builder().nIn(d).nOut(5).build())
                .layer(1, new OutputLayer.Builder().nOut(10).build())
                .setInputType(InputType.convolutional(h,w,d))
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        try{
            net.feedForward(Nd4j.create(1,5*h*w));
            fail("Expected DL4JException");
        }catch (DL4JException e){
            System.out.println("testInputNinRank2Convolutional(): " + e.getMessage());
        }catch (Exception e){
            e.printStackTrace();
            fail("Expected DL4JException");
        }
    }

}
