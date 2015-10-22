package org.deeplearning4j.nn.conf.preprocessor;

import static org.junit.Assert.*;

import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.convolution.ConvolutionLayer;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.layers.feedforward.dense.DenseLayer;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

public class TestPreProcessors {
	
	@Test
	public void testRnnToFeedForwardPreProcessor(){
		int[] miniBatchSizes = {5,1,5,1};
		int[] timeSeriesLengths = {9,9,1,1};
		
		for( int x=0; x<miniBatchSizes.length; x++ ){
			int miniBatchSize = miniBatchSizes[x];
			int layerSize = 7;
			int timeSeriesLength = timeSeriesLengths[x];

			RnnToFeedForwardPreProcessor proc = new RnnToFeedForwardPreProcessor();
			NeuralNetConfiguration nnc = new NeuralNetConfiguration.Builder()
			.layer(new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder()
                    .nIn(layerSize).nOut(layerSize).build())
                    .build();

			DenseLayer layer = LayerFactories.getFactory(nnc.getLayer()).create(nnc);
			layer.setInputMiniBatchSize(miniBatchSize);

			INDArray activations3d = Nd4j.rand(new int[]{miniBatchSize,layerSize,timeSeriesLength});
			for( int i=0; i<miniBatchSize; i++ ){
				for( int j=0; j<layerSize; j++ ){
					for( int k=0; k<timeSeriesLength; k++ ){
						double value = 100*i + 10*j + k;	//value abc -> example=a, neuronNumber=b, time=c
						activations3d.putScalar(new int[]{i,j,k},value);
					}
				}
			}

			INDArray activations2d = proc.preProcess(activations3d,layer);
			assertArrayEquals(activations2d.shape(),new int[]{miniBatchSize*timeSeriesLength,layerSize});

			//Expect each row in activations2d to have order:
			//(example=0,t=0), (example=0,t=1), (example=0,t=2), ..., (example=1,t=0), (example=1,t=1), ...
			int nRows = activations2d.rows();
			for( int i=0; i<nRows; i++ ){
				INDArray row = activations2d.getRow(i);
				assertArrayEquals(row.shape(),new int[]{1,layerSize});
				int origExampleNum = i / timeSeriesLength;
				int time = i % timeSeriesLength;
				INDArray expectedRow = activations3d.tensorAlongDimension(time,1,0).getRow(origExampleNum);
				assertTrue(row.equals(expectedRow));
			}

			//Given that epsilons and activations have same shape, we can do this (even though it's not the intended use)
			//Basically backprop should be exact opposite of preProcess
			INDArray out = proc.backprop(activations2d,layer);

			assertTrue(out.equals(activations3d));
		}
	}
	
	@Test
	public void testFeedForwardToRnnPreProcessor(){
		Nd4j.getRandom().setSeed(12345L);

		int[] miniBatchSizes = {5,1,5,1};
		int[] timeSeriesLengths = {9,9,1,1};

		for( int x=0; x<miniBatchSizes.length; x++ ){
			int miniBatchSize = miniBatchSizes[x];
			int layerSize = 7;
			int timeSeriesLength = timeSeriesLengths[x];

			FeedForwardToRnnPreProcessor proc = new FeedForwardToRnnPreProcessor();

			NeuralNetConfiguration nnc = new NeuralNetConfiguration.Builder()
			.layer(new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder()
                    .nIn(layerSize).nOut(layerSize).build())
                    .build();

			DenseLayer layer = LayerFactories.getFactory(nnc.getLayer()).create(nnc);
			layer.setInputMiniBatchSize(miniBatchSize);

			INDArray activations2d = Nd4j.rand(miniBatchSize*timeSeriesLength,layerSize);

			INDArray activations3d = proc.preProcess(activations2d,layer);
			assertArrayEquals(activations3d.shape(),new int[]{miniBatchSize,layerSize,timeSeriesLength});

			int nRows2D = miniBatchSize*timeSeriesLength;
			for( int i=0; i<nRows2D; i++ ){
				int time = i % timeSeriesLength;
				int example = i / timeSeriesLength;

				INDArray row2d = activations2d.getRow(i);
				INDArray row3d = activations3d.tensorAlongDimension(time,1,0).getRow(example);

				assertTrue(row2d.equals(row3d));
			}

			//Again epsilons and activations have same shape, we can do this (even though it's not the intended use)
			INDArray epsilon2d = proc.backprop(activations3d,layer);

			assertTrue(epsilon2d.equals(activations2d));
		}
	}

	@Test
    public void testCnnToRnnPreProcessor(){
        //Two ways to test this:
        // (a) check that doing preProcess + backprop on a given input gives same result
        // (b) compare to ComposableInputPreProcessor(CNNtoFF, FFtoRNN)

        int[] miniBatchSizes = {5,1};
        int[] timeSeriesLengths = {9,1};
        int[] inputHeights = {10,30};
        int[] inputWidths = {10,30};
        int[] numChannels = {1,3,6};
        int cnnNChannelsIn = 3;

        Nd4j.getRandom().setSeed(12345);

        System.out.println();
        for( int miniBatchSize : miniBatchSizes ){
            for(int timeSeriesLength : timeSeriesLengths ){
                for(int inputHeight : inputHeights ){
                    for( int inputWidth : inputWidths ){
                        for(int nChannels : numChannels ){
                            InputPreProcessor proc = new CnnToRnnPreProcessor(inputHeight,inputWidth,nChannels);

                            NeuralNetConfiguration nnc = new NeuralNetConfiguration.Builder()
                                    .layer( new org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder(inputWidth,inputHeight)
                                    .nIn(cnnNChannelsIn).nOut(nChannels).build()).build();

                            ConvolutionLayer layer = LayerFactories.getFactory(nnc.getLayer()).create(nnc);
                            layer.setInputMiniBatchSize(miniBatchSize);

                            INDArray activationsCnn = Nd4j.rand(
                                    new int[]{miniBatchSize * timeSeriesLength, nChannels, inputHeight, inputWidth});

                            //Check shape of outputs:
                            int prod = nChannels*inputHeight*inputWidth;
                            INDArray activationsRnn = proc.preProcess(activationsCnn,layer);
                            assertArrayEquals(new int[]{miniBatchSize,prod,timeSeriesLength},
                                    activationsRnn.shape());

                            //Check backward pass. Given that activations and epsilons have same shape, they should
                            //be opposite operations - i.e., get the same thing back out
                            INDArray twiceProcessed = proc.backprop(activationsRnn,layer);
                            assertArrayEquals(activationsCnn.shape(), twiceProcessed.shape());
                            assertEquals(activationsCnn,twiceProcessed);

                            //Second way to check: compare to ComposableInputPreProcessor(CNNtoFF, FFtoRNN)
                            InputPreProcessor compProc = new ComposableInputPreProcessor(
                                    new CnnToFeedForwardPreProcessor(inputHeight,inputWidth,nChannels),
                                    new FeedForwardToRnnPreProcessor());

                            INDArray activationsRnnComp = compProc.preProcess(activationsCnn,layer);
                            assertEquals(activationsRnnComp,activationsRnn);

                            INDArray epsilonsRnn = Nd4j.rand(
                                    new int[]{miniBatchSize,nChannels*inputHeight*inputWidth,timeSeriesLength});
                            INDArray epsilonsCnnComp = compProc.backprop(epsilonsRnn,layer);
                            INDArray epsilonsCnn = proc.backprop(epsilonsRnn,layer);
							if(!epsilonsCnn.equals(epsilonsCnnComp)){
                                System.out.println(miniBatchSize + "\t" + timeSeriesLength + "\t" + inputHeight + "\t" +
                                    inputWidth + "\t" + nChannels );
                                System.out.println("expected - epsilonsCnnComp");
                                System.out.println(Arrays.toString(epsilonsCnnComp.shape()));
                                System.out.println(epsilonsCnnComp);
                                System.out.println("actual - epsilonsCnn");
                                System.out.println(Arrays.toString(epsilonsCnn.shape()));
                                System.out.println(epsilonsCnn);
                            }
                            assertEquals(epsilonsCnnComp,epsilonsCnn);
                        }
                    }
                }
            }
        }
    }


    @Test
    public void testRnnToCnnPreProcessor(){
        //Two ways to test this:
        // (a) check that doing preProcess + backprop on a given input gives same result
        // (b) compare to ComposableInputPreProcessor(CNNtoFF, FFtoRNN)

        int[] miniBatchSizes = {5,1};
        int[] timeSeriesLengths = {9,1};
        int[] inputHeights = {10,30};
        int[] inputWidths = {10,30};
        int[] numChannels = {1,3,6};
        int cnnNChannelsIn = 3;

        Nd4j.getRandom().setSeed(12345);

        System.out.println();
        for( int miniBatchSize : miniBatchSizes ){
            for(int timeSeriesLength : timeSeriesLengths ){
                for(int inputHeight : inputHeights ){
                    for( int inputWidth : inputWidths ){
                        for(int nChannels : numChannels ){
                            InputPreProcessor proc = new RnnToCnnPreProcessor(inputHeight,inputWidth,nChannels);

                            NeuralNetConfiguration nnc = new NeuralNetConfiguration.Builder()
                                    .layer( new org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder(inputWidth,inputHeight)
                                            .nIn(cnnNChannelsIn).nOut(nChannels).build()).build();

                            ConvolutionLayer layer = LayerFactories.getFactory(nnc.getLayer()).create(nnc);
                            layer.setInputMiniBatchSize(miniBatchSize);

                            INDArray activationsRnn = Nd4j.rand(
                                    new int[]{miniBatchSize , nChannels*inputHeight*inputWidth, timeSeriesLength});

                            //Check shape of outputs:
                            INDArray activationsCnn = proc.preProcess(activationsRnn,layer);
                            assertArrayEquals(new int[]{miniBatchSize*timeSeriesLength,nChannels,inputHeight,inputWidth},
                                    activationsCnn.shape());

                            //Check backward pass. Given that activations and epsilons have same shape, they should
                            //be opposite operations - i.e., get the same thing back out
                            INDArray twiceProcessed = proc.backprop(activationsCnn,layer);
                            assertArrayEquals(activationsRnn.shape(), twiceProcessed.shape());
                            assertEquals(activationsRnn,twiceProcessed);

                            //Second way to check: compare to ComposableInputPreProcessor(RNNtoFF, FFtoCNN)
                            InputPreProcessor compProc = new ComposableInputPreProcessor(
                                    new RnnToFeedForwardPreProcessor(),
                                    new FeedForwardToCnnPreProcessor(inputHeight,inputWidth,nChannels));

                            INDArray activationsCnnComp = compProc.preProcess(activationsRnn,layer);
                            assertEquals(activationsCnnComp,activationsCnn);

                            INDArray epsilonsCnn = Nd4j.rand(
                                    new int[]{miniBatchSize*timeSeriesLength,nChannels,inputHeight,inputWidth});
                            INDArray epsilonsRnnComp = compProc.backprop(epsilonsCnn,layer);
                            INDArray epsilonsRnn = proc.backprop(epsilonsCnn,layer);
                            if(!epsilonsRnn.equals(epsilonsRnnComp)){
                                System.out.println(miniBatchSize + "\t" + timeSeriesLength + "\t" + inputHeight + "\t" +
                                        inputWidth + "\t" + nChannels );
                                System.out.println("expected - epsilonsRnnComp");
                                System.out.println(Arrays.toString(epsilonsRnnComp.shape()));
                                System.out.println(epsilonsRnnComp);
                                System.out.println("actual - epsilonsRnn");
                                System.out.println(Arrays.toString(epsilonsRnn.shape()));
                                System.out.println(epsilonsRnn);
                            }
                            assertEquals(epsilonsRnnComp,epsilonsRnn);
                        }
                    }
                }
            }
        }
    }
}
