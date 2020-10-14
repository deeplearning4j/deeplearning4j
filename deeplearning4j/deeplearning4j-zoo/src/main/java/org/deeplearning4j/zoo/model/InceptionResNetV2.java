/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.zoo.model;


import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.ModelMetaData;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.ZooType;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.lossfunctions.LossFunctions;




/**
 *
 * This is dl4j implementation of Inception ResNet V2 base on this paper
 * https://arxiv.org/pdf/1602.07261v1.pdf
 * and reference this repository  https://github.com/titu1994/Inception-v4
 *
 * Convolution2D subsample in keras=dl4j strides
 * Convolution2D border_mode in keras=dl4j convolutionMode
 * @author wenfengxu  wechatid:italybaby
 */
@AllArgsConstructor
@Builder
@Slf4j
public class InceptionResNetV2 extends ZooModel {

    @Builder.Default private final  long seed = 1234;
    @Builder.Default private  int[] inputShape = new int[] {3, 299, 299};
    @Builder.Default private  int numClasses = 0;
    @Builder.Default private IUpdater updater =new Adam(0.0001);
    @Builder.Default private CacheMode cacheMode = CacheMode.NONE;
    @Builder.Default private WorkspaceMode workspaceMode = WorkspaceMode.ENABLED;
    @Builder.Default private ConvolutionLayer.AlgoMode cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST;



	private static String ACTIVATION_LAYER ="activation-layer";

	private static String MERGE_VERTEX ="merge-vertex";

	private static String MAX_POOLING ="max-pooling";

	private static String AVG_POOLING ="avg-pooling";

	private static String CNN ="cnn";

    private InceptionResNetV2() {

	}

    @Override
    public String pretrainedUrl(PretrainedType pretrainedType) {
        return null;
    }

    @Override
    public long pretrainedChecksum(PretrainedType pretrainedType) {
        return 0L;
    }

    @Override
    public Class<? extends Model> modelType() {
        return ComputationGraph.class;
    }



    @SuppressWarnings("unchecked")
	@Override
    public ComputationGraph init() {
      
    	ComputationGraphConfiguration.GraphBuilder graphBuilder = graphBuilder("input");


        graphBuilder.addInputs("input").setInputTypes(InputType.convolutional(inputShape[2], inputShape[1], inputShape[0]))
        
				        .addLayer("outputLayer",new OutputLayer.Builder().nOut(numClasses)
										.lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
										.activation(Activation.SOFTMAX).build()

                                                      ,
								"drop_out_layer")

                        .setOutputs("outputLayer");

        ComputationGraphConfiguration conf = graphBuilder.build();



        ComputationGraph model = new ComputationGraph(conf);

        model.init();

        return model;
    }


	public ComputationGraphConfiguration.GraphBuilder graphBuilder(String input) {

		ComputationGraphConfiguration.GraphBuilder graphBuilder
				= new NeuralNetConfiguration.Builder().seed(seed)
				.activation(Activation.RELU)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.updater(updater)
				.weightInit(WeightInit.XAVIER)

				.miniBatch(true)
				.cacheMode(cacheMode)
				.trainingWorkspaceMode(workspaceMode)
				.inferenceWorkspaceMode(workspaceMode)
				.graphBuilder();


		graphBuilder=buildInceptionStem(graphBuilder, input);


		input=createLayerName("stem", ACTIVATION_LAYER,0,16);

        int inceptionABatchSize=5;

        graphBuilder=buildBatchInceptionA(graphBuilder,input,inceptionABatchSize);


		input=createLayerName("inception-A", ACTIVATION_LAYER,(inceptionABatchSize-1),9);

		graphBuilder=buildReductionA(graphBuilder,input);


		input=createLayerName("reduction-A", ACTIVATION_LAYER,0,6);


		int inceptionBBatchSize=10;

		graphBuilder=buildBatchInceptionB(graphBuilder,input,inceptionBBatchSize);


		input=createLayerName("inception-B", ACTIVATION_LAYER,(inceptionBBatchSize-1),7);

		graphBuilder=buildReductionB(graphBuilder,input);

		input=createLayerName("reduction-B", ACTIVATION_LAYER,0,9);


		int inceptionCBatchSize=5;

		graphBuilder=buildBatchInceptionC(graphBuilder,input,inceptionCBatchSize);


		AveragePooling2D(graphBuilder,"average-pooling",0,0,createLayerName("inception-C", ACTIVATION_LAYER,(inceptionCBatchSize-1),7),new int[] {8,8},new int[] {1,1},ConvolutionMode.Same);


		graphBuilder.addLayer("drop_out_layer",new DropoutLayer.Builder(0.8).build(),createLayerName("average-pooling", AVG_POOLING,0,0));


		return graphBuilder;
	}



	private ComputationGraphConfiguration.GraphBuilder buildBatchInceptionA(ComputationGraphConfiguration.GraphBuilder graph,String input,int batchSize) {

		for(int i=0;i<batchSize;i++) {

			if(i>0) {
				input=createLayerName("inception-A", ACTIVATION_LAYER,i-1,9);
			}

			graph=buildInceptionA(graph, input, i);

		}
		return graph;
	}

	private ComputationGraphConfiguration.GraphBuilder buildBatchInceptionB(ComputationGraphConfiguration.GraphBuilder graph,String input,  int batchSize) {

		for(int i=0;i<batchSize;i++) {

			if(i>0) {
				input=createLayerName("inception-B", ACTIVATION_LAYER,i-1,7);
			}

			graph=buildInceptionB(graph, input, i);

		}
		return graph;
	}

	private ComputationGraphConfiguration.GraphBuilder buildBatchInceptionC(ComputationGraphConfiguration.GraphBuilder graph,String input, int batchSize) {

		for(int i=0;i<batchSize;i++) {

			if(i>0) {
				input=createLayerName("inception-C",ACTIVATION_LAYER,i-1,7);
			}

			graph=buildInceptionC(graph, input, i);

		}
		return graph;
	}

	/**
	 * build stem for model
	 *
	 * @param graph
	 * @param input
	 * @return
	 */
	private ComputationGraphConfiguration.GraphBuilder buildInceptionStem(ComputationGraphConfiguration.GraphBuilder graph,String input) {

		String moduleName="stem";

		int moduleIndex=0;
		//c
		convBlock(graph, moduleName, moduleIndex,0, input, new int[] {3,3},new int[] {2,2}, inputShape[0], 32, ConvolutionMode.Truncate);
		//c
		convBlock(graph, moduleName, moduleIndex,1, createLayerName(moduleName, CNN,moduleIndex,0), new int[] {3,3}, 32, ConvolutionMode.Truncate);
        //c
		convBlock(graph, moduleName, moduleIndex,2, createLayerName(moduleName, CNN,moduleIndex,1), new int[] {3,3}, 64, ConvolutionMode.Same);

        //c1
		MaxPooling2D(graph,moduleName,moduleIndex,3, createLayerName(moduleName, CNN,moduleIndex,2), new int[] {3, 3}, new int[] {2, 2}, ConvolutionMode.Truncate);
        //c2
		convBlock(graph, moduleName, moduleIndex,4, createLayerName(moduleName, CNN,moduleIndex,2), new int[] {3,3},new int[] {2,2}, 96, ConvolutionMode.Truncate);

		//m
		graph.addVertex(createLayerName(moduleName, MERGE_VERTEX,moduleIndex,5), new MergeVertex(), new String[]{createLayerName(moduleName, MAX_POOLING,moduleIndex,3),createLayerName(moduleName, CNN,moduleIndex,4)});

		//c1
		convBlock(graph, moduleName, moduleIndex,6, createLayerName(moduleName, MERGE_VERTEX,moduleIndex,5), new int[] {1,1}, 64, ConvolutionMode.Same);
		//c1
		convBlock(graph, moduleName, moduleIndex,7, createLayerName(moduleName, CNN,moduleIndex,6), new int[] {3,3}, 96, ConvolutionMode.Truncate);

		//c2
		convBlock(graph, moduleName, moduleIndex,8, createLayerName(moduleName, MERGE_VERTEX,moduleIndex,5), new int[] {1,1}, 64, ConvolutionMode.Same);
		//c2
		convBlock(graph, moduleName, moduleIndex,9, createLayerName(moduleName, CNN,moduleIndex,8), new int[] {7,1}, 64, ConvolutionMode.Same);
		//c2
		convBlock(graph, moduleName, moduleIndex,10, createLayerName(moduleName, CNN,moduleIndex,9), new int[] {1,7}, 64, ConvolutionMode.Same);
		//c2
		convBlock(graph, moduleName, moduleIndex,11, createLayerName(moduleName, CNN,moduleIndex,10), new int[] {3,3}, 96, ConvolutionMode.Truncate);

		//m2
		graph.addVertex(createLayerName(moduleName, MERGE_VERTEX,moduleIndex,12), new MergeVertex(), new String[]{createLayerName(moduleName, CNN,moduleIndex,7),createLayerName(moduleName, CNN,moduleIndex,11)});

        //p1
		MaxPooling2D(graph,moduleName,moduleIndex,13, createLayerName(moduleName, MERGE_VERTEX,moduleIndex,12), new int[] {3, 3}, new int[] {2, 2}, ConvolutionMode.Truncate);
		//p2
		convBlock(graph, moduleName, moduleIndex,14, createLayerName(moduleName, MERGE_VERTEX,moduleIndex,12), new int[] {3,3},new int[] {2,2}, 192, ConvolutionMode.Truncate);

		//m3
		graph.addVertex(createLayerName(moduleName, MERGE_VERTEX,moduleIndex,15), new MergeVertex(), new String[]{createLayerName(moduleName, MAX_POOLING,moduleIndex,13),createLayerName(moduleName, CNN,moduleIndex,14)});

		batchNormAndActivation(graph,createLayerName(moduleName, MERGE_VERTEX,moduleIndex,15), moduleName,moduleIndex,16);

		return graph;
	}


	/**
	 * build InceptionA
	 * @param graph
	 * @param input
	 * @param moduleIndex
	 * @return
	 */
	private ComputationGraphConfiguration.GraphBuilder buildInceptionA(ComputationGraphConfiguration.GraphBuilder graph,String input,Integer moduleIndex) {

		String moduleName="inception-A";
		//ir1
        convBlock(graph, moduleName, moduleIndex,0, input, new int[] {1,1}, 32, ConvolutionMode.Same);

        //ir2
        convBlock(graph, moduleName, moduleIndex,1, input, new int[] {1,1}, 32, ConvolutionMode.Same);
        //ir2
        convBlock(graph, moduleName, moduleIndex,2, createLayerName(moduleName, CNN,moduleIndex,1), new int[] {3,3}, 32, ConvolutionMode.Same);

        //ir3
        convBlock(graph, moduleName, moduleIndex,3, input, new int[] {1,1}, 32, ConvolutionMode.Same);
        //ir3
        convBlock(graph, moduleName, moduleIndex,4, createLayerName(moduleName, CNN,moduleIndex,3), new int[] {3,3}, 48, ConvolutionMode.Same);
        //ir3
        convBlock(graph, moduleName, moduleIndex,5, createLayerName(moduleName, CNN,moduleIndex,4), new int[] {3,3}, 64, ConvolutionMode.Same);

        //ir_merge
        graph.addVertex(createLayerName(moduleName, MERGE_VERTEX,moduleIndex,6), new MergeVertex(), new String[]{createLayerName(moduleName, CNN,moduleIndex,0),createLayerName(moduleName, CNN,moduleIndex,2),createLayerName(moduleName, CNN,moduleIndex,5)});


        convBlock(graph,moduleName,moduleIndex,7,createLayerName(moduleName, MERGE_VERTEX,moduleIndex,6),new int[]{1,1},new int[]{1,1},0, 384,ConvolutionMode.Same,Activation.IDENTITY);

        graph.addVertex(createLayerName(moduleName, MERGE_VERTEX,moduleIndex,8), new MergeVertex(), new String[]{input,createLayerName(moduleName, CNN,moduleIndex,7)});


        batchNormAndActivation(graph,createLayerName(moduleName, MERGE_VERTEX,moduleIndex,8), moduleName,moduleIndex,9);


        return graph;

	}


	private ComputationGraphConfiguration.GraphBuilder buildReductionA(ComputationGraphConfiguration.GraphBuilder graph,String input) {

		String moduleName="reduction-A";

		int moduleIndex=0;

		//r1
		MaxPooling2D(graph,moduleName,moduleIndex,0,input,new int[] {3,3},new int[] {2,2},ConvolutionMode.Truncate);
        //r2
		convBlock(graph, moduleName, moduleIndex,1, input, new int[] {3,3}, new int[] {2,2},384, ConvolutionMode.Truncate);

		//r3
		convBlock(graph, moduleName, moduleIndex,2, input, new int[] {1,1}, 256, ConvolutionMode.Same);
		//r3
		convBlock(graph, moduleName, moduleIndex,3, createLayerName(moduleName, CNN,moduleIndex,2), new int[] {3,3}, 256, ConvolutionMode.Same);
		//r3
		convBlock(graph, moduleName, moduleIndex,4, createLayerName(moduleName, CNN,moduleIndex,3), new int[] {3,3}, new int[] {2,2},384, ConvolutionMode.Truncate);


		graph.addVertex(createLayerName(moduleName, MERGE_VERTEX,moduleIndex,5), new MergeVertex(), new String[]{createLayerName(moduleName, MAX_POOLING,moduleIndex,0),createLayerName(moduleName, CNN,moduleIndex,1),createLayerName(moduleName, CNN,moduleIndex,4)});

		batchNormAndActivation(graph,createLayerName(moduleName, MERGE_VERTEX,moduleIndex,5), moduleName,moduleIndex,6);

		return graph;
	}


	private ComputationGraphConfiguration.GraphBuilder buildInceptionB(ComputationGraphConfiguration.GraphBuilder graph,String input,Integer moduleIndex) {

		String moduleName="inception-B";

		//ir1
		convBlock(graph, moduleName, moduleIndex,0, input, new int[] {1,1}, 192, ConvolutionMode.Same);

		//ir2
		convBlock(graph, moduleName, moduleIndex,1, input, new int[] {1,1}, 128, ConvolutionMode.Same);
		//ir2
		convBlock(graph, moduleName, moduleIndex,2, createLayerName(moduleName, CNN,moduleIndex,1), new int[] {1,7}, 160, ConvolutionMode.Same);
        //ir2
		convBlock(graph, moduleName, moduleIndex,3, createLayerName(moduleName, CNN,moduleIndex,2), new int[] {7,1}, 192, ConvolutionMode.Same);


		graph.addVertex(createLayerName(moduleName, MERGE_VERTEX,moduleIndex,4), new MergeVertex(), new String[]{createLayerName(moduleName, CNN,moduleIndex,0),createLayerName(moduleName, CNN,moduleIndex,3)});


		//ir_conv
		convBlock(graph,moduleName,moduleIndex,5,createLayerName(moduleName, MERGE_VERTEX,moduleIndex,4),new int[]{1,1},new int[]{1,1},0, 1152,ConvolutionMode.Same,Activation.IDENTITY);



		graph.addVertex(createLayerName(moduleName, MERGE_VERTEX,moduleIndex,6), new MergeVertex(), new String[]{input,createLayerName(moduleName, CNN,moduleIndex,5)});


		batchNormAndActivation(graph,createLayerName(moduleName, MERGE_VERTEX,moduleIndex,6), moduleName,moduleIndex,7);


		return graph;



	}


	private ComputationGraphConfiguration.GraphBuilder buildReductionB(ComputationGraphConfiguration.GraphBuilder graph,String input) {

		String moduleName="reduction-B";

		int moduleIndex=0;

		//r1
		MaxPooling2D(graph,moduleName,moduleIndex,0,input,new int[] {3,3},new int[] {2,2},ConvolutionMode.Truncate);

		//r2
		convBlock(graph, moduleName, moduleIndex,1, input, new int[] {1,1}, 256, ConvolutionMode.Same);
		//r2
		convBlock(graph, moduleName, moduleIndex,2, createLayerName(moduleName, CNN,moduleIndex,1), new int[] {3,3}, new int[] {2,2},384, ConvolutionMode.Truncate);

		//r3
		convBlock(graph, moduleName, moduleIndex,3, input, new int[] {1,1}, 256, ConvolutionMode.Same);
		//r3
		convBlock(graph, moduleName, moduleIndex,4, createLayerName(moduleName, CNN,moduleIndex,3), new int[] {3,3},new int[] {2,2}, 288, ConvolutionMode.Truncate);


		//r4
		convBlock(graph, moduleName, moduleIndex,5, input, new int[] {1,1}, 256, ConvolutionMode.Same);
		//r4
		convBlock(graph, moduleName, moduleIndex,6, createLayerName(moduleName, CNN,moduleIndex,5), new int[] {3,3}, 288, ConvolutionMode.Same);
		//r4
		convBlock(graph, moduleName, moduleIndex,7, createLayerName(moduleName, CNN,moduleIndex,6), new int[] {3,3},new int[] {2,2}, 320, ConvolutionMode.Truncate);


		graph.addVertex(createLayerName(moduleName, MERGE_VERTEX,moduleIndex,8), new MergeVertex(), new String[]{
				createLayerName(moduleName, MAX_POOLING,moduleIndex,0),
				createLayerName(moduleName, CNN,moduleIndex,2),
				createLayerName(moduleName, CNN,moduleIndex,4),
				createLayerName(moduleName, CNN,moduleIndex,7)});


		batchNormAndActivation(graph,createLayerName(moduleName, MERGE_VERTEX,moduleIndex,8), moduleName,moduleIndex,9);

		return graph;
	}


	private ComputationGraphConfiguration.GraphBuilder buildInceptionC(ComputationGraphConfiguration.GraphBuilder graph,String input,Integer moduleIndex) {

		String moduleName="inception-C";

		//ir1
		convBlock(graph, moduleName, moduleIndex,0, input, new int[] {1,1}, 192, ConvolutionMode.Same);

		//ir2
		convBlock(graph, moduleName, moduleIndex,1, input, new int[] {1,1}, 192, ConvolutionMode.Same);
		//ir2
		convBlock(graph, moduleName, moduleIndex,2, createLayerName(moduleName, CNN,moduleIndex,1), new int[] {1,3}, 224, ConvolutionMode.Same);
		//ir2
		convBlock(graph, moduleName, moduleIndex,3, createLayerName(moduleName, CNN,moduleIndex,2), new int[] {3,1}, 256, ConvolutionMode.Same);


		graph.addVertex(createLayerName(moduleName, MERGE_VERTEX,moduleIndex,4), new MergeVertex(), new String[]{createLayerName(moduleName, CNN,moduleIndex,0),createLayerName(moduleName, CNN,moduleIndex,3)});

		//ir_conv
		convBlock(graph,moduleName,moduleIndex,5,createLayerName(moduleName, MERGE_VERTEX,moduleIndex,4),new int[]{1,1},new int[]{1,1},0, 2144,ConvolutionMode.Same,Activation.IDENTITY);



		graph.addVertex(createLayerName(moduleName, MERGE_VERTEX,moduleIndex,6), new MergeVertex(), new String[]{input,createLayerName(moduleName, CNN,moduleIndex,5)});


		batchNormAndActivation(graph,createLayerName(moduleName, MERGE_VERTEX,moduleIndex,6), moduleName,moduleIndex,7);


		return graph;

	}

	/**
	 * One model has one or more
	 * module,one module has one
	 * or more block,so the name of
	 * layer is constructed with
	 * moduleName+"-"+layerName+"-"+moduleIndex+"-"+blockIndex
	 * @param moduleName
	 * @param layerName
	 * @param moduleIndex
	 * @param blockIndex
	 * @return
	 */
	private String createLayerName(String moduleName, String layerName,Integer moduleIndex,Integer blockIndex) {
		String newLayerName=moduleName.concat("-").concat(layerName).concat("-").concat(String.valueOf(moduleIndex)).concat("-").concat(String.valueOf(blockIndex));
		return newLayerName;
	}

	private ComputationGraphConfiguration.GraphBuilder convBlock(ComputationGraphConfiguration.GraphBuilder graph,String moduleName,int moduleIndex,int blockIndex,String input,int[] kernelSize, int[] stride,int in,int out,ConvolutionMode convolutionMode) {

		ConvolutionLayer.Builder builder=new ConvolutionLayer.Builder(kernelSize, stride);

		if(in>0){

			builder.nIn(in);
		}

		graph.addLayer(createLayerName(moduleName,CNN,moduleIndex,blockIndex),
				builder
						.nOut(out)
						.convolutionMode(convolutionMode)
						.cudnnAlgoMode(cudnnAlgoMode)
						.build(),
				input);



		return graph;
	}


    private ComputationGraphConfiguration.GraphBuilder convBlock(ComputationGraphConfiguration.GraphBuilder graph,String moduleName,int moduleIndex,int blockIndex,String input,int[] kernelSize, int[] stride,int in,int out,ConvolutionMode convolutionMode,Activation activation) {

        ConvolutionLayer.Builder builder=new ConvolutionLayer.Builder(kernelSize, stride);

        if(in>0){

            builder.nIn(in);
        }

        graph.addLayer(createLayerName(moduleName,CNN,moduleIndex,blockIndex),
                builder
                        .nOut(out)
                        .activation(activation)
                        .convolutionMode(convolutionMode)
                        .cudnnAlgoMode(cudnnAlgoMode)
                        .build(),
                input);
        return graph;
    }


	private void batchNormAndActivation(ComputationGraphConfiguration.GraphBuilder graph,String batchNormAndActivationInput, String moduleName,int moduleIndex,int blockIndex) {
		graph.addLayer(createLayerName(moduleName,"batch",moduleIndex,blockIndex),
				new BatchNormalization.Builder(false)
						.decay(0.99)
						.eps(0.001)
						.build(),
				batchNormAndActivationInput);
		graph.addLayer(createLayerName(moduleName,"activation-layer",moduleIndex,blockIndex),
				new ActivationLayer.Builder()
						.activation(Activation.RELU)
						.build(), createLayerName(moduleName,"batch",moduleIndex,blockIndex));
	}

	private ComputationGraphConfiguration.GraphBuilder convBlock(ComputationGraphConfiguration.GraphBuilder graph,String moduleName,int moduleIndex,int blockIndex,String input,int[] kernelSize, int[] stride,int out,ConvolutionMode convolutionMode) {
		int in=0;
		return convBlock(graph, moduleName, moduleIndex,blockIndex, input, kernelSize, stride,in, out, convolutionMode);
	}

	private ComputationGraphConfiguration.GraphBuilder convBlock(ComputationGraphConfiguration.GraphBuilder graph,String moduleName,int moduleIndex,int blockIndex,String input,int[] kernelSize,int out,ConvolutionMode convolutionMode) {
		int in=0;
		int[] stride= {1,1};
		return convBlock(graph, moduleName,moduleIndex, blockIndex, input, kernelSize, stride,in, out, convolutionMode);
	}
	private ComputationGraphConfiguration.GraphBuilder convBlock(ComputationGraphConfiguration.GraphBuilder graph,String moduleName,int moduleIndex,int blockIndex,String input,int[] kernelSize,int out) {
		int in=0;
		int[] stride= {1,1};
		ConvolutionMode convolutionMode=ConvolutionMode.Same;
		return convBlock(graph, moduleName,moduleIndex, blockIndex, input, kernelSize, stride,in, out, convolutionMode);
	}

	private ComputationGraphConfiguration.GraphBuilder MaxPooling2D(ComputationGraphConfiguration.GraphBuilder graph,int[] kernelSize, int[] stride,ConvolutionMode convolutionMode, String moduleName, Integer moduleIndex, Integer blockIndex,String input) {

		graph

				.addLayer(createLayerName(moduleName,"max-pooling",moduleIndex,blockIndex),
						new SubsamplingLayer.Builder(
								SubsamplingLayer.PoolingType.MAX,
								kernelSize,
								stride)
								.convolutionMode(convolutionMode)
								.build(),
						input);
		return graph;
	}

	private ComputationGraphConfiguration.GraphBuilder AveragePooling2D(ComputationGraphConfiguration.GraphBuilder graph,int[] kernelSize, int[] stride,ConvolutionMode convolutionMode, String moduleName, Integer moduleIndex, Integer blockIndex,String input) {

		graph

				.addLayer(createLayerName(moduleName,"avg-pooling",moduleIndex,blockIndex),
						new SubsamplingLayer.Builder(
								SubsamplingLayer.PoolingType.AVG,
								kernelSize,
								stride)
								.convolutionMode(convolutionMode)
								.build(),
						input);
		return graph;
	}

	private ComputationGraphConfiguration.GraphBuilder MaxPooling2D(ComputationGraphConfiguration.GraphBuilder graph, String moduleName,Integer moduleIndex,Integer blockIndex,String input,int[] kernelSize, int[] stride,ConvolutionMode convolutionMode) {

		graph

				.addLayer(createLayerName(moduleName,"max-pooling",moduleIndex,blockIndex),
						new SubsamplingLayer.Builder(
								SubsamplingLayer.PoolingType.MAX,
								kernelSize,
								stride)
								.convolutionMode(convolutionMode)
								.build(),
						input);
		return graph;
	}

	private ComputationGraphConfiguration.GraphBuilder AveragePooling2D(ComputationGraphConfiguration.GraphBuilder graph, String moduleName, Integer moduleIndex,Integer blockIndex,String input,int[] kernelSize, int[] stride,ConvolutionMode convolutionMode) {

		graph

				.addLayer(createLayerName(moduleName,"avg-pooling",moduleIndex,blockIndex),
						new SubsamplingLayer.Builder(
								SubsamplingLayer.PoolingType.AVG,
								kernelSize,
								stride)
								.convolutionMode(convolutionMode)
								.build(),
						input);
		return graph;
	}





	@Override
    public ModelMetaData metaData() {
        return new ModelMetaData(new int[][] {inputShape}, 1, ZooType.CNN);
    }

    @Override
    public void setInputShape(int[][] inputShape) {
        this.inputShape = inputShape[0];
    }

}
