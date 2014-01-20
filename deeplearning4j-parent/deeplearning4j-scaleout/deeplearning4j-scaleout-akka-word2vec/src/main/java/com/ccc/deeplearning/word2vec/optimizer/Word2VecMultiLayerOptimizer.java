package com.ccc.deeplearning.word2vec.optimizer;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.StringTokenizer;

import org.apache.commons.io.FileUtils;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.deeplearning.optimize.MultiLayerNetworkOptimizer;
import com.ccc.deeplearning.word2vec.Word2Vec;
import com.ccc.deeplearning.word2vec.ner.InputHomogenization;
import com.ccc.deeplearning.word2vec.nn.multilayer.Word2VecMultiLayerNetwork;
import com.ccc.deeplearning.word2vec.util.WordConverter;

public class Word2VecMultiLayerOptimizer extends MultiLayerNetworkOptimizer {


	private static final long serialVersionUID = 6744897075130553165L;
	protected Word2VecMultiLayerNetwork network;
	protected Word2Vec vec;
	private static Logger log = LoggerFactory.getLogger(Word2VecMultiLayerOptimizer.class);

	public Word2VecMultiLayerOptimizer(Word2VecMultiLayerNetwork network, double lr,Word2Vec vec) {
		super(network, lr);
		this.network = (Word2VecMultiLayerNetwork) super.network;
		this.vec = vec;
	}

	@Override
	public void optimize(DoubleMatrix labels, double lr, int epochs) {
		throw new UnsupportedOperationException("Please use the word inputs, we are unable to update word vectors without knowing the input words");


	}


	public void optimize(double lr, int epochs,List<String> examples,List<String> labels) {

		WordConverter converter = new WordConverter(examples,vec);
		DoubleMatrix labelMatrix = converter.toLabelMatrix(labels);
		//sample from the final layer in the network and train on the result
		DoubleMatrix layerInput = network.sigmoidLayers[network.sigmoidLayers.length - 1].sample_h_given_v();
		network.logLayer.input = layerInput;
		network.logLayer.labels = labelMatrix;
		Double crossEntropy = null;
		StringBuffer debug = new StringBuffer();
		for(int i = 0; i < epochs; i++) {
			network.logLayer.train(layerInput,labelMatrix, lr);
			lr *= network.learningRateUpdate;
			/*for(String example : examples) {
				StringTokenizer tokenizer = new StringTokenizer(new InputHomogenization(example).transform());
				while(tokenizer.hasMoreTokens()) {
					String token = tokenizer.nextToken();
					if(network.vec.getVocab().containsKey(token)) {
						DoubleMatrix curr = network.vec.getWordVectorMatrix(token).mul(lr);
						DoubleMatrix gradient = curr.sub(network.vec.getWordVectorMatrix(token));
						DoubleMatrix add = network.vec.getWordVectorMatrix(token).add(gradient);
						int idx = network.vec.indexOf(token);
						network.vec.getSyn0().putRow(idx, add);
						log.info("Updated word " + token);
					}
				}
			}*/
			if(crossEntropy == null)
				crossEntropy = network.negativeLogLikelihood();
			else {
				Double currError = network.negativeLogLikelihood();
				Double diff = Math.abs(currError - crossEntropy);
				crossEntropy = currError;
			}
			debug.append(i + "," + crossEntropy + "\n");


			log.info("Negative log likelihood on epoch " + i + " " + network.negativeLogLikelihood());
		}

		try {
			FileUtils.writeStringToFile(new File("/home/agibsonccc/Desktop/layer-logistic.csv"), debug.toString());
		} catch (IOException e) {
			throw new RuntimeException(e);
		}

	}







}
