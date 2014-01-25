package com.ccc.deeplearning.word2vec.optimizer;

import java.util.List;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.deeplearning.optimize.MultiLayerNetworkOptimizer;
import com.ccc.deeplearning.word2vec.Word2Vec;
import com.ccc.deeplearning.word2vec.iterator.Word2VecDataSetIterator;
import com.ccc.deeplearning.word2vec.nn.multilayer.Word2VecMultiLayerNetwork;
import com.ccc.deeplearning.word2vec.util.Window;
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

	public void optimize(double lr, int epochs,Word2VecDataSetIterator iter,List<String> labels) {

		while(iter.hasNext()) {
			List<Window> next = iter.next();
			DoubleMatrix labelMatrix = WordConverter.toLabelMatrix(labels, next);
			network.labels = labelMatrix;
			//sample from the final layer in the network and train on the result
			super.optimize(labelMatrix, lr, epochs);
		}


	}

	public void optimize(double lr, int epochs,List<String> examples,List<String> labels) {

		WordConverter converter = new WordConverter(examples,vec);
		DoubleMatrix labelMatrix = converter.toLabelMatrix(labels);
		network.labels = labelMatrix;
		//sample from the final layer in the network and train on the result
		super.optimize(labelMatrix, lr, epochs);
	}







}
