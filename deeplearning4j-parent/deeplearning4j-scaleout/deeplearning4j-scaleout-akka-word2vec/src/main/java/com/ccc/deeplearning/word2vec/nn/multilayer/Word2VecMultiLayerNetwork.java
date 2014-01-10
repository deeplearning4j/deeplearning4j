package com.ccc.deeplearning.word2vec.nn.multilayer;

import java.util.List;

import org.jblas.DoubleMatrix;

import com.ccc.deeplearning.dbn.CDBN;
import com.ccc.deeplearning.nn.BaseMultiLayerNetwork;
import com.ccc.deeplearning.util.MatrixUtil;
import com.ccc.deeplearning.word2vec.Word2Vec;
import com.ccc.deeplearning.word2vec.ner.InputHomogenization;
import com.ccc.deeplearning.word2vec.optimizer.Word2VecMultiLayerOptimizer;
import com.ccc.deeplearning.word2vec.util.Window;
import com.ccc.deeplearning.word2vec.util.WindowConverter;
import com.ccc.deeplearning.word2vec.util.Windows;
import com.ccc.deeplearning.word2vec.util.WordConverter;

public class Word2VecMultiLayerNetwork extends CDBN {

	
	private static final long serialVersionUID = -289194661963198148L;
	public Word2Vec vec;
	protected Word2VecMultiLayerOptimizer optimizer;
	protected List<String> labels;
	@Override
	public void finetune(double lr, int epochs) {
		throw new UnsupportedOperationException("Please use the other method. We need to update word2vec as well");
	}
	
	
	public void finetune(double lr, int epochs,List<String> words,List<String> labels) {
		if(this.labels == null)
			this.labels = labels;
		
		if(optimizer == null)
			optimizer = new Word2VecMultiLayerOptimizer(this, lr,vec);
		
		optimizer.optimize(lr, epochs,words,labels);
		
	}
	
	@Override
	public void finetune(DoubleMatrix labels, double lr, int epochs) {
		throw new UnsupportedOperationException("Please use the other method. We need to update word2vec as well");
	}

	
	
	public DoubleMatrix predict(String words) {
		List<Window> windows = Windows.windows(new InputHomogenization(words).transform());
		double[][] ret = new double[windows.size()][labels.size()];

		
		
		for(int i = 0; i < windows.size(); i++) {
			Window window = windows.get(i);
			DoubleMatrix prediction = MatrixUtil.normalize(new DoubleMatrix(WindowConverter.asExample(window,vec)));
			DoubleMatrix toPredict = super.predict(prediction);
			ret[i] = toPredict.toArray();
		}
		return new DoubleMatrix(ret);
	}
	
	@Override
	public DoubleMatrix predict(DoubleMatrix x) {
		throw new UnsupportedOperationException("Please input words. This network will not accept arbitrary input.");
	}
	
	@Override
	protected void update(BaseMultiLayerNetwork matrix) {
		super.update(matrix);
		Word2VecMultiLayerNetwork network = (Word2VecMultiLayerNetwork) matrix;
		this.vec = network.vec;
		
	}
	
	
	@Override
	public void merge(BaseMultiLayerNetwork network, int batchSize) {
		super.merge(network, batchSize);
		Word2VecMultiLayerNetwork network2 = (Word2VecMultiLayerNetwork) network;
		DoubleMatrix curr = vec.getSyn0();
		DoubleMatrix delta = network2.vec.getSyn0();
		if(delta.length != curr.length)
			throw new IllegalArgumentException("Unmergable, must be at least same length");
		for(int i = 0; i < curr.rows; i++) {
			//detected a difference; this is updateable
			double diff = delta.sum() - curr.sum();
			if(diff != 0) {
				DoubleMatrix gradient = delta.add(curr).div(batchSize);
				vec.getSyn0().putRow(i,gradient);
			}
		}
	}
	
	
	
	
	@Override
	public void trainNetwork(DoubleMatrix input, DoubleMatrix labels,
			Object[] otherParams) {
		throw new UnsupportedOperationException("Please use the proper inputs");
	}

	public void trainNetwork(List<String> examples,
			Object[] otherParams) {
		WordConverter converter = new WordConverter(examples,vec);
		super.trainNetwork(converter.toInputMatrix(), converter.toLabelMatrix(labels), otherParams);
	}

	@Override
	public void pretrain(DoubleMatrix input, int k, double learningRate,
			int epochs) {
		throw new UnsupportedOperationException("Please use the proper inputs");

	}

	
	public void pretrain(List<String> input, int k, double learningRate,
			int epochs) {
		WordConverter converter = new WordConverter(input,vec);
		super.pretrain(converter.toInputMatrix(), k, learningRate, epochs);

	}
	@Override
	public void pretrain(int k, double learningRate, int epochs) {
		throw new UnsupportedOperationException("Please use the proper inputs");

	}




	@Override
	protected void initializeLayers(DoubleMatrix input) {
		super.initializeLayers(input);
		//this.layers[0].getW().mul(vec.getVocab().size());
	}




	public static class Builder extends BaseMultiLayerNetwork.Builder<Word2VecMultiLayerNetwork> {
		protected Word2Vec vec;
		
		public Builder() {
			super();
			this.clazz = Word2VecMultiLayerNetwork.class;
		}
		
		
		public Builder withWord2Vec(Word2Vec vec) {
			this.vec = vec;
			return this;
		}
		
		@Override
		public Word2VecMultiLayerNetwork build() {
			Word2VecMultiLayerNetwork ret = super.build();
			ret.vec = vec;
			return ret;
		}
		
		
		
		
		
	}

	
}
