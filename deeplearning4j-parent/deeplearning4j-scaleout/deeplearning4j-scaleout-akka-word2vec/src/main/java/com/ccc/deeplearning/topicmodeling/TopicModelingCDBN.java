package com.ccc.deeplearning.topicmodeling;

import org.apache.commons.math3.random.RandomGenerator;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.deeplearning.dbn.CDBN;
import com.ccc.deeplearning.dbn.DBN;
import com.ccc.deeplearning.nn.NeuralNetwork;
import com.ccc.deeplearning.sda.StackedDenoisingAutoEncoder;

public class TopicModelingCDBN extends DBN {

	/**
	 * 
	 */
	private static final long serialVersionUID = 8284466149702619426L;
	private static Logger log = LoggerFactory.getLogger(TopicModelingCDBN.class);

	public TopicModelingCDBN() {
		super();
		
	}

	public TopicModelingCDBN(int n_ins, int[] hidden_layer_sizes, int n_outs,
			int n_layers, RandomGenerator rng, DoubleMatrix input,
			DoubleMatrix labels) {
		super(n_ins, hidden_layer_sizes, n_outs, n_layers, rng, input, labels);
		
	}

	public TopicModelingCDBN(int n_ins, int[] hidden_layer_sizes, int n_outs,
			int n_layers, RandomGenerator rng) {
		super(n_ins, hidden_layer_sizes, n_outs, n_layers, rng);
		
	}

	@Override
	public NeuralNetwork createLayer(DoubleMatrix input, int nVisible,
			int nHidden, DoubleMatrix W, DoubleMatrix hBias,
			DoubleMatrix vBias, RandomGenerator rng, int index) {
		NeuralNetwork ret =  super.createLayer(input, nVisible, nHidden, W, hBias, vBias, rng, index);
		if(index == 0) {
			ret.getvBias().muli(input.columns);
			log.info("Augmented weights of first layer by " + input.columns);
		}
		return ret;
	}


	public static class Builder extends DBN.Builder {

		public Builder() {
			super();
			this.clazz = TopicModelingCDBN.class;
		}

		@Override
		public TopicModelingCDBN buildEmpty() {
			return (TopicModelingCDBN) super.buildEmpty();
		}

		@Override
		public TopicModelingCDBN build() {
			return (TopicModelingCDBN) super.build();
		}
		
		
	}

}
