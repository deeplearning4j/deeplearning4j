package com.ccc.deeplearning.word2vec.updateable;

import java.util.ArrayList;
import java.util.List;

import com.ccc.deeplearning.word2vec.nn.multilayer.Word2VecMultiLayerNetwork;

public class Word2VecAccumulator {
	private List<Word2VecMultiLayerNetwork> workers = new ArrayList<Word2VecMultiLayerNetwork>();

	public void accumulate(Word2VecMultiLayerNetwork sda) {
		workers.add(sda);
	}

	public Word2VecMultiLayerNetwork averaged() {
		if(workers.isEmpty())
			return null;


	

		Word2VecMultiLayerNetwork first = workers.get(0);
		Word2VecMultiLayerNetwork ret = first;



		//sum and scale each of the weight vectors
		//start with the second worker as the baseline
		for(int worker = 1; worker <  workers.size(); worker++) {
			Word2VecMultiLayerNetwork network = workers.get(worker);
			first.merge(network, workers.size());

		}

	
		return ret;
	}

}
