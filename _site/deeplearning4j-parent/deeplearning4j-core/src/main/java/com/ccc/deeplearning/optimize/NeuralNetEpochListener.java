package com.ccc.deeplearning.optimize;

import com.ccc.deeplearning.nn.NeuralNetwork;

public interface NeuralNetEpochListener {

	void epochDone(int epoch);
	
}
