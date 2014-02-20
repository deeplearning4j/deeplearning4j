package org.deeplearning4j.optimize;

import org.deeplearning4j.nn.NeuralNetwork;

public interface NeuralNetEpochListener {

	void epochDone(int epoch);
	
}
