package org.deeplearning4j.optimize.api;


/**
 * Each epoch the listener is called, mainly used for debugging or visualizations
 * @author Adam Gibson
 *
 */
public interface NeuralNetEpochListener  {
	/**
	 * Event listener for each iteration
	 * @param epoch
	 */
	void iterationDone(int epoch);
	
}
