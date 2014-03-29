package org.deeplearning4j.optimize;


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
	void epochDone(int epoch);
	
}
