package org.deeplearning4j.optimize.api;


/**
 * Each epoch the listener is called, mainly used for debugging or visualizations
 * @author Adam Gibson
 *
 */
public interface IterationListener {
	/**
	 * Event listener for each iteration
	 * @param iteration
	 */
	void iterationDone(int iteration);
	
}
