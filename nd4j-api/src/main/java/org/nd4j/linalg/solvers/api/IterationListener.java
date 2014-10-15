package org.nd4j.linalg.solvers.api;


/**
 * Each epoch the listener is called, mainly used for debugging or visualizations
 * @author Adam Gibson
 *
 */
public interface IterationListener {
	/**
	 * Event listener for each iteration
	 * @param epoch
	 */
	void iterationDone(int epoch);
	
}
