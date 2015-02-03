package org.deeplearning4j.optimize.api;


import org.deeplearning4j.nn.api.Model;

/**
 * Each epoch the listener is called, mainly used for debugging or visualizations
 * @author Adam Gibson
 *
 */
public interface IterationListener {
	/**
	 * Event listener for each iteration
	 * @param iteration the iteration
     * @param model the model iterating
	 */
	void iterationDone(Model model,int iteration);
	
}
