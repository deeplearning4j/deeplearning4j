package org.deeplearning4j.optimize.api;


import org.deeplearning4j.nn.api.Model;

import java.io.Serializable;

/**
 * Each epoch the listener is called, mainly used for debugging or visualizations
 * @author Adam Gibson
 *
 */
public interface IterationListener extends Serializable {
	/**
	 * Event listener for each iteration
	 * @param iteration the iteration
     * @param model the model iterating
	 */
	void iterationDone(Model model,int iteration);
	
}
