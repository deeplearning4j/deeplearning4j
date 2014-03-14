package org.deeplearning4j.gradient;

import java.io.Serializable;

import org.deeplearning4j.nn.gradient.NeuralNetworkGradient;

/**
 * Class for listening to gradient events.
 * This is primarily used for debugging gradients.
 * 
 * One primary use case is for understanding when
 * your gradients aren't converting right.
 * This allows you to do some basic numerical transforms
 * and averaging.
 * 
 * 
 * @author Adam Gibson
 *
 */
public interface NeuralNetworkGradientListener extends Serializable {

	/**
	 * Responds to gradient events
	 * @param gradient the gradient
	 */
	void onGradient(NeuralNetworkGradient gradient);
}
