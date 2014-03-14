package org.deeplearning4j.gradient.multilayer;

import java.io.Serializable;

import org.deeplearning4j.nn.gradient.MultiLayerGradient;


/**
 * Multi layer networks emit a multi layer gradient whenever one is calculated
 * @author Adam Gibson
 *
 */
public interface MultiLayerGradientListener extends Serializable {

	void onMultiLayerGradient(MultiLayerGradient gradient);

}
