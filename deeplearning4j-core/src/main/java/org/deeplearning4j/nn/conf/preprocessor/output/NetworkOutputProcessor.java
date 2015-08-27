package org.deeplearning4j.nn.conf.preprocessor.output;

import java.io.Serializable;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;

/** NetworkOutputProcessor processes both final network output
 * (during forward pass) and the labels (during backprop).<br>
 * Can be used to do things like reshape output and labels, for
 * example.
 * @author Alex Black
 */
@JsonTypeInfo(use= JsonTypeInfo.Id.NAME, include= JsonTypeInfo.As.WRAPPER_OBJECT)
@JsonSubTypes(value={
        @JsonSubTypes.Type(value = RnnOutputProcessor.class, name = "rnnOutputProcessor"),
})
public interface NetworkOutputProcessor extends Serializable, Cloneable {
	
	/** Process the output of the final layer*/
	INDArray processOutput( INDArray output, MultiLayerNetwork network );
	
	/** Process the labels before passing them to the
	 * output layer during backprop.
	 */
	INDArray processLabels( INDArray labels, MultiLayerNetwork network );

}
