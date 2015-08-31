package org.deeplearning4j.nn.layers.recurrent;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.nd4j.linalg.api.ndarray.INDArray;

public abstract class BaseRecurrentLayer<LayerConfT extends org.deeplearning4j.nn.conf.layers.Layer> extends BaseLayer<LayerConfT> {

	/** stateMap stores the INDArrays needed to do rnnTimeStep() forward pass. */
	protected Map<String,INDArray> stateMap = new ConcurrentHashMap<>();

	public BaseRecurrentLayer(NeuralNetConfiguration conf) {
		super(conf);
	}

	public BaseRecurrentLayer(NeuralNetConfiguration conf, INDArray input) {
		super(conf, input);
	}

	/**Do one or more time steps using the previous time step state stored in stateMap.<br>
	 * Can be used to efficiently do forward pass one or n-steps at a time (instead of doing
	 * forward pass always from t=0)<br>
	 * If stateMap is empty, default initialization (usually zeros) is used<br>
	 * Implementations also update stateMap at the end of this method
	 * @param input Input to this layer
	 * @return activations
	 */
	public abstract INDArray rnnTimeStep(INDArray input);

	/** Returns a shallow copy of the stateMap */
	public Map<String,INDArray> rnnGetPreviousState(){
		return new HashMap<>(stateMap);
	}

	/** Set the state map. Values set using this method will be used
	 * in next call to rnnTimeStep()
	 */
	public void rnnSetPreviousState(Map<String,INDArray> stateMap){
		this.stateMap.clear();
		this.stateMap.putAll(stateMap);
	}

	/** Reset/clear the stateMap for rnnTimeStep() */
	public void rnnClearPreviousState(){
		stateMap.clear();
	}
}
