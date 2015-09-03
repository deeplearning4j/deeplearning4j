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

	/** State map for use specifically in truncated BPTT training. Whereas stateMap contains the
	 * state from which forward pass is initialized, the tBpttStateMap contains the state at the
	 * end of the last truncated bptt
	 * */
	protected Map<String,INDArray> tBpttStateMap = new ConcurrentHashMap<>();

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

	/** Reset/clear the stateMap for rnnTimeStep() and tBpttStateMap for rnnActivateUsingStoredState() */
	public void rnnClearPreviousState(){
		stateMap.clear();
		tBpttStateMap.clear();
	}

	/** Similar to rnnTimeStep, this method is used for activations using the state
	 * stored in the stateMap as the initialization. However, unlike rnnTimeStep this
	 * method does not alter the stateMap; therefore, unlike rnnTimeStep, multiple calls to
	 * this method (with identical input) will:<br>
	 * (a) result in the same output<br>
	 * (b) leave the state maps (both stateMap and tBpttStateMap) in an identical state
	 * @param input Layer input
	 * @param training if true: training. Otherwise: test
	 * @param storeLastForTBPTT If true: store the final state in tBpttStateMap for use in truncated BPTT training
	 * @return Layer activations
	 */
	public abstract INDArray rnnActivateUsingStoredState(INDArray input, boolean training, boolean storeLastForTBPTT);

	public Map<String,INDArray> rnnGetTBPTTState(){
		return new HashMap<>(tBpttStateMap);
	}

	public void rnnSetTBPTTState(Map<String,INDArray> state){
		tBpttStateMap.clear();
		tBpttStateMap.putAll(state);
	}
}
