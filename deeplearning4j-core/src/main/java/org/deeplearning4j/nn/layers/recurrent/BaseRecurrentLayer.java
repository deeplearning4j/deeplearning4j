package org.deeplearning4j.nn.layers.recurrent;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.nd4j.linalg.api.ndarray.INDArray;

public abstract class BaseRecurrentLayer<LayerConfT extends org.deeplearning4j.nn.conf.layers.Layer> extends BaseLayer<LayerConfT> {

	protected Map<String,INDArray> stateMap = new ConcurrentHashMap<>();
	
	public BaseRecurrentLayer(NeuralNetConfiguration conf) {
		super(conf);
	}

	public BaseRecurrentLayer(NeuralNetConfiguration conf, INDArray input) {
		super(conf, input);
	}
	
	public abstract INDArray rnnTimeStep(INDArray input);
	
	public Map<String,INDArray> rnnGetPreviousState(){
		return new HashMap<>(stateMap);
	}
	
	public void rnnSetPreviousState(Map<String,INDArray> stateMap){
		this.stateMap.clear();
		this.stateMap.putAll(stateMap);
	}
	
	public void rnnClearPreviousState(){
		stateMap.clear();
	}	
}
