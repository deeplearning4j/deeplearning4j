package org.deeplearning4j.nn.layers.recurrent;

import org.deeplearning4j.nn.api.layers.RecurrentLayer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public abstract class BaseRecurrentLayer<LayerConfT extends org.deeplearning4j.nn.conf.layers.BaseLayer>
                extends BaseLayer<LayerConfT> implements RecurrentLayer {

    /**
     * stateMap stores the INDArrays needed to do rnnTimeStep() forward pass.
     */
    protected Map<String, INDArray> stateMap = new ConcurrentHashMap<>();

    /**
     * State map for use specifically in truncated BPTT training. Whereas stateMap contains the
     * state from which forward pass is initialized, the tBpttStateMap contains the state at the
     * end of the last truncated bptt
     */
    protected Map<String, INDArray> tBpttStateMap = new ConcurrentHashMap<>();

    public BaseRecurrentLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    public BaseRecurrentLayer(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
    }

    /**
     * Returns a shallow copy of the stateMap
     */
    @Override
    public Map<String, INDArray> rnnGetPreviousState() {
        return new HashMap<>(stateMap);
    }

    /**
     * Set the state map. Values set using this method will be used
     * in next call to rnnTimeStep()
     */
    @Override
    public void rnnSetPreviousState(Map<String, INDArray> stateMap) {
        this.stateMap.clear();
        this.stateMap.putAll(stateMap);
    }

    /**
     * Reset/clear the stateMap for rnnTimeStep() and tBpttStateMap for rnnActivateUsingStoredState()
     */
    @Override
    public void rnnClearPreviousState() {
        stateMap.clear();
        tBpttStateMap.clear();
    }

    @Override
    public Map<String, INDArray> rnnGetTBPTTState() {
        return new HashMap<>(tBpttStateMap);
    }

    @Override
    public void rnnSetTBPTTState(Map<String, INDArray> state) {
        tBpttStateMap.clear();
        tBpttStateMap.putAll(state);
    }

}
