package org.deeplearning4j.ui.weights;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;

/**
 * This IterationListener is suited for general model performance/architecture overview
 *
 * PLEASE NOTE: WORK IN PROGRESS, DO NOT USE IT
 * @author raver119@gmail.com
 */
public class FlowIterationListener implements IterationListener {
    /**
     * Get if listener invoked
     */
    @Override
    public boolean invoked() {
        return false;
    }

    /**
     * Change invoke to true
     */
    @Override
    public void invoke() {

    }

    /**
     * Event listener for each iteration
     *
     * @param model     the model iterating
     * @param iteration the iteration
     */
    @Override
    public void iterationDone(Model model, int iteration) {
        /*
            Basic plan:
                1. We should detect, if that's CompGraph or MultilayerNetwork. However the actual difference will be limited to number of non-linear connections.
                2. Network structure should be converted to JSON
                3. Params for each node should be packed to JSON as well
                4. For specific cases (like CNN) binary data should be wrapped into base64
                5. For arrays/params gzip could be used (to be investigated)
                ......
                Later, on client side, this JSON should be parsed and rendered. So, proper object structure to be considered.
         */
        if (model instanceof ComputationGraph) {

        } else if (model instanceof MultiLayerNetwork) {

        } else throw new IllegalStateException("");
    }
}
