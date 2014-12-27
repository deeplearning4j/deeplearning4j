package org.deeplearning4j.nn.gradient;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.Map;

/**
 * Default gradient implementation. Basically lookup table
 * for ndarrays
 *
 * @author Adam Gibson
 */
public class DefaultGradient implements Gradient {
    private Map<String,INDArray> gradients = new HashMap<>();



    @Override
    public Map<String, INDArray> gradientLookupTable() {
        return gradients;
    }

    @Override
    public INDArray gradient() {
        return Nd4j.toFlattened(gradients.values());
    }

    @Override
    public void clear() {
        gradients.clear();
    }
}
