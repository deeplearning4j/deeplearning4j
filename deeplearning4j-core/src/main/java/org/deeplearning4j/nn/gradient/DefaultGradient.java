package org.deeplearning4j.nn.gradient;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

/**
 * Default gradient implementation. Basically lookup table
 * for ndarrays
 *
 * @author Adam Gibson
 */
public class DefaultGradient implements Gradient {
    private Map<String,INDArray> gradients = new LinkedHashMap<>();



    @Override
    public Map<String, INDArray> gradientLookupTable() {
        return gradients;
    }

    @Override
    public INDArray gradient(List<String> order) {
        List<INDArray> ret = new ArrayList<>();
        for(String s : order) {
            if(!gradientLookupTable().containsKey(s))
                throw new IllegalStateException("Illegal key " + s + " no gradient with key found");
            ret.add(gradientLookupTable().get(s));
        }
        return Nd4j.toFlattened(ret);
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
