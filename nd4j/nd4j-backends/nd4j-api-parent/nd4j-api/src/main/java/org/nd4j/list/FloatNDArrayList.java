package org.nd4j.list;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * An {@link BaseNDArrayList} for float
 *
 * @author Adam Gibson
 */
public class FloatNDArrayList extends BaseNDArrayList<Float> {
    public FloatNDArrayList() {
    }

    public FloatNDArrayList(INDArray container) {
        super(container);
    }

    @Override
    public Float get(int i) {
        Number ret = container.getDouble(i);
        return ret.floatValue();
    }
}
