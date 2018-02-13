package org.nd4j.list;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * An {@link BaseNDArrayList} for integers
 *
 * @author Adam Gibson
 */
public class IntNDArrayList extends BaseNDArrayList<Integer> {
    public IntNDArrayList() {
    }

    public IntNDArrayList(INDArray container) {
        super(container);
    }


    @Override
    public Integer get(int i) {
        Number ret = container.getDouble(i);
        return ret.intValue();
    }


}
