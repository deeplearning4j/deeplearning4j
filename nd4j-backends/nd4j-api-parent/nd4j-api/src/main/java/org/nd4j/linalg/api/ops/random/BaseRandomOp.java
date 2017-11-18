package org.nd4j.linalg.api.ops.random;

import org.nd4j.linalg.api.ops.BaseOp;
import org.nd4j.linalg.api.ops.RandomOp;

import java.util.ArrayList;
import java.util.List;

/**
 * @author raver119@gmail.com
 */
public abstract class BaseRandomOp extends BaseOp implements RandomOp {


    @Override
    public List<int[]> calculateOutputShape() {
        List<int[]> ret = new ArrayList<>(1);
        ret.add(this.shape);
        return ret;
    }


}
