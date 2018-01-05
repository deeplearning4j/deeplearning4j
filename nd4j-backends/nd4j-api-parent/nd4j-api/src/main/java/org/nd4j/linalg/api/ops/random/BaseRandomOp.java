package org.nd4j.linalg.api.ops.random;

import lombok.val;
import org.nd4j.linalg.api.ops.BaseOp;
import org.nd4j.linalg.api.ops.RandomOp;

import java.util.ArrayList;
import java.util.List;

/**
 * @author raver119@gmail.com
 */
public abstract class BaseRandomOp extends BaseOp implements RandomOp {

    @Override
    public Type opType() {
        return Type.RANDOM;
    }

    @Override
    public List<int[]> calculateOutputShape() {
        List<int[]> ret = new ArrayList<>(1);
        val shape = sameDiff.getShapeForVarName(args()[0].getVarName());
        if(shape != null)
            ret.add(shape);
        return ret;
    }


}
