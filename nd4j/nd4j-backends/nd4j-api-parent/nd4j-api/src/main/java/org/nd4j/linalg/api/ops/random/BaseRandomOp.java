package org.nd4j.linalg.api.ops.random;

import lombok.NoArgsConstructor;
import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.BaseOp;
import org.nd4j.linalg.api.ops.RandomOp;
import org.nd4j.linalg.api.shape.Shape;

import java.util.ArrayList;
import java.util.List;

/**
 * @author raver119@gmail.com
 */
@NoArgsConstructor
public abstract class BaseRandomOp extends BaseOp implements RandomOp {

    public BaseRandomOp(SameDiff sameDiff,
                            SDVariable i_v) {
        if (i_v != null) {
            this.sameDiff = sameDiff;
            this.xVertexId = i_v.getVarName();
            sameDiff.addArgsFor(new String[]{xVertexId},this);
            if(Shape.isPlaceholderShape(i_v.getShape())) {
                sameDiff.addPropertyToResolve(this,i_v.getVarName());
            }
        } else {
            throw new IllegalArgumentException("Input can't be null.");
        }
    }

    @Override
    public Type opType() {
        return Type.RANDOM;
    }

    @Override
    public List<long[]> calculateOutputShape() {
        List<long[]> ret = new ArrayList<>(1);
        val shape = sameDiff.getShapeForVarName(args()[0].getVarName());
        if(shape != null)
            ret.add(shape);
        return ret;
    }


}
