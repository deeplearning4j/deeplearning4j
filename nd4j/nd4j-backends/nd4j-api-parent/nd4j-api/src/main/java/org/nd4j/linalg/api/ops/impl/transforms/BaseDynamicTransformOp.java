package org.nd4j.linalg.api.ops.impl.transforms;

import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public abstract class BaseDynamicTransformOp extends DynamicCustomOp {

    public BaseDynamicTransformOp() {}

    public BaseDynamicTransformOp(SameDiff sameDiff, SDVariable[] args, boolean inPlace) {
        super(null, sameDiff, args, inPlace);
    }

    public BaseDynamicTransformOp(INDArray[] inputs, INDArray[] outputs) {
        super(null, inputs, outputs);
    }


    @Override
    public List<long[]> calculateOutputShape() {
        val args = args();
        if(args.length < 2) {
            if(args[0] == null || args[0].getShape() == null) {
                return Collections.emptyList();
            }

            return Arrays.asList(args[0].getShape());
        }

        val firstArgShape = args[0].getShape();
        val secondArgShape = args[1].getShape();
        if(args[0] == null || args[0].getShape() == null) {
            return Collections.emptyList();
        }

        if(args[1] == null || args[1].getShape() == null) {
            return Collections.emptyList();
        }

        if(Arrays.equals(firstArgShape, secondArgShape)){
            return Collections.singletonList(firstArgShape);
        }
        //Handle broadcast shape: [1,4]+[3,1] = [3,4]
        Shape.assertBroadcastable(firstArgShape, secondArgShape);
        val outShape = Shape.broadcastOutputShape(firstArgShape, secondArgShape);

        return Collections.singletonList(outShape);
    }
}
