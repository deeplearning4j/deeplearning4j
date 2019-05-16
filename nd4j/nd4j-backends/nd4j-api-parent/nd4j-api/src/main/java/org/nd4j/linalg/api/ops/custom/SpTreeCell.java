package org.nd4j.linalg.api.ops.custom;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;

public class SpTreeCell extends DynamicCustomOp {
    public SpTreeCell(INDArray corner, INDArray width, INDArray point, long N,
                      boolean contains) {
        inputArguments.add(corner);
        inputArguments.add(width);
        inputArguments.add(point);

        iArguments.add(N);

        outputArguments.add(Nd4j.scalar(contains));
    }

    @Override
    public String opName() {
        return "cell_contains";
    }
}
