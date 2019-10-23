package org.nd4j.linalg.api.ops.custom;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

public class KnnMinDistance extends DynamicCustomOp {

    public KnnMinDistance() {
    }

    public KnnMinDistance(INDArray point, INDArray lowest, INDArray highest, INDArray distance) {
        inputArguments.add(point);
        inputArguments.add(lowest);
        inputArguments.add(highest);

        outputArguments.add(distance);
    }

    @Override
    public String opName() {
        return "knn_mindistance";
    }
}
