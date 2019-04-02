package org.deeplearning4j.clustering.cluster;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

public class CentersHolder {
    private INDArray centers;

    public CentersHolder(long[] shape) {
        this.centers = Nd4j.create(shape);
    }

    public void addCenter(INDArray pointView) {
        centers.addRowVector(pointView);
    }

    public double getMinDistanceToCenter(Point point, String distanceFunction) {
        return Nd4j.getExecutioner().execAndReturn(
                ClusterUtils.createDistanceFunctionOp(distanceFunction, centers, point.getArray()))
                .getFinalResult().doubleValue();
    }
}
