package org.deeplearning4j.clustering.cluster;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class CentersHolder {
    private INDArray centers;
    private Map<String, Double> clusterByCenter = new HashMap<>();

    public CentersHolder(long[] shape) {
        this.centers = Nd4j.create(shape);
    }

    public void addCenter(INDArray pointView) {
        centers.addRowVector(pointView);
    }

    public double getMinDistanceToCenter(Point point, String distanceFunction) {
        return Nd4j.getExecutioner().execAndReturn(
                ClusterUtils.createDistanceFunctionOp(distanceFunction, centers, point.getArray())).getFinalResult().doubleValue();

    }

    public Pair<Double, Long> getCenterByMinDistance(Point point, String distanceFunction) {
        INDArray minDistances = Nd4j.getExecutioner().exec(ClusterUtils.createDistanceFunctionOp(distanceFunction, centers, point.getArray()));
        INDArray index = Nd4j.argMin(minDistances, 0);
        Pair<Double, Long> result = new Pair<>();
        result.setFirst(minDistances.getDouble(0));
        result.setSecond(index.getLong(0));
        return result;
    }
}
