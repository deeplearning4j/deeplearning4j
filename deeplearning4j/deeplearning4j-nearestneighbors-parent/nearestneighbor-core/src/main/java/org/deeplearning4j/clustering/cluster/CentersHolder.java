package org.deeplearning4j.clustering.cluster;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

public class CentersHolder {
    private INDArray centers;

    public CentersHolder(long[] shape) {
        this.centers = Nd4j.create(shape);
    }

    public void addCenter(INDArray pointView) {
        centers = centers.add(pointView);
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
