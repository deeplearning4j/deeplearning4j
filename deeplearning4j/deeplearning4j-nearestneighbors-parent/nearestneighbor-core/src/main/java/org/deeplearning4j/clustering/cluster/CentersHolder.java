package org.deeplearning4j.clustering.cluster;

import lombok.val;
import org.deeplearning4j.clustering.algorithm.Distance;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

public class CentersHolder {
    private INDArray centers;
    private long index = 0;

    public CentersHolder(long rows, long cols) {
        this.centers = Nd4j.create(rows, cols);
    }

    public void addCenter(INDArray pointView) {

        centers.putRow(index++, pointView);
    }

    public Pair<Double, Long> getCenterByMinDistance(Point point, Distance distanceFunction) {
        val op = ClusterUtils.createDistanceFunctionOp(distanceFunction, centers, point.getArray(), 1);
        INDArray minDistances = Nd4j.getExecutioner().exec(op);
        INDArray index = Nd4j.argMin(minDistances);
        Pair<Double, Long> result = new Pair<>();
        result.setFirst(minDistances.getDouble(0));
        result.setSecond(index.getLong(0));
        return result;
    }
}
