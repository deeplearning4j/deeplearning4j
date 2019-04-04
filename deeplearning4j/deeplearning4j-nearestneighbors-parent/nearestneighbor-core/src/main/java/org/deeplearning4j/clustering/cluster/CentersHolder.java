package org.deeplearning4j.clustering.cluster;

import lombok.val;
import org.deeplearning4j.clustering.algorithm.Distance;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.ReduceOp;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMin;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

public class CentersHolder {
    private INDArray centers;
    private long index = 0;

    protected transient ReduceOp op;
    protected IMin imin;
    protected transient INDArray distances;
    protected transient INDArray argMin;

    private long rows, cols;

    public CentersHolder(long rows, long cols) {
        this.rows = rows;
        this.cols = cols;
    }

    public synchronized void addCenter(INDArray pointView) {
        if (centers == null)
            this.centers = Nd4j.create(pointView.dataType(), new long[] {rows, cols});

        centers.putRow(index++, pointView);
    }

    public Pair<Double, Long> getCenterByMinDistance(Point point, Distance distanceFunction) {
        if (distances == null)
            distances = Nd4j.create(centers.dataType(), centers.rows());

        if (argMin == null)
            argMin = Nd4j.createUninitialized(DataType.LONG, new long[0]);

        if (op == null) {
            op = ClusterUtils.createDistanceFunctionOp(distanceFunction, centers, point.getArray(), 1);
            imin = new IMin(distances, argMin);
            op.setZ(distances);
        }

        op.setY(point.getArray());

        Nd4j.getExecutioner().exec(op);
        Nd4j.getExecutioner().exec(imin);

        Pair<Double, Long> result = new Pair<>();
        result.setFirst(distances.getDouble(argMin.getLong(0)));
        result.setSecond(argMin.getLong(0));
        return result;
    }
}
