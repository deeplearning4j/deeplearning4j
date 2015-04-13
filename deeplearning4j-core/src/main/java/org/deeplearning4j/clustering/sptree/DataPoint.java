package org.deeplearning4j.clustering.sptree;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.distances.EuclideanDistance;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;

/**
 * @author Adam Gibson
 */
public class DataPoint implements Serializable {
    private int index;
    private INDArray point;
    private int d;
    public DataPoint(int index, INDArray point) {
        this.index = index;
        this.point = point;
        this.d = point.length();
    }

    /**
     * Euclidean distance
     * @param point the distance from this point to the given point
     * @return the euclidean distance between the two points
     */
    public double euclidean(DataPoint point) {
        return Nd4j.getExecutioner().execAndReturn(new EuclideanDistance(this.point,point.point)).currentResult().doubleValue();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        DataPoint dataPoint = (DataPoint) o;

        if (index != dataPoint.index) return false;
        return !(point != null ? !point.equals(dataPoint.point) : dataPoint.point != null);

    }

    @Override
    public int hashCode() {
        int result = index;
        result = 31 * result + (point != null ? point.hashCode() : 0);
        return result;
    }

    public int getD() {
        return d;
    }

    public void setD(int d) {
        this.d = d;
    }

    public int getIndex() {
        return index;
    }

    public void setIndex(int index) {
        this.index = index;
    }

    public INDArray getPoint() {
        return point;
    }

    public void setPoint(INDArray point) {
        this.point = point;
    }
}
