package org.nd4j.linalg.indexing;

/**
 * @author Adam Gibson
 */
public class IndexInfo {
    private INDArrayIndex[] indexes;
    private boolean[] point;
    private boolean[] newAxis;
    private int numNewAxes = 0;
    private int numPoints = 0;

    public IndexInfo(INDArrayIndex... indexes) {
        this.indexes = indexes;
        for (int i = 0; i < indexes.length; i++) {
            if (indexes[i] instanceof PointIndex)
                numPoints++;
            if (indexes[i] instanceof IntervalIndex) {

            }
            if (indexes[i] instanceof NewAxis)
                numNewAxes++;
        }

    }

    public int getNumNewAxes() {
        return numNewAxes;
    }

    public int getNumPoints() {
        return numPoints;
    }
}
