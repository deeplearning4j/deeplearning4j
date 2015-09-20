package org.nd4j.linalg.api.shape.loop.coordinatefunction;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by agibsonccc on 9/15/15.
 */
public class CopyCoordinateFunction implements CoordinateFunction {
    private INDArray from,to;

    public CopyCoordinateFunction(INDArray from, INDArray to) {
        this.from = from;
        this.to = to;
    }

    @Override
    public void process(int[]... coord) {
        to.putScalar(coord[1],from.getDouble(coord[1]));
    }
}
