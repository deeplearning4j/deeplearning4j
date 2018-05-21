package org.nd4j.linalg.api.shape.loop.coordinatefunction;

/**
 * Coordinate function for handling
 * 1 or more calls based on a set of coordinates
 */
public interface CoordinateFunction {

    void process(long[]... coord);

}
