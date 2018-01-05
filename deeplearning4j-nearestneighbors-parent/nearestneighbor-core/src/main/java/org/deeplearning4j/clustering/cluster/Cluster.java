/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.clustering.cluster;

import lombok.Data;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.UUID;

/**
 * A cluster.
 *
 *
 */
@Data
public class Cluster implements Serializable {

    private String id = UUID.randomUUID().toString();
    private String label;

    private Point center;
    private List<Point> points = Collections.synchronizedList(new ArrayList<Point>());
    private boolean inverse = false;
    private String distanceFunction;

    public Cluster() {
        super();
    }

    /**
     *
     * @param center
     * @param distanceFunction
     */
    public Cluster(Point center, String distanceFunction) {
        this(center, false, distanceFunction);
    }

    /**
     *
     * @param center
     * @param distanceFunction
     */
    public Cluster(Point center, boolean inverse, String distanceFunction) {
        this.distanceFunction = distanceFunction;
        this.inverse = inverse;
        setCenter(center);
    }


    /**
     * Get the distance to the given
     * point from the cluster
     * @param point the point to get the distance for
     * @return
     */
    public double getDistanceToCenter(Point point) {
        return Nd4j.getExecutioner().execAndReturn(
                        Nd4j.getOpFactory().createAccum(distanceFunction, center.getArray(), point.getArray()))
                        .getFinalResult().doubleValue();
    }

    /**
     * Add a point to the cluster
     * @param point
     */
    public void addPoint(Point point) {
        addPoint(point, true);
    }

    /**
     * Add a point to the cluster
     * @param point the point to add
     * @param moveClusterCenter whether to update
     *                          the cluster centroid or not
     */
    public void addPoint(Point point, boolean moveClusterCenter) {
        if (moveClusterCenter) {
            if (isInverse()) {
                center.getArray().muli(points.size()).subi(point.getArray()).divi(points.size() + 1);
            } else {
                center.getArray().muli(points.size()).addi(point.getArray()).divi(points.size() + 1);
            }
        }

        getPoints().add(point);
    }

    /**
     * Clear out the ponits
     */
    public void removePoints() {
        if (getPoints() != null)
            getPoints().clear();
    }

    /**
     * Whether the cluster is empty or not
     * @return
     */
    public boolean isEmpty() {
        return points == null || points.isEmpty();
    }

    /**
     * Return the point with the given id
     * @param id
     * @return
     */
    public Point getPoint(String id) {
        for (Point point : points)
            if (id.equals(point.getId()))
                return point;
        return null;
    }

    /**
     * Remove the point and return it
     * @param id
     * @return
     */
    public Point removePoint(String id) {
        Point removePoint = null;
        for (Point point : points)
            if (id.equals(point.getId()))
                removePoint = point;
        if (removePoint != null)
            points.remove(removePoint);
        return removePoint;
    }


}
