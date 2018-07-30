/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.clustering.cluster;

import lombok.Data;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.io.Serializable;
import java.util.*;

@Data
public class ClusterSet implements Serializable {

    private String distanceFunction;
    private List<Cluster> clusters;
    private Map<String, String> pointDistribution;
    private boolean inverse;

    public ClusterSet(boolean inverse) {
        this(null, inverse);
    }

    public ClusterSet(String distanceFunction, boolean inverse) {
        this.distanceFunction = distanceFunction;
        this.inverse = inverse;
        this.clusters = Collections.synchronizedList(new ArrayList<Cluster>());
        this.pointDistribution = Collections.synchronizedMap(new HashMap<String, String>());
    }


    public boolean isInverse() {
        return inverse;
    }

    /**
     *
     * @param center
     * @return
     */
    public Cluster addNewClusterWithCenter(Point center) {
        Cluster newCluster = new Cluster(center, distanceFunction);
        getClusters().add(newCluster);
        setPointLocation(center, newCluster);
        return newCluster;
    }

    /**
     *
     * @param point
     * @return
     */
    public PointClassification classifyPoint(Point point) {
        return classifyPoint(point, true);
    }

    /**
     *
     * @param points
     */
    public void classifyPoints(List<Point> points) {
        classifyPoints(points, true);
    }

    /**
     *
     * @param points
     * @param moveClusterCenter
     */
    public void classifyPoints(List<Point> points, boolean moveClusterCenter) {
        for (Point point : points)
            classifyPoint(point, moveClusterCenter);
    }

    /**
     *
     * @param point
     * @param moveClusterCenter
     * @return
     */
    public PointClassification classifyPoint(Point point, boolean moveClusterCenter) {
        Pair<Cluster, Double> nearestCluster = nearestCluster(point);
        Cluster newCluster = nearestCluster.getKey();
        boolean locationChange = isPointLocationChange(point, newCluster);
        addPointToCluster(point, newCluster, moveClusterCenter);
        return new PointClassification(nearestCluster.getKey(), nearestCluster.getValue(), locationChange);
    }

    private boolean isPointLocationChange(Point point, Cluster newCluster) {
        if (!getPointDistribution().containsKey(point.getId()))
            return true;
        return !getPointDistribution().get(point.getId()).equals(newCluster.getId());
    }

    private void addPointToCluster(Point point, Cluster cluster, boolean moveClusterCenter) {
        cluster.addPoint(point, moveClusterCenter);
        setPointLocation(point, cluster);
    }

    private void setPointLocation(Point point, Cluster cluster) {
        pointDistribution.put(point.getId(), cluster.getId());
    }


    /**
     *
     * @param point
     * @return
     */
    public Pair<Cluster, Double> nearestCluster(Point point) {

        Cluster nearestCluster = null;
        double minDistance = isInverse() ? Float.MIN_VALUE : Float.MAX_VALUE;

        double currentDistance;
        for (Cluster cluster : getClusters()) {
            currentDistance = cluster.getDistanceToCenter(point);
            if (isInverse()) {
                if (currentDistance > minDistance) {
                    minDistance = currentDistance;
                    nearestCluster = cluster;
                }
            } else {
                if (currentDistance < minDistance) {
                    minDistance = currentDistance;
                    nearestCluster = cluster;
                }
            }

        }

        return Pair.of(nearestCluster, minDistance);

    }

    /**
     *
     * @param m1
     * @param m2
     * @return
     */
    public double getDistance(Point m1, Point m2) {
        return Nd4j.getExecutioner()
                        .execAndReturn(Nd4j.getOpFactory().createAccum(distanceFunction, m1.getArray(), m2.getArray()))
                        .getFinalResult().doubleValue();
    }

    /**
     *
     * @param point
     * @return
     */
    public double getDistanceFromNearestCluster(Point point) {
        return nearestCluster(point).getValue();
    }


    /**
     *
     * @param clusterId
     * @return
     */
    public String getClusterCenterId(String clusterId) {
        Point clusterCenter = getClusterCenter(clusterId);
        return clusterCenter == null ? null : clusterCenter.getId();
    }

    /**
     *
     * @param clusterId
     * @return
     */
    public Point getClusterCenter(String clusterId) {
        Cluster cluster = getCluster(clusterId);
        return cluster == null ? null : cluster.getCenter();
    }

    /**
     *
     * @param id
     * @return
     */
    public Cluster getCluster(String id) {
        for (int i = 0, j = clusters.size(); i < j; i++)
            if (id.equals(clusters.get(i).getId()))
                return clusters.get(i);
        return null;
    }

    /**
     *
     * @return
     */
    public int getClusterCount() {
        return getClusters() == null ? 0 : getClusters().size();
    }

    /**
     *
     */
    public void removePoints() {
        for (Cluster cluster : getClusters())
            cluster.removePoints();
    }

    /**
     *
     * @param count
     * @return
     */
    public List<Cluster> getMostPopulatedClusters(int count) {
        List<Cluster> mostPopulated = new ArrayList<>(clusters);
        Collections.sort(mostPopulated, new Comparator<Cluster>() {
            public int compare(Cluster o1, Cluster o2) {
                return new Integer(o1.getPoints().size()).compareTo(new Integer(o2.getPoints().size()));
            }
        });
        return mostPopulated.subList(0, count);
    }

    /**
     *
     * @return
     */
    public List<Cluster> removeEmptyClusters() {
        List<Cluster> emptyClusters = new ArrayList<>();
        for (Cluster cluster : clusters)
            if (cluster.isEmpty())
                emptyClusters.add(cluster);
        clusters.removeAll(emptyClusters);
        return emptyClusters;
    }

}
