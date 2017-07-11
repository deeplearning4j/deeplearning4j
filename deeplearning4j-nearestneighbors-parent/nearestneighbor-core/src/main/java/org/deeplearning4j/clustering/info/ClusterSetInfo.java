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

package org.deeplearning4j.clustering.info;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import org.deeplearning4j.clustering.cluster.Cluster;
import org.deeplearning4j.clustering.cluster.ClusterSet;

import java.io.Serializable;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

public class ClusterSetInfo implements Serializable {

    private Map<String, ClusterInfo> clustersInfos = new HashMap<>();
    private Table<String, String, Double> distancesBetweenClustersCenters = HashBasedTable.create();
    private AtomicInteger pointLocationChange;
    private boolean threadSafe;
    private boolean inverse;

    public ClusterSetInfo(boolean inverse) {
        this(inverse, false);
    }

    /**
     *
     * @param inverse
     * @param threadSafe
     */
    public ClusterSetInfo(boolean inverse, boolean threadSafe) {
        this.pointLocationChange = new AtomicInteger(0);
        this.threadSafe = threadSafe;
        this.inverse = inverse;
        if (threadSafe) {
            clustersInfos = Collections.synchronizedMap(clustersInfos);
        }
    }


    /**
     *
     * @param clusterSet
     * @param threadSafe
     * @return
     */
    public static ClusterSetInfo initialize(ClusterSet clusterSet, boolean threadSafe) {
        ClusterSetInfo info = new ClusterSetInfo(clusterSet.isInverse(), threadSafe);
        for (int i = 0, j = clusterSet.getClusterCount(); i < j; i++)
            info.addClusterInfo(clusterSet.getClusters().get(i).getId());
        return info;
    }

    public void removeClusterInfos(List<Cluster> clusters) {
        for (Cluster cluster : clusters) {
            clustersInfos.remove(cluster.getId());
        }
    }

    public ClusterInfo addClusterInfo(String clusterId) {
        ClusterInfo clusterInfo = new ClusterInfo(this.threadSafe);
        clustersInfos.put(clusterId, clusterInfo);
        return clusterInfo;
    }

    public ClusterInfo getClusterInfo(String clusterId) {
        return clustersInfos.get(clusterId);
    }

    public double getAveragePointDistanceFromClusterCenter() {
        if (clustersInfos == null || clustersInfos.isEmpty())
            return 0;

        double average = 0;
        for (ClusterInfo info : clustersInfos.values())
            average += info.getAveragePointDistanceFromCenter();
        return average / clustersInfos.size();
    }

    public double getPointDistanceFromClusterVariance() {
        if (clustersInfos == null || clustersInfos.isEmpty())
            return 0;

        double average = 0;
        for (ClusterInfo info : clustersInfos.values())
            average += info.getPointDistanceFromCenterVariance();
        return average / clustersInfos.size();
    }

    public int getPointsCount() {
        int count = 0;
        for (ClusterInfo clusterInfo : clustersInfos.values())
            count += clusterInfo.getPointDistancesFromCenter().size();
        return count;
    }

    public Map<String, ClusterInfo> getClustersInfos() {
        return clustersInfos;
    }

    public void setClustersInfos(Map<String, ClusterInfo> clustersInfos) {
        this.clustersInfos = clustersInfos;
    }

    public Table<String, String, Double> getDistancesBetweenClustersCenters() {
        return distancesBetweenClustersCenters;
    }

    public void setDistancesBetweenClustersCenters(Table<String, String, Double> interClusterDistances) {
        this.distancesBetweenClustersCenters = interClusterDistances;
    }

    public AtomicInteger getPointLocationChange() {
        return pointLocationChange;
    }

    public void setPointLocationChange(AtomicInteger pointLocationChange) {
        this.pointLocationChange = pointLocationChange;
    }

}
