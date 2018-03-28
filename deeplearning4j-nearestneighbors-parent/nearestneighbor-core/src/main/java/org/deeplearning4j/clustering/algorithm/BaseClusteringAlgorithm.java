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

package org.deeplearning4j.clustering.algorithm;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.ArrayUtils;
import org.deeplearning4j.clustering.cluster.Cluster;
import org.deeplearning4j.clustering.cluster.ClusterSet;
import org.deeplearning4j.clustering.cluster.ClusterUtils;
import org.deeplearning4j.clustering.cluster.Point;
import org.deeplearning4j.clustering.info.ClusterSetInfo;
import org.deeplearning4j.clustering.iteration.IterationHistory;
import org.deeplearning4j.clustering.iteration.IterationInfo;
import org.deeplearning4j.clustering.strategy.ClusteringStrategy;
import org.deeplearning4j.clustering.strategy.ClusteringStrategyType;
import org.deeplearning4j.clustering.strategy.OptimisationStrategy;
import org.deeplearning4j.clustering.util.MultiThreadUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;

/**
 *
 * adapted to ndarray matrices
 *
 * @author Adam Gibson
 * @author Julien Roch
 *
 */
@Slf4j
@NoArgsConstructor(access = AccessLevel.PROTECTED)
public class BaseClusteringAlgorithm implements ClusteringAlgorithm, Serializable {

    private static final long serialVersionUID = 338231277453149972L;

    private ClusteringStrategy clusteringStrategy;
    private IterationHistory iterationHistory;
    private int currentIteration = 0;
    private ClusterSet clusterSet;
    private List<Point> initialPoints;
    private transient ExecutorService exec;



    protected BaseClusteringAlgorithm(ClusteringStrategy clusteringStrategy) {
        this.clusteringStrategy = clusteringStrategy;
        this.exec = MultiThreadUtils.newExecutorService();
    }

    /**
     *
     * @param clusteringStrategy
     * @return
     */
    public static BaseClusteringAlgorithm setup(ClusteringStrategy clusteringStrategy) {
        return new BaseClusteringAlgorithm(clusteringStrategy);
    }

    /**
     *
     * @param points
     * @return
     */
    public ClusterSet applyTo(List<Point> points) {
        resetState(points);
        initClusters();
        iterations();
        return clusterSet;
    }

    private void resetState(List<Point> points) {
        this.iterationHistory = new IterationHistory();
        this.currentIteration = 0;
        this.clusterSet = null;
        this.initialPoints = points;
    }

    /** Run clustering iterations until a
     * termination condition is hit.
     * This is done by first classifying all points,
     * and then updating cluster centers based on
     * those classified points
     */
    private void iterations() {
        int iterationCount = 0;
        while ((clusteringStrategy.getTerminationCondition() != null
                        && !clusteringStrategy.getTerminationCondition().isSatisfied(iterationHistory))
                        || iterationHistory.getMostRecentIterationInfo().isStrategyApplied()) {
            currentIteration++;
            removePoints();
            classifyPoints();
            applyClusteringStrategy();
            log.info("Completed clustering iteration {}", ++iterationCount);
        }
    }

    protected void classifyPoints() {
        //Classify points. This also adds each point to the ClusterSet
        ClusterSetInfo clusterSetInfo = ClusterUtils.classifyPoints(clusterSet, initialPoints, exec);
        //Update the cluster centers, based on the points within each cluster
        ClusterUtils.refreshClustersCenters(clusterSet, clusterSetInfo, exec);
        iterationHistory.getIterationsInfos().put(currentIteration,
                        new IterationInfo(currentIteration, clusterSetInfo));
    }

    /**
     * Initialize the
     * cluster centers at random
     */
    protected void initClusters() {
        log.info("Generating initial clusters");
        List<Point> points = new ArrayList<>(initialPoints);

        //Initialize the ClusterSet with a single cluster center (based on position of one of the points chosen randomly)
        Random random = new Random();
        clusterSet = new ClusterSet(clusteringStrategy.getDistanceFunction(),
                        clusteringStrategy.inverseDistanceCalculation());
        clusterSet.addNewClusterWithCenter(points.remove(random.nextInt(points.size())));
        int initialClusterCount = clusteringStrategy.getInitialClusterCount();

        //dxs: distances between
        // each point and nearest cluster to that point
        INDArray dxs = Nd4j.create(points.size());
        dxs.addi(clusteringStrategy.inverseDistanceCalculation() ? -Double.MAX_VALUE : Double.MAX_VALUE);

        //Generate the initial cluster centers, by randomly selecting a point between 0 and max distance
        //Thus, we are more likely to select (as a new cluster center) a point that is far from an existing cluster
        while (clusterSet.getClusterCount() < initialClusterCount && !points.isEmpty()) {
            dxs = ClusterUtils.computeSquareDistancesFromNearestCluster(clusterSet, points, dxs, exec);
            double r = random.nextFloat() * dxs.maxNumber().doubleValue();
            for (int i = 0; i < dxs.length(); i++) {
                if (dxs.getDouble(i) >= r) {
                    clusterSet.addNewClusterWithCenter(points.remove(i));
                    dxs = Nd4j.create(ArrayUtils.remove(dxs.data().asDouble(), i));
                    break;
                }
            }
        }

        ClusterSetInfo initialClusterSetInfo = ClusterUtils.computeClusterSetInfo(clusterSet);
        iterationHistory.getIterationsInfos().put(currentIteration,
                        new IterationInfo(currentIteration, initialClusterSetInfo));
    }

    protected void applyClusteringStrategy() {
        if (!isStrategyApplicableNow())
            return;

        ClusterSetInfo clusterSetInfo = iterationHistory.getMostRecentClusterSetInfo();
        if (!clusteringStrategy.isAllowEmptyClusters()) {
            int removedCount = removeEmptyClusters(clusterSetInfo);
            if (removedCount > 0) {
                iterationHistory.getMostRecentIterationInfo().setStrategyApplied(true);

                if (clusteringStrategy.isStrategyOfType(ClusteringStrategyType.FIXED_CLUSTER_COUNT)
                                && clusterSet.getClusterCount() < clusteringStrategy.getInitialClusterCount()) {
                    int splitCount = ClusterUtils.splitMostSpreadOutClusters(clusterSet, clusterSetInfo,
                                    clusteringStrategy.getInitialClusterCount() - clusterSet.getClusterCount(), exec);
                    if (splitCount > 0)
                        iterationHistory.getMostRecentIterationInfo().setStrategyApplied(true);
                }
            }
        }
        if (clusteringStrategy.isStrategyOfType(ClusteringStrategyType.OPTIMIZATION))
            optimize();
    }

    protected void optimize() {
        ClusterSetInfo clusterSetInfo = iterationHistory.getMostRecentClusterSetInfo();
        OptimisationStrategy optimization = (OptimisationStrategy) clusteringStrategy;
        boolean applied = ClusterUtils.applyOptimization(optimization, clusterSet, clusterSetInfo, exec);
        iterationHistory.getMostRecentIterationInfo().setStrategyApplied(applied);
    }

    private boolean isStrategyApplicableNow() {
        return clusteringStrategy.isOptimizationDefined() && iterationHistory.getIterationCount() != 0
                        && clusteringStrategy.isOptimizationApplicableNow(iterationHistory);
    }

    protected int removeEmptyClusters(ClusterSetInfo clusterSetInfo) {
        List<Cluster> removedClusters = clusterSet.removeEmptyClusters();
        clusterSetInfo.removeClusterInfos(removedClusters);
        return removedClusters.size();
    }

    protected void removePoints() {
        clusterSet.removePoints();
    }

}
