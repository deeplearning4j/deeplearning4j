/*
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

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import org.apache.commons.lang3.ArrayUtils;
import org.deeplearning4j.clustering.algorithm.optimisation.ClusteringOptimizationType;
import org.deeplearning4j.clustering.algorithm.strategy.OptimisationStrategy;
import org.deeplearning4j.clustering.cluster.info.ClusterInfo;
import org.deeplearning4j.clustering.cluster.info.ClusterSetInfo;
import org.deeplearning4j.util.MathUtils;
import org.deeplearning4j.util.MultiThreadUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * Basic cluster utilities
 */
public class ClusterUtils {

	/** Classify the set of points base on cluster centers. This also adds each point to the ClusterSet */
	public static ClusterSetInfo classifyPoints(final ClusterSet clusterSet, List<Point> points,
												ExecutorService executorService) {
		final ClusterSetInfo clusterSetInfo = ClusterSetInfo.initialize(clusterSet, true);

		List<Runnable> tasks = new ArrayList<>();
		for (final Point point : points) {
			tasks.add(new Runnable() {
				public void run() {
					try {
						PointClassification result = classifyPoint(clusterSet, point);
						if (result.isNewLocation())
							clusterSetInfo.getPointLocationChange().incrementAndGet();
						clusterSetInfo.getClusterInfo(result.getCluster().getId()).getPointDistancesFromCenter()
								.put(point.getId(), result.getDistanceFromCenter());
					} catch (Exception e) {
						e.printStackTrace();
					}
				}
			});
		}
		MultiThreadUtils.parallelTasks(tasks, executorService);
		return clusterSetInfo;
	}

	public static PointClassification classifyPoint(ClusterSet clusterSet, Point point) {
		return clusterSet.classifyPoint(point, false);
	}

	public static void refreshClustersCenters(final ClusterSet clusterSet, final ClusterSetInfo clusterSetInfo,
											  ExecutorService executorService) {
		List<Runnable> tasks = new ArrayList<>();
		int nClusters = clusterSet.getClusterCount();
		for (int i = 0; i < nClusters; i++) {
			final Cluster cluster = clusterSet.getClusters().get(i);
			tasks.add(new Runnable() {
				public void run() {
					try {
						final ClusterInfo clusterInfo = clusterSetInfo.getClusterInfo(cluster.getId());
						refreshClusterCenter(cluster, clusterInfo);
						deriveClusterInfoDistanceStatistics(clusterInfo);
					} catch (Exception e) {

						e.printStackTrace();
					}
				}
			});
		}
		MultiThreadUtils.parallelTasks(tasks, executorService);
	}

	public static void refreshClusterCenter(Cluster cluster, ClusterInfo clusterInfo) {
		int pointsCount = cluster.getPoints().size();
		if (pointsCount == 0)
			return;
		Point center = new Point(Nd4j.create(cluster.getPoints().get(0).getArray().length()));
		for (Point point : cluster.getPoints()) {
			INDArray arr = point.getArray();
			center.getArray().addi(arr);
		}
		center.getArray().divi(pointsCount);
		cluster.setCenter(center);
	}

	public static void deriveClusterInfoDistanceStatistics(ClusterInfo info) {
		int pointCount = info.getPointDistancesFromCenter().size();
		if (pointCount==0)
			return;

		double[] distances = ArrayUtils.toPrimitive(info.getPointDistancesFromCenter().values().toArray(new Double[] {}));
		double max = MathUtils.max(distances);
		double total = MathUtils.sum(distances);

		info.setMaxPointDistanceFromCenter(max);
		info.setTotalPointDistanceFromCenter(total);
		info.setAveragePointDistanceFromCenter(total/pointCount);
		info.setPointDistanceFromCenterVariance(MathUtils.variance(distances));
	}

	public static INDArray computeSquareDistancesFromNearestCluster(final ClusterSet clusterSet, final List<Point> points,
																	INDArray previousDxs, ExecutorService executorService) {
		final int pointsCount = points.size();
		final INDArray dxs = Nd4j.create(pointsCount);
		final Cluster newCluster = clusterSet.getClusters().get(clusterSet.getClusters().size()-1);

		List<Runnable> tasks = new ArrayList<>();
		for (int i = 0; i < pointsCount; i++) {
			final int i2 = i;
			tasks.add(new Runnable() {
				public void run() {
					Point point = points.get(i2);
					dxs.putScalar(i2, Math.pow(newCluster.getDistanceToCenter(point), 2));
				}
			});

		}
		MultiThreadUtils.parallelTasks(tasks, executorService);

		for(int i=0;i<pointsCount;i++) {
			double previousMinDistance = previousDxs.getDouble(i);
			if( dxs.getDouble(i)>previousMinDistance )
				dxs.putScalar(i, previousMinDistance);
		}

		return dxs;
	}

	public static ClusterSetInfo computeClusterSetInfo(ClusterSet clusterSet) {
		ExecutorService executor = MultiThreadUtils.newExecutorService();
		ClusterSetInfo info = computeClusterSetInfo(clusterSet, executor);
		executor.shutdownNow();
		return info;
	}

	public static ClusterSetInfo computeClusterSetInfo(final ClusterSet clusterSet, ExecutorService executorService) {
		final ClusterSetInfo info = new ClusterSetInfo(true);
		int clusterCount = clusterSet.getClusterCount();

		List<Runnable> tasks = new ArrayList<>();
		for (int i = 0; i < clusterCount; i++) {
			final Cluster cluster = clusterSet.getClusters().get(i);
			tasks.add(new Runnable() {
				public void run() {
					info.getClustersInfos().put(cluster.getId(), computeClusterInfos(cluster, clusterSet.getAccumulation()));
				}
			});
		}
		MultiThreadUtils.parallelTasks(tasks, executorService);

		tasks = new ArrayList<>();
		for (int i = 0; i < clusterCount; i++) {
			final int clusterIdx = i;
			final Cluster fromCluster = clusterSet.getClusters().get(i);
			tasks.add(new Runnable() {
				public void run() {
					try {
						for (int k = clusterIdx + 1, l = clusterSet.getClusterCount(); k < l; k++) {
							Cluster toCluster = clusterSet.getClusters().get(k);
							double distance =
									Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createAccum(clusterSet.getAccumulation(),
											fromCluster.getCenter().getArray(),toCluster.getCenter().getArray())).currentResult().doubleValue();
							info.getDistancesBetweenClustersCenters().put(fromCluster.getId(), toCluster.getId(), distance);
						}
					} catch (Exception e) {

						e.printStackTrace();
					}
				}
			});

		}
		MultiThreadUtils.parallelTasks(tasks, executorService);

		return info;
	}

	public static ClusterInfo computeClusterInfos(Cluster cluster,String distanceFunction) {
		ClusterInfo info = new ClusterInfo(true);
		for (int i = 0, j = cluster.getPoints().size(); i < j; i++) {
			Point point = cluster.getPoints().get(i);
			double distance =
					Nd4j.getExecutioner().execAndReturn(
							Nd4j.getOpFactory().createAccum(distanceFunction,cluster.getCenter().getArray(),
							point.getArray())).currentResult().doubleValue();
			info.getPointDistancesFromCenter().put(point.getId(), distance);
			info.setTotalPointDistanceFromCenter(info.getTotalPointDistanceFromCenter() + distance);
		}
		if (cluster.getPoints().size() > 0)
			info.setAveragePointDistanceFromCenter(info.getTotalPointDistanceFromCenter()/ cluster.getPoints().size());
		return info;
	}

	public static boolean applyOptimization(OptimisationStrategy optimization, ClusterSet clusterSet,
											ClusterSetInfo clusterSetInfo, ExecutorService executor) {

		if (optimization.isClusteringOptimizationType(
				ClusteringOptimizationType.MINIMIZE_AVERAGE_POINT_TO_CENTER_DISTANCE)) {
			int splitCount =
					ClusterUtils.splitClustersWhereAverageDistanceFromCenterGreaterThan(clusterSet, clusterSetInfo,
							optimization.getClusteringOptimizationValue(), executor);
			return splitCount>0;
		}

		if (optimization.isClusteringOptimizationType(
				ClusteringOptimizationType.MINIMIZE_MAXIMUM_POINT_TO_CENTER_DISTANCE)) {
			int splitCount =
					ClusterUtils.splitClustersWhereMaximumDistanceFromCenterGreaterThan(clusterSet, clusterSetInfo,
							optimization.getClusteringOptimizationValue(), executor);
			return splitCount > 0;
		}

		return false;
	}

	public static List<Cluster> getMostSpreadOutClusters(
			final ClusterSet clusterSet, final ClusterSetInfo info, int count) {
		List<Cluster> clusters = new ArrayList<>(clusterSet.getClusters());
		Collections.sort(clusters, new Comparator<Cluster>() {
			public int compare(Cluster o1, Cluster o2) {
				Double o1TotalDistance = info.getClusterInfo(o1.getId()).getTotalPointDistanceFromCenter();
				Double o2TotalDistance = info.getClusterInfo(o2.getId()).getTotalPointDistanceFromCenter();
				return -o1TotalDistance.compareTo(o2TotalDistance);
			}
		});
		return clusters.subList(0, count);
	}

	public static List<Cluster> getClustersWhereAverageDistanceFromCenterGreaterThan(final ClusterSet clusterSet,
																					 final ClusterSetInfo info,
																					 double maximumAverageDistance) {
		List<Cluster> clusters = new ArrayList<>();
		for (Cluster cluster : clusterSet.getClusters()) {
			ClusterInfo clusterInfo = info.getClusterInfo(cluster.getId());
			if (clusterInfo != null && clusterInfo.getAveragePointDistanceFromCenter() > maximumAverageDistance)
				clusters.add(cluster);
		}
		return clusters;
	}

	public static List<Cluster> getClustersWhereMaximumDistanceFromCenterGreaterThan(final ClusterSet clusterSet, final ClusterSetInfo info, double maximumDistance) {
		List<Cluster> clusters = new ArrayList<>();
		for (Cluster cluster : clusterSet.getClusters()) {
			ClusterInfo clusterInfo = info.getClusterInfo(cluster.getId());
			if (clusterInfo != null && clusterInfo.getMaxPointDistanceFromCenter()>maximumDistance )
				clusters.add(cluster);
		}
		return clusters;
	}

	public static int splitMostSpreadOutClusters(ClusterSet clusterSet, ClusterSetInfo clusterSetInfo,
												 int count, ExecutorService executorService) {
		List<Cluster> clustersToSplit = getMostSpreadOutClusters(clusterSet, clusterSetInfo, count);
		splitClusters(clusterSet, clusterSetInfo, clustersToSplit, executorService);
		return clustersToSplit.size();
	}

	public static int splitClustersWhereAverageDistanceFromCenterGreaterThan(ClusterSet clusterSet,
																			 ClusterSetInfo clusterSetInfo,
																			 double maxWithinClusterDistance,
																			 ExecutorService executorService) {
		List<Cluster> clustersToSplit =
				getClustersWhereAverageDistanceFromCenterGreaterThan(clusterSet, clusterSetInfo, maxWithinClusterDistance);
		splitClusters(clusterSet, clusterSetInfo, clustersToSplit, maxWithinClusterDistance, executorService);
		return clustersToSplit.size();
	}

	public static int splitClustersWhereMaximumDistanceFromCenterGreaterThan(ClusterSet clusterSet,
																			 ClusterSetInfo clusterSetInfo,
																			 double maxWithinClusterDistance,
																			 ExecutorService executorService) {
		List<Cluster> clustersToSplit =
				getClustersWhereMaximumDistanceFromCenterGreaterThan(clusterSet, clusterSetInfo, maxWithinClusterDistance);
		splitClusters(clusterSet, clusterSetInfo, clustersToSplit, maxWithinClusterDistance, executorService);
		return clustersToSplit.size();
	}

	public static void splitMostPopulatedClusters(ClusterSet clusterSet, ClusterSetInfo clusterSetInfo,
												  int count, ExecutorService executorService) {
		List<Cluster> clustersToSplit = clusterSet.getMostPopulatedClusters(count);
		splitClusters(clusterSet, clusterSetInfo, clustersToSplit, executorService);
	}

	public static void splitClusters(final ClusterSet clusterSet, final ClusterSetInfo clusterSetInfo,
									 List<Cluster> clusters, final double maxDistance,
									 ExecutorService executorService) {
		final Random random = new Random();
		List<Runnable> tasks = new ArrayList<>();
		for (final Cluster cluster : clusters) {
			tasks.add(new Runnable() {
				public void run() {
					try {
						ClusterInfo clusterInfo = clusterSetInfo.getClusterInfo(cluster.getId());
						List<String> fartherPoints = clusterInfo.getPointsFartherFromCenterThan(maxDistance);
						int rank = Math.min(fartherPoints.size(), 3);
						String pointId = fartherPoints.get(random.nextInt(rank));
						Point point = cluster.removePoint(pointId);
						clusterSet.addNewClusterWithCenter(point);
					} catch (Exception e) {

						e.printStackTrace();
					}
				}
			});
		}
		MultiThreadUtils.parallelTasks(tasks, executorService);
	}

	public static void splitClusters(final ClusterSet clusterSet, final ClusterSetInfo clusterSetInfo,
									 List<Cluster> clusters, ExecutorService executorService) {
		final Random random = new Random();
		List<Runnable> tasks = new ArrayList<>();
		for (final Cluster cluster : clusters) {
			tasks.add(new Runnable() {
				public void run() {
					Point point = cluster.getPoints().remove(random.nextInt(cluster.getPoints().size()));
					clusterSet.addNewClusterWithCenter(point);
				}
			});
		}
		MultiThreadUtils.parallelTasks(tasks, executorService);
	}
}
