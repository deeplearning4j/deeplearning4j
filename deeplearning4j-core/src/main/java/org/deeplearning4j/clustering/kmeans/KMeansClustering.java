package org.deeplearning4j.clustering.kmeans;

import org.deeplearning4j.clustering.algorithm.BaseClusteringAlgorithm;
import org.deeplearning4j.clustering.algorithm.strategy.ClusteringStrategy;
import org.deeplearning4j.clustering.algorithm.strategy.FixedClusterCountStrategy;
import org.nd4j.linalg.distancefunction.DistanceFunction;

/**
 * 
 * @author Julien Roch
 *
 */
public class KMeansClustering extends BaseClusteringAlgorithm {

	private static final long	serialVersionUID	= 8476951388145944776L;
	
	

	protected KMeansClustering(ClusteringStrategy clusteringStrategy) {
		super(clusteringStrategy);
	}



	public static KMeansClustering setup(int clusterCount, int maxIterationCount, Class<? extends DistanceFunction> distanceFunction) {
		ClusteringStrategy clusteringStrategy = FixedClusterCountStrategy.setup(clusterCount, distanceFunction);
		clusteringStrategy.endWhenIterationCountEquals(maxIterationCount);
		return new KMeansClustering(clusteringStrategy);
	}

	public static KMeansClustering setup(int clusterCount, double minDistributionVariationRate, Class<? extends DistanceFunction> distanceFunction, boolean allowEmptyClusters) {
		ClusteringStrategy clusteringStrategy = FixedClusterCountStrategy.setup(clusterCount, distanceFunction);
		clusteringStrategy.endWhenDistributionVariationRateLessThan(minDistributionVariationRate);
		return new KMeansClustering(clusteringStrategy);
	}
	

}
