package org.deeplearning4j.clustering.kmeans;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.deeplearning4j.clustering.algorithm.ClusteringAlgorithm;
import org.deeplearning4j.clustering.algorithm.condition.ClusteringEndCondition;
import org.deeplearning4j.clustering.algorithm.iteration.IterationHistory;
import org.deeplearning4j.clustering.algorithm.iteration.IterationInfo;
import org.deeplearning4j.clustering.cluster.ClusterSet;
import org.deeplearning4j.clustering.cluster.ClusterUtils;
import org.deeplearning4j.clustering.cluster.Point;
import org.deeplearning4j.clustering.cluster.info.ClusterSetInfo;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.distancefunction.DistanceFunction;
import org.nd4j.linalg.distancefunction.EuclideanDistance;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.ConditionBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * adapted to ndarray matrices
 * 
 * @author Adam Gibson
 *
 */
public class KMeansClustering implements ClusteringAlgorithm, Serializable {

	private static final long					serialVersionUID		= 338231277453149972L;
	private static Logger						log						= LoggerFactory.getLogger(KMeansClustering.class);
	private static int							defaultIterationCount	= 100;

	private Class<? extends DistanceFunction>	distanceFunction;
	private ClusteringEndCondition				endCondition;
	private IterationHistory					iterationHistory;
	
	private Integer								nbCluster;
	private ClusterSet							clusterSet;
	
	private transient ExecutorService			exec;

	public KMeansClustering(Integer nbCluster) {
		this(nbCluster, EuclideanDistance.class);
	}

	public KMeansClustering(Integer nbCluster, Class<? extends DistanceFunction> distanceFunction) {
		this(nbCluster, distanceFunction, new ClusteringEndCondition().iterationCountLessThan(defaultIterationCount));
	}

	public KMeansClustering(Integer nbCluster, Class<? extends DistanceFunction> distanceFunction, ClusteringEndCondition endCondition) {
		this.nbCluster = nbCluster;
		this.endCondition = endCondition;
		this.distanceFunction = distanceFunction;
		this.iterationHistory = new IterationHistory();
	}

	public ClusterSet applyTo(List<Point> points) {
		exec = Executors.newScheduledThreadPool(Runtime.getRuntime().availableProcessors());

		initClusters(points);
		iterations(points);
		return clusterSet;
	}

	private void iterations(List<Point> points) {
		int iterationCount = 0;
		while( !endCondition.isSatisfied(iterationHistory) ) {
			clusterSet.removePoints();
			clusterSet.addPoints(points, true);
			
			ClusterSetInfo clusterSetInfo = ClusterUtils.computeClusterSetInfo(clusterSet);
			iterationHistory.getIterationsInfos().put(iterationCount, new IterationInfo(iterationCount, clusterSetInfo));
		}
	}

	protected void initClusters(List<Point> initialPointsList) {

		List<Point> points = new ArrayList<Point>(initialPointsList);

		clusterSet = new ClusterSet(distanceFunction);
		Random random = new Random();
		final Point center = points.remove(random.nextInt(points.size()));
		clusterSet.addNewClusterWithCenter(center);

		while (clusterSet.getClusterCount() < nbCluster) {
			INDArray dxs = computeDxs(points);
			double r = random.nextFloat() * dxs.getDouble(dxs.length() - 1);
			for (int i = 0; i < dxs.length(); i++) {
				if (dxs.getDouble(i) >= r) {
					clusterSet.addNewClusterWithCenter(points.remove(i));
					break;
				}
			}
		}

	}

	protected INDArray computeDxs(final List<Point> points) {
		final INDArray dxs = Nd4j.create(points.size(), points.get(0).columns());

		final CountDownLatch latch = new CountDownLatch(points.size());
		for (int i = 0; i < points.size(); i++) {
			final int i2 = i;
			exec.execute(new Runnable() {
				@Override
				public void run() {
					Point point = points.get(i2);
					dxs.putScalar(i2, (int) Math.pow(clusterSet.getDistanceFromNearestCluster(point), 2));
					latch.countDown();
				}
			});

		}

		try {
			latch.await();
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
		}

		return dxs;
	}

}
