package org.deeplearning4j.clustering.kmeans;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.deeplearning4j.clustering.ClusteringAlgorithm;
import org.deeplearning4j.clustering.cluster.ClusterSet;
import org.deeplearning4j.clustering.cluster.Point;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.distancefunction.DistanceFunction;
import org.nd4j.linalg.distancefunction.EuclideanDistance;
import org.nd4j.linalg.factory.Nd4j;
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

	private static final long					serialVersionUID	= 338231277453149972L;
	private static Logger						log					= LoggerFactory.getLogger(KMeansClustering.class);
	private static int							defaultIterationCount = 100;

	private Class<? extends DistanceFunction>	distanceFunction;
	private int									iterationCount;
	private transient ExecutorService			exec;
	
	private Integer								nbCluster;
	private ClusterSet							clusterSet;

	public KMeansClustering(Integer nbCluster, int iterationCount, Class<? extends DistanceFunction> distanceFunction) {
		this.nbCluster = nbCluster;
		this.iterationCount = iterationCount;
		this.distanceFunction = distanceFunction;
	}

	public KMeansClustering(Integer nbCluster) {
		this(nbCluster, defaultIterationCount, EuclideanDistance.class);
	}

	public ClusterSet applyTo(List<INDArray> vectors) {
		exec = Executors.newScheduledThreadPool(Runtime.getRuntime().availableProcessors());
		
		List<Point> points = Point.toPoints(vectors);
		initClusters(points);
		iterations(points);
		return clusterSet;
	}

	private void iterations(List<Point> points) {
		for( int i=0;i<iterationCount;i++) {
			clusterSet.removePoints();
			clusterSet.addPoints(points, true);
		}
	}

	
	protected void initClusters(List<Point> initialPointsList) {
		
		List<Point> points = new ArrayList<Point>(initialPointsList);
		
		clusterSet = new ClusterSet(distanceFunction);
		Random random = new Random();
		final Point center = points.remove(random.nextInt(points.size()));
		clusterSet.addNewClusterWithCenter(center);
		
		while( clusterSet.getClusterCount()<nbCluster ) {
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
