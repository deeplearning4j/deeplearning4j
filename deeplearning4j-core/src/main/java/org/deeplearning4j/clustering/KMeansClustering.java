package org.deeplearning4j.clustering;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;



import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.distancefunction.DistanceFunction;
import org.deeplearning4j.linalg.distancefunction.EuclideanDistance;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * Shamelessly based on:
 * https://github.com/pmerienne/trident-ml/blob/master/src/main/java/com/github/pmerienne/trident/ml/clustering/KMeans.java
 * 
 * adapted to jblas double matrices
 * @author Adam Gibson
 *
 */
public class KMeansClustering implements Serializable {


	private static final long serialVersionUID = 338231277453149972L;
	private static Logger log = LoggerFactory.getLogger(KMeansClustering.class);

	private List<Long> counts = null;
	private INDArray centroids;
	private List<INDArray> initFeatures = new ArrayList<INDArray>();
	private Class<DistanceFunction> clazz;

	private Integer nbCluster;

	public KMeansClustering(Integer nbCluster,Class<? extends DistanceFunction> clazz) {
		this.nbCluster = nbCluster;
	}


	public KMeansClustering(Integer nbCluster) {
		this(nbCluster,EuclideanDistance.class);
	}


	public Integer classify(INDArray features) {
		if (!this.isReady()) {
			throw new IllegalStateException("KMeans is not ready yet");
		}

		// Find nearest centroid
		Integer nearestCentroidIndex = this.nearestCentroid(features);
		return nearestCentroidIndex;
	}


	public Integer update(INDArray features) {
		if (!this.isReady()) {
			this.initIfPossible(features);
			log.info("Initializing feature vector with length of " + features.length());
			return null;
		} else {
			Integer nearestCentroid = this.classify(features);

			// Increment count
			this.counts.set(nearestCentroid, this.counts.get(nearestCentroid) + 1);

			// Move centroid
			INDArray update = features.sub(this.centroids.getRow(nearestCentroid)).mul( 1.0 / this.counts.get(nearestCentroid));
			this.centroids.putRow(nearestCentroid,this.centroids.getRow(nearestCentroid).add(update));

			return nearestCentroid;
		}
	}


	public INDArray distribution(INDArray features) {
		if (!this.isReady()) {
			throw new IllegalStateException("KMeans is not ready yet");
		}

		INDArray distribution = NDArrays.create(1,this.nbCluster);
		INDArray currentCentroid;
		for (int i = 0; i < this.nbCluster; i++) {
			currentCentroid = this.centroids.getRow(i);
			distribution.putScalar(i,getDistance(currentCentroid,features));
		}

		return distribution;
	}


	private double getDistance(INDArray m1,INDArray m2) {
		DistanceFunction function;
		try {
			function = clazz.getConstructor(INDArray.class).newInstance(m1);
		} catch (Exception e) {
			throw new RuntimeException(e);
		} 
		return function.apply(m2);
	}

	public INDArray getCentroids() {
		return this.centroids;
	}

	protected Integer nearestCentroid(INDArray features) {
		// Find nearest centroid
		Integer nearestCentroidIndex = 0;

		Double minDistance = Double.MAX_VALUE;
		INDArray currentCentroid;
		Double currentDistance;
		for (int i = 0; i < this.centroids.rows(); i++) {
			currentCentroid = this.centroids.getRow(i);
			if (currentCentroid != null) {
				currentDistance = getDistance(currentCentroid,features);
				if (currentDistance < minDistance) {
					minDistance = currentDistance;
					nearestCentroidIndex = i;
				}
			}
		}

		return nearestCentroidIndex;
	}

	protected boolean isReady() {
		boolean countsReady = this.counts != null;
		boolean centroidsReady = this.centroids != null;
		return countsReady && centroidsReady;
	}

	protected void initIfPossible(INDArray features) {
		this.initFeatures.add(features);
		log.info("Added feature vector of length " + features.length());
		// magic number : 10 ??!
		if (this.initFeatures.size() >= 10 * this.nbCluster) {
			this.initCentroids();
		}
	}

	/**
	 * Init clusters using the k-means++ algorithm. (Arthur, D. and
	 * Vassilvitskii, S. (2007). "k-means++: the advantages of careful seeding".
	 * 
	 */
	protected void initCentroids() {
		// Init counts
		this.counts = new ArrayList<>(this.nbCluster);
		for (int i = 0; i < this.nbCluster; i++) {
			this.counts.add(0L);
		}


		Random random = new Random();

		// Choose one centroid uniformly at random from among the data points.
		final INDArray firstCentroid = this.initFeatures.remove(random.nextInt(this.initFeatures.size()));
		this.centroids = NDArrays.create(this.nbCluster,firstCentroid.columns());
		this.centroids.putRow(0,firstCentroid);
		log.info("Added initial centroid");
		INDArray dxs;

		for (int j = 1; j < this.nbCluster; j++) {
			// For each data point x, compute D(x)
			dxs = this.computeDxs();

			// Add one new data point as a center.
			INDArray features;
			double r = random.nextDouble() * (double) dxs.getScalar(dxs.length() - 1).element();
			for (int i = 0; i < dxs.length(); i++) {
				if ((double) dxs.getScalar(i).element() >= r) {
					features = this.initFeatures.remove(i);
					this.centroids.putRow(j,features);
					break;
				}
			}
		}

		this.initFeatures.clear();
	}


	protected INDArray computeDxs() {
		INDArray dxs = NDArrays.create(this.initFeatures.size(), this.initFeatures.get(0).columns());

		int sum = 0;
		INDArray features;
		int nearestCentroidIndex;
		INDArray nearestCentroid;

		for (int i = 0; i < this.initFeatures.size(); i++) {
			features = this.initFeatures.get(i);
			nearestCentroidIndex = this.nearestCentroid(features);
			nearestCentroid = this.centroids.getRow(nearestCentroidIndex);
			sum += Math.pow(getDistance(features, nearestCentroid), 2);
			dxs.putScalar(i,sum);
		}

		return dxs;
	}


	public void reset() {
		this.counts = null;
		this.centroids = null;
		this.initFeatures = new ArrayList<>();
	}

}
