package com.ccc.deeplearning.clustering;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class KMeansClustering implements Serializable {


	private static final long serialVersionUID = 338231277453149972L;
	private static Logger log = LoggerFactory.getLogger(KMeansClustering.class);
	
	private List<Long> counts = null;
	private DoubleMatrix centroids;
	private List<DoubleMatrix> initFeatures = new ArrayList<DoubleMatrix>();

	private Integer nbCluster;

	public KMeansClustering(Integer nbCluster) {
		this.nbCluster = nbCluster;
	}


	public Integer classify(DoubleMatrix features) {
		if (!this.isReady()) {
			throw new IllegalStateException("KMeans is not ready yet");
		}

		// Find nearest centroid
		Integer nearestCentroidIndex = this.nearestCentroid(features);
		return nearestCentroidIndex;
	}


	public Integer update(DoubleMatrix features) {
		if (!this.isReady()) {
			this.initIfPossible(features);
			log.info("Initializing feature vector with length of " + features.length);
			return null;
		} else {
			Integer nearestCentroid = this.classify(features);

			// Increment count
			this.counts.set(nearestCentroid, this.counts.get(nearestCentroid) + 1);

			// Move centroid
			DoubleMatrix update = features.sub(this.centroids.getRow(nearestCentroid)).mul( 1.0 / this.counts.get(nearestCentroid));
			this.centroids.putRow(nearestCentroid,this.centroids.getRow(nearestCentroid).add(update));

			return nearestCentroid;
		}
	}


	public DoubleMatrix distribution(DoubleMatrix features) {
		if (!this.isReady()) {
			throw new IllegalStateException("KMeans is not ready yet");
		}

		DoubleMatrix distribution = new DoubleMatrix(1,this.nbCluster);
		DoubleMatrix currentCentroid;
		for (int i = 0; i < this.nbCluster; i++) {
			currentCentroid = this.centroids.getRow(i);
			distribution.put(i,currentCentroid.distance2(features));
		}

		return distribution;
	}


	public DoubleMatrix getCentroids() {
		return this.centroids;
	}

	protected Integer nearestCentroid(DoubleMatrix features) {
		// Find nearest centroid
		Integer nearestCentroidIndex = 0;

		Double minDistance = Double.MAX_VALUE;
		DoubleMatrix currentCentroid;
		Double currentDistance;
		for (int i = 0; i < this.centroids.rows; i++) {
			currentCentroid = this.centroids.getRow(i);
			log.info("Current centroid " + currentCentroid.length);
			if (currentCentroid != null) {
				currentDistance = currentCentroid.distance2(features);
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

	protected void initIfPossible(DoubleMatrix features) {
		this.initFeatures.add(features);
		log.info("Added feature vector of length " + features.length);
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
		this.counts = new ArrayList<Long>(this.nbCluster);
		for (int i = 0; i < this.nbCluster; i++) {
			this.counts.add(0L);
		}


		Random random = new Random();

		// Choose one centroid uniformly at random from among the data points.
		final DoubleMatrix firstCentroid = this.initFeatures.remove(random.nextInt(this.initFeatures.size()));
		this.centroids = new DoubleMatrix(this.nbCluster,firstCentroid.columns);
		this.centroids.putRow(0,firstCentroid);
		log.info("Added initial centroid");
		DoubleMatrix dxs;

		for (int j = 1; j < this.nbCluster; j++) {
			// For each data point x, compute D(x)
			dxs = this.computeDxs();

			// Add one new data point as a center.
			DoubleMatrix features;
			double r = random.nextDouble() * dxs.get(dxs.length - 1);
			for (int i = 0; i < dxs.length; i++) {
				if (dxs.get(i) >= r) {
					features = this.initFeatures.remove(i);
					this.centroids.putRow(j,features);
					break;
				}
			}
		}

		this.initFeatures.clear();
	}

	/**
	 * For each features in {@link KMeans#initFeatures}, compute D(x), the
	 * distance between x and the nearest center that has already been chosen.
	 * 
	 * @return
	 */
	protected DoubleMatrix computeDxs() {
		DoubleMatrix dxs = new DoubleMatrix(this.initFeatures.size(),this.initFeatures.get(0).columns);

		int sum = 0;
		DoubleMatrix features;
		int nearestCentroidIndex;
		DoubleMatrix nearestCentroid;
		for (int i = 0; i < this.initFeatures.size(); i++) {
			features = this.initFeatures.get(i);
			nearestCentroidIndex = this.nearestCentroid(features);
			nearestCentroid = this.centroids.getRow(nearestCentroidIndex);
			sum += MatrixFunctions.pow(features.distance2(nearestCentroid), 2);
			dxs.put(i,sum);
		}

		return dxs;
	}


	public void reset() {
		this.counts = null;
		this.centroids = null;
		this.initFeatures = new ArrayList<DoubleMatrix>();
	}

}
