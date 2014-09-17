package org.deeplearning4j.clustering;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.distancefunction.DistanceFunction;
import org.nd4j.linalg.distancefunction.EuclideanDistance;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * Shamelessly based on:
 * https://github.com/pmerienne/trident-ml/blob/master/src/main/java/com/github/pmerienne/trident/ml/clustering/KMeans.java
 *
 * adapted to ndarray matrices
 * @author Adam Gibson
 *
 */
public class KMeansClustering implements Serializable {


    private static final long serialVersionUID = 338231277453149972L;
    private static Logger log = LoggerFactory.getLogger(KMeansClustering.class);

    private List<Long> counts = null;
    private INDArray centroids;
    private List<INDArray> initFeatures = new ArrayList<>();
    private Class<? extends DistanceFunction> clazz;
    private transient ExecutorService exec;


    private Integer nbCluster;

    public KMeansClustering(Integer nbCluster,Class<? extends DistanceFunction> clazz) {
        this.nbCluster = nbCluster;
        this.clazz = clazz;
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
            this.centroids.getRow(nearestCentroid).addi(update);
            return nearestCentroid;
        }
    }


    public INDArray distribution(INDArray features) {
        if (!this.isReady()) {
            throw new IllegalStateException("KMeans is not ready yet");
        }

        INDArray distribution = Nd4j.create(1,this.nbCluster);
        INDArray currentCentroid;
        for (int i = 0; i < this.nbCluster; i++) {
            currentCentroid = this.centroids.getRow(i);
            distribution.putScalar(i,getDistance(currentCentroid,features));
        }

        return distribution;
    }


    private float getDistance(INDArray m1,INDArray m2) {
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

        Float minDistance = Float.MAX_VALUE;
        INDArray currentCentroid;
        Float currentDistance;
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
        if(exec == null) {
            exec = Executors.newScheduledThreadPool(Runtime.getRuntime().availableProcessors());
        }
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
        final INDArray firstCentroid = this.initFeatures.remove(random.nextInt(this.initFeatures.size())).linearView();
        this.centroids = Nd4j.create(this.nbCluster,firstCentroid.columns());
        this.centroids.putRow(0,firstCentroid);
        log.info("Added initial centroid");
        INDArray dxs;

        for (int j = 1; j < this.nbCluster; j++) {
            // For each data point x, compute D(x)
            dxs = this.computeDxs();

            // Add one new data point as a center.
            INDArray features;
            double r = random.nextFloat() *   dxs.get(dxs.length() - 1);
            for (int i = 0; i < dxs.length(); i++) {
                if (dxs.get(i) >= r) {
                    features = this.initFeatures.remove(i);
                    this.centroids.putRow(j,features);
                    break;
                }
            }
        }

        this.initFeatures.clear();
    }


    protected INDArray computeDxs() {
        final INDArray dxs = Nd4j.create(this.initFeatures.size(), this.initFeatures.get(0).columns());

        final AtomicInteger sum = new AtomicInteger(0);

        final CountDownLatch latch = new CountDownLatch(initFeatures.size());
        for (int i = 0; i < this.initFeatures.size(); i++) {
            final int i2 = i;
            exec.execute(new Runnable() {
                @Override
                public void run() {
                    INDArray features = initFeatures.get(i2);
                    int nearestCentroidIndex = nearestCentroid(features);
                    INDArray  nearestCentroid = centroids.getRow(nearestCentroidIndex);
                    sum.getAndAdd((int) Math.pow(getDistance(features, nearestCentroid), 2));
                    dxs.putScalar(i2,sum.get());
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


    public void reset() {
        this.counts = null;
        this.centroids = null;
        this.initFeatures = new ArrayList<>();
    }

}
