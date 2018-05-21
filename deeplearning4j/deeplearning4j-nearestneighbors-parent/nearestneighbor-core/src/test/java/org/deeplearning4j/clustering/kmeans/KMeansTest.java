package org.deeplearning4j.clustering.kmeans;

import org.deeplearning4j.clustering.cluster.ClusterSet;
import org.deeplearning4j.clustering.cluster.Point;
import org.deeplearning4j.clustering.cluster.PointClassification;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 7/2/17.
 */
public class KMeansTest {

    @Test
    public void testKMeans() {
        Nd4j.getRandom().setSeed(7);
        KMeansClustering kMeansClustering = KMeansClustering.setup(5, 5, "euclidean");
        List<Point> points = Point.toPoints(Nd4j.randn(5, 5));
        ClusterSet clusterSet = kMeansClustering.applyTo(points);
        PointClassification pointClassification = clusterSet.classifyPoint(points.get(0));
        System.out.println(pointClassification);
    }

    @Test
    public void testKmeansCosine() {
        Nd4j.getRandom().setSeed(7);
        int numClusters = 5;
        KMeansClustering kMeansClustering = KMeansClustering.setup(numClusters, 1000, "cosinesimilarity", true);
        List<Point> points = Point.toPoints(Nd4j.randn(5, 5));
        ClusterSet clusterSet = kMeansClustering.applyTo(points);
        PointClassification pointClassification = clusterSet.classifyPoint(points.get(0));


        KMeansClustering kMeansClusteringEuclidean = KMeansClustering.setup(numClusters, 1000, "euclidean");
        ClusterSet clusterSetEuclidean = kMeansClusteringEuclidean.applyTo(points);
        PointClassification pointClassificationEuclidean = clusterSetEuclidean.classifyPoint(points.get(0));
        System.out.println("Cosine " + pointClassification);
        System.out.println("Euclidean " + pointClassificationEuclidean);


        assertEquals(pointClassification.getCluster().getPoints().get(0),
                        pointClassificationEuclidean.getCluster().getPoints().get(0));
    }

}
