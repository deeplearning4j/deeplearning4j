package org.deeplearning4j.clustering.kmeans;

import org.deeplearning4j.clustering.cluster.Cluster;
import org.deeplearning4j.clustering.cluster.ClusterSet;
import org.deeplearning4j.clustering.cluster.Point;
import org.deeplearning4j.clustering.cluster.PointClassification;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

/**
 * Created by agibsonccc on 7/2/17.
 */
public class KMeansTest {

    @Test
    public void testKMeans() {
        KMeansClustering kMeansClustering = KMeansClustering.setup(5,5,"euclidean");
        List<Point> points = Point.toPoints(Nd4j.randn(5,5));
        ClusterSet clusterSet = kMeansClustering.applyTo(points);
        PointClassification pointClassification = clusterSet.classifyPoint(points.get(0));
        System.out.println(pointClassification);
    }

}
