package org.deeplearning4j.clustering.cluster;

import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

public class ClusterSetTest {
    @Test
    public void testGetMostPopulatedClusters() {
        ClusterSet clusterSet = new ClusterSet(false);
        List<Cluster> clusters = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            Cluster cluster = new Cluster();
            cluster.setPoints(Point.toPoints(Nd4j.randn(i + 1, 5)));
            clusters.add(cluster);
        }
        clusterSet.setClusters(clusters);
        List<Cluster> mostPopulatedClusters = clusterSet.getMostPopulatedClusters(5);
        for (int i = 0; i < 5; i++) {
            Assert.assertEquals(5 - i, mostPopulatedClusters.get(i).getPoints().size());
        }
    }
}
