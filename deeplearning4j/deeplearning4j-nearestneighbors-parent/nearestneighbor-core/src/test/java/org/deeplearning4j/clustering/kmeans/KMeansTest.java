/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.clustering.kmeans;

import lombok.val;
import lombok.var;
import org.apache.commons.lang3.time.StopWatch;
import org.deeplearning4j.clustering.algorithm.Distance;
import org.deeplearning4j.clustering.cluster.Cluster;
import org.deeplearning4j.clustering.cluster.ClusterSet;
import org.deeplearning4j.clustering.cluster.Point;
import org.deeplearning4j.clustering.cluster.PointClassification;
import org.joda.time.Duration;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 7/2/17.
 */
public class KMeansTest {

    @Test
    public void testKMeans() {
        Nd4j.getRandom().setSeed(7);
        KMeansClustering kMeansClustering = KMeansClustering.setup(5, 5, Distance.EUCLIDIAN);
        List<Point> points = Point.toPoints(Nd4j.randn(5, 5));
        ClusterSet clusterSet = kMeansClustering.applyTo(points);
        PointClassification pointClassification = clusterSet.classifyPoint(points.get(0));
        System.out.println(pointClassification);
    }

    @Test
    public void testKmeansCosine() {

        Nd4j.getRandom().setSeed(7);
        int numClusters = 5;
        KMeansClustering kMeansClustering = KMeansClustering.setup(numClusters, 1000, Distance.COSINE_DISTANCE, true);
        List<Point> points = Point.toPoints(Nd4j.rand(5, 300));
        ClusterSet clusterSet = kMeansClustering.applyTo(points);
        PointClassification pointClassification = clusterSet.classifyPoint(points.get(0));


        KMeansClustering kMeansClusteringEuclidean = KMeansClustering.setup(numClusters, 1000, Distance.EUCLIDIAN);
        ClusterSet clusterSetEuclidean = kMeansClusteringEuclidean.applyTo(points);
        PointClassification pointClassificationEuclidean = clusterSetEuclidean.classifyPoint(points.get(0));
        System.out.println("Cosine " + pointClassification);
        System.out.println("Euclidean " + pointClassificationEuclidean);

        assertEquals(pointClassification.getCluster().getPoints().get(0),
                        pointClassificationEuclidean.getCluster().getPoints().get(0));
    }

    @Test
    public void testPerformance() {
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
        Nd4j.getRandom().setSeed(7);
        int numClusters = 3;
        StopWatch watch = new StopWatch();
        watch.start();
        KMeansClustering kMeansClustering = KMeansClustering.setup(numClusters, 1000, Distance.COSINE_DISTANCE, true);
        List<Point> points = Point.toPoints(Nd4j.linspace(0, 5000*300, 5000*300).reshape(5000,300 ));

        ClusterSet clusterSet = kMeansClustering.applyTo(points);
        watch.stop();
        System.out.println("Elapsed for clustering : " + watch);

        watch.reset();
        watch.start();
        for (Point p : points) {
            PointClassification pointClassification = clusterSet.classifyPoint(p);
        }
        watch.stop();
        System.out.println("Elapsed for search: " + watch);
    }

    @Test
    public void testCorrectness() {

        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
        Nd4j.getRandom().setSeed(7);
        int numClusters = 3;
        KMeansClustering kMeansClustering = KMeansClustering.setup(numClusters, 1000, Distance.EUCLIDIAN, false);
        double[] data = new double[]{
                15, 16,
                16, 18.5,
                17, 20.2,
                16.4, 17.12,
                17.23, 18.12,
                43, 43,
                44.43, 45.212,
                45.8, 54.23,
                46.313, 43.123,
                50.21, 46.3,
                99, 99.22,
                100.32, 98.123,
                100.32, 97.423,
                102, 93.23,
                102.23, 94.23
        };
        val points = Point.toPoints(Nd4j.createFromArray(data).reshape(15,2 ));

        val clusterSet = kMeansClustering.applyTo(points);


        val row0 = Nd4j.createFromArray(new double[]{26.1433,   28.8970});
        val row1 = Nd4j.createFromArray(new double[]{51.6358,   49.0883});
        val row2 = Nd4j.createFromArray(new double[]{90.3700,   86.8592});

        var clusters = clusterSet.getClusters();
        assertEquals(clusters.get(0).getCenter().getArray(), row0);
        assertEquals(clusters.get(1).getCenter().getArray(), row1);
        assertEquals(clusters.get(2).getCenter().getArray(), row2);

        PointClassification pointClassification = null;
        for (Point p : points) {
            pointClassification = clusterSet.classifyPoint(p);
        }
        assertEquals(Nd4j.createFromArray(new double[]{51.6358,   49.0883}),
                pointClassification.getCluster().getCenter().getArray());

        clusters = clusterSet.getClusters();
        assertEquals(clusters.get(0).getCenter().getArray(), row0);
        assertEquals(clusters.get(1).getCenter().getArray(), row1);
        assertEquals(clusters.get(2).getCenter().getArray(), row2);
    }
}
