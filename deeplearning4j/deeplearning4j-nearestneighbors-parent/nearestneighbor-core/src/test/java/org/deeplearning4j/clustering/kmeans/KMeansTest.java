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
import org.deeplearning4j.clustering.cluster.*;
import org.joda.time.Duration;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

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

    @Ignore
    @Test
    public void testPerformanceAllIterations() {
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
        Nd4j.getRandom().setSeed(7);
        int numClusters = 20;
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
    @Ignore
    public void testPerformanceWithConvergence() {
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
        Nd4j.getRandom().setSeed(7);
        int numClusters = 20;
        StopWatch watch = new StopWatch();
        watch.start();
        KMeansClustering kMeansClustering = KMeansClustering.setup(numClusters, Distance.COSINE_DISTANCE, false);

        List<Point> points = Point.toPoints(Nd4j.linspace(0, 10000*300, 10000*300).reshape(10000,300 ));

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

        watch.reset();
        watch.start();
        kMeansClustering = KMeansClustering.setup(numClusters, 0.05, Distance.COSINE_DISTANCE, false);

        points = Point.toPoints(Nd4j.linspace(0, 10000*300, 10000*300).reshape(10000,300 ));

        clusterSet = kMeansClustering.applyTo(points);
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

        /*for (int c = 0; c < 10; ++c)*/ {
            Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
            Nd4j.getRandom().setSeed(7);
            int numClusters = 3;
            KMeansClustering kMeansClustering = KMeansClustering.setup(numClusters, 1000, Distance.EUCLIDIAN, true);
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
            List<Point> points = Point.toPoints(Nd4j.createFromArray(data).reshape(15, 2));

            ClusterSet clusterSet = kMeansClustering.applyTo(points);


            INDArray row0 = Nd4j.createFromArray(new double[]{16.6575, 18.4850});
            INDArray row1 = Nd4j.createFromArray(new double[]{32.6050, 31.1500});
            INDArray row2 = Nd4j.createFromArray(new double[]{75.9348, 74.1990});

            /*List<Cluster> clusters = clusterSet.getClusters();
            assertEquals(row0, clusters.get(0).getCenter().getArray());
            assertEquals(row1, clusters.get(1).getCenter().getArray());
            assertEquals(row2, clusters.get(2).getCenter().getArray());*/

            PointClassification pointClassification = null;
            for (Point p : points) {
                pointClassification = clusterSet.classifyPoint(p);
                System.out.println("Point: " + p.getArray() + " " + " assigned to cluster: " + pointClassification.getCluster().getCenter().getArray());
                List<Cluster> clusters = clusterSet.getClusters();
                for (int i = 0; i < clusters.size(); ++i)
                    System.out.println("Choice: " + clusters.get(i).getCenter().getArray());
            }
            /*assertEquals(Nd4j.createFromArray(new double[]{75.9348, 74.1990}),
                    pointClassification.getCluster().getCenter().getArray());*/

        /*clusters = clusterSet.getClusters();
        assertEquals(row0, clusters.get(0).getCenter().getArray());
        assertEquals(row1, clusters.get(1).getCenter().getArray());
        assertEquals(row2, clusters.get(2).getCenter().getArray());*/
        }
    }

    @Test
    public void testCentersHolder() {
            int rows = 3, cols = 2;
            CentersHolder ch = new CentersHolder(rows, cols);

            INDArray row0 = Nd4j.createFromArray(new double[]{16.4000, 17.1200});
            INDArray row1 = Nd4j.createFromArray(new double[]{45.8000, 54.2300});
            INDArray row2 = Nd4j.createFromArray(new double[]{95.9348, 94.1990});

            ch.addCenter(row0);
            ch.addCenter(row1);
            ch.addCenter(row2);

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

        INDArray pointData = Nd4j.createFromArray(data);
        List<Point> points = Point.toPoints(pointData.reshape(15,2));

        for (int i = 0 ; i < points.size(); ++i) {
            INDArray dist = ch.getMinDistances(points.get(i), Distance.EUCLIDIAN);
            System.out.println("Point: " + points.get(i).getArray());
            System.out.println("Centers: " + ch.getCenters());
            System.out.println("Distance: " + dist);
            System.out.println();
        }
    }
}
