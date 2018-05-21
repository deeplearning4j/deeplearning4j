/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.clustering.vptree;

import org.deeplearning4j.clustering.sptree.DataPoint;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Counter;
import org.nd4j.linalg.primitives.Pair;

import java.util.ArrayList;
import java.util.List;
import java.util.PriorityQueue;
import java.util.TreeSet;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author Anatoly Borisov
 */
public class VpTreeNodeTest {


    private static class DistIndex implements Comparable<DistIndex> {
        public double dist;
        public int index;

        public int compareTo(DistIndex r) {
            return Double.compare(dist, r.dist);
        }
    }

    @Test
    public void testKnnK() {
        INDArray arr = Nd4j.randn(10, 5);
        VPTree t = new VPTree(arr, false);
        List<DataPoint> resultList = new ArrayList<>();
        List<Double> distances = new ArrayList<>();
        t.search(arr.getRow(0), 5, resultList, distances);
        assertEquals(5, resultList.size());
    }


    @Test
    public void testParallel() {
        Nd4j.getRandom().setSeed(7);
        INDArray randn = Nd4j.rand(1000, 100);
        VPTree vpTree = new VPTree(randn, false, 2);
        Nd4j.getRandom().setSeed(7);
        VPTree vpTreeNoParallel = new VPTree(randn, false, 1);
        List<DataPoint> results = new ArrayList<>();
        List<Double> distances = new ArrayList<>();
        List<DataPoint> noParallelResults = new ArrayList<>();
        List<Double> noDistances = new ArrayList<>();
        vpTree.search(randn.getRow(0), 10, results, distances);
        vpTreeNoParallel.search(randn.getRow(0), 10, noParallelResults, noDistances);
        assertEquals(noParallelResults.size(), results.size());
        assertEquals(noParallelResults, results);
        assertEquals(noDistances, distances);

    }

    @Test
    public void knnManualRandom() {
        knnManual(Nd4j.randn(3, 5));
    }

    @Test
    public void knnManualNaturals() {
        knnManual(generateNaturalsMatrix(20, 2));
    }

    public static void knnManual(INDArray arr) {
        Nd4j.getRandom().setSeed(7);
        VPTree t = new VPTree(arr, false);
        int k = 1;
        int m = arr.rows();
        for (int targetIndex = 0; targetIndex < m; targetIndex++) {
            // Do an exhaustive search
            TreeSet<Integer> s = new TreeSet<>();
            INDArray query = arr.getRow(targetIndex);

            Counter<Integer> counter = new Counter<>();
            for (int j = 0; j < m; j++) {
                double d = t.distance(query, (arr.getRow(j)));
                counter.setCount(j, (float) d);

            }

            PriorityQueue<Pair<Integer, Double>> pq = counter.asReversedPriorityQueue();
            // keep closest k
            for (int i = 0; i < k; i++) {
                Pair<Integer, Double> di = pq.poll();
                System.out.println("exhaustive d=" + di.getFirst());
                s.add(di.getFirst());
            }

            // Check what VPTree gives for results
            List<DataPoint> results = new ArrayList<>();
            VPTreeFillSearch fillSearch = new VPTreeFillSearch(t, k, query);
            fillSearch.search();
            results = fillSearch.getResults();

            //List<DataPoint> items = t.getItems();
            TreeSet<Integer> resultSet = new TreeSet<>();

            // keep k in a set
            for (int i = 0; i < k; ++i) {
                DataPoint result = results.get(i);
                int r = result.getIndex();
                resultSet.add(r);
            }



            // check
            for (int r : resultSet) {
                INDArray expectedResult = arr.getRow(r);
                if (!s.contains(r)) {
                    fillSearch = new VPTreeFillSearch(t, k, query);
                    fillSearch.search();
                    results = fillSearch.getResults();
                }
                assertTrue(String.format(
                                "VPTree result" + " %d is not in the " + "closest %d " + " " + "from the exhaustive"
                                                + " search with query point %s and "
                                                + "result %s and target not found %s",
                                r, k, query.toString(), results.toString(), expectedResult.toString()), s.contains(r));
            }

        }
    }

    @Test
    public void vpTreeTest() {
        List<DataPoint> points = new ArrayList<>();
        points.add(new DataPoint(0, Nd4j.create(new double[] {55, 55})));
        points.add(new DataPoint(1, Nd4j.create(new double[] {60, 60})));
        points.add(new DataPoint(2, Nd4j.create(new double[] {65, 65})));
        VPTree tree = new VPTree(points, "euclidean");
        List<DataPoint> add = new ArrayList<>();
        List<Double> distances = new ArrayList<>();
        tree.search(Nd4j.create(new double[] {50, 50}), 1, add, distances);
        DataPoint assertion = add.get(0);
        assertEquals(new DataPoint(0, Nd4j.create(new double[] {55, 55})), assertion);

        tree.search(Nd4j.create(new double[] {60, 60}), 1, add, distances);
        assertion = add.get(0);
        assertEquals(Nd4j.create(new double[] {60, 60}), assertion.getPoint());
    }

    @Test(expected = ND4JIllegalStateException.class)
    public void vpTreeTest2() {
        List<DataPoint> points = new ArrayList<>();
        points.add(new DataPoint(0, Nd4j.create(new double[] {55, 55})));
        points.add(new DataPoint(1, Nd4j.create(new double[] {60, 60})));
        points.add(new DataPoint(2, Nd4j.create(new double[] {65, 65})));
        VPTree tree = new VPTree(points, "euclidean");

        tree.search(Nd4j.create(1, 10), 2, new ArrayList<DataPoint>(), new ArrayList<Double>());
    }

    @Test(expected = ND4JIllegalStateException.class)
    public void vpTreeTest3() {
        List<DataPoint> points = new ArrayList<>();
        points.add(new DataPoint(0, Nd4j.create(new double[] {55, 55})));
        points.add(new DataPoint(1, Nd4j.create(new double[] {60, 60})));
        points.add(new DataPoint(2, Nd4j.create(new double[] {65, 65})));
        VPTree tree = new VPTree(points, "euclidean");

        tree.search(Nd4j.create(2, 10), 2, new ArrayList<DataPoint>(), new ArrayList<Double>());
    }

    @Test(expected = ND4JIllegalStateException.class)
    public void vpTreeTest4() {
        List<DataPoint> points = new ArrayList<>();
        points.add(new DataPoint(0, Nd4j.create(new double[] {55, 55})));
        points.add(new DataPoint(1, Nd4j.create(new double[] {60, 60})));
        points.add(new DataPoint(2, Nd4j.create(new double[] {65, 65})));
        VPTree tree = new VPTree(points, "euclidean");

        tree.search(Nd4j.create(2, 10, 10), 2, new ArrayList<DataPoint>(), new ArrayList<Double>());
    }

    public static INDArray generateNaturalsMatrix(int nrows, int ncols) {
        INDArray col = Nd4j.arange(0, nrows).transpose();
        INDArray points = Nd4j.zeros(nrows, ncols);
        if (points.isColumnVectorOrScalar())
            points = col.dup();
        else {
            for (int i = 0; i < ncols; i++)
                points.putColumn(i, col);
        }
        return points;
    }

    @Test
    public void testVPSearchOverNaturals1D() throws Exception {
        testVPSearchOverNaturalsPD(20, 1, 5);
    }

    @Test
    public void testVPSearchOverNaturals2D() throws Exception {
        testVPSearchOverNaturalsPD(20, 2, 5);
    }

    public static void testVPSearchOverNaturalsPD(int nrows, int ncols, int K) throws Exception {
        final int queryPoint = 12;

        INDArray points = generateNaturalsMatrix(nrows, ncols);
        INDArray query = Nd4j.zeros(1, ncols);
        for (int i = 0; i < ncols; i++)
            query.putScalar(0, i, queryPoint);

        INDArray trueResults = Nd4j.zeros(K, ncols);
        for (int j = 0; j < K; j++) {
            int pt = queryPoint - K / 2 + j;
            for (int i = 0; i < ncols; i++)
                trueResults.putScalar(j, i, pt);
        }

        VPTree tree = new VPTree(points, "euclidean", 1, false);

        List<DataPoint> results = new ArrayList<>();
        List<Double> distances = new ArrayList<>();
        tree.search(query, K, results, distances);
        int dimensionToSort = 0;

        INDArray sortedResults = Nd4j.zeros(K, ncols);
        int i = 0;
        for (DataPoint p : results) {
            sortedResults.putRow(i++, p.getPoint());
        }

        sortedResults = Nd4j.sort(sortedResults, dimensionToSort, true);
        assertTrue(trueResults.equalsWithEps(sortedResults, 1e-12));

        VPTreeFillSearch fillSearch = new VPTreeFillSearch(tree, K, query);
        fillSearch.search();
        results = fillSearch.getResults();
        sortedResults = Nd4j.zeros(K, ncols);
        i = 0;
        for (DataPoint p : results)
            sortedResults.putRow(i++, p.getPoint());
        INDArray[] sortedWithIndices = Nd4j.sortWithIndices(sortedResults, dimensionToSort, true);;
        sortedResults = sortedWithIndices[1];
        assertEquals(trueResults.sumNumber().doubleValue(), sortedResults.sumNumber().doubleValue(), 1e-12);
    }

}
