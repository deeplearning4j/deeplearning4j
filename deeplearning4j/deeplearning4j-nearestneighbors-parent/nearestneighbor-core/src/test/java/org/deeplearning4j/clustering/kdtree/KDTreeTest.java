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

package org.deeplearning4j.clustering.kdtree;

import com.google.common.primitives.Doubles;
import lombok.val;
import org.joda.time.Duration;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.opencv.ml.KNearest;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by agibsonccc on 1/1/15.
 */
public class KDTreeTest {

    private KDTree kdTree;

    @BeforeClass
    public static void beforeClass(){
        Nd4j.setDataType(DataType.FLOAT);
    }

    @Before
    public void setUp() {
         kdTree = new KDTree(2);
        double[] data = new double[]{7,2};
        kdTree.insert(Nd4j.createFromArray(data));
        data = new double[]{5,4};
        kdTree.insert(Nd4j.createFromArray(data));
        data = new double[]{2,3};
        kdTree.insert(Nd4j.createFromArray(data));
        data = new double[]{4,7};
        kdTree.insert(Nd4j.createFromArray(data));
        data = new double[]{9,6};
        kdTree.insert(Nd4j.createFromArray(data));
        data = new double[]{8,1};
        kdTree.insert(Nd4j.createFromArray(data));
    }

    @Test
    public void testTree() {
        KDTree tree = new KDTree(2);
        INDArray half = Nd4j.create(new double[] {0.5, 0.5}, new long[]{1,2}).castTo(DataType.FLOAT);
        INDArray one = Nd4j.create(new double[] {1, 1}, new long[]{1,2}).castTo(DataType.FLOAT);
        tree.insert(half);
        tree.insert(one);
        Pair<Double, INDArray> pair = tree.nn(Nd4j.create(new double[] {0.5, 0.5}, new long[]{1,2}).castTo(DataType.FLOAT));
        assertEquals(half, pair.getValue());
    }

    @Test
    public void testInsert() {
        int elements = 10;
        List<Double> digits = Arrays.asList(1.0, 0.0, 2.0, 3.0);

        KDTree kdTree = new KDTree(digits.size());
        List<List<Double>> lists = new ArrayList<>();
        for (int i = 0; i < elements; i++) {
            List<Double> thisList = new ArrayList<>(digits.size());
            for (int k = 0; k < digits.size(); k++) {
                thisList.add(digits.get(k) + i);
            }
            lists.add(thisList);
        }

        for (int i = 0; i < elements; i++) {
            double[] features = Doubles.toArray(lists.get(i));
            INDArray ind = Nd4j.create(features, new long[]{1, features.length}, DataType.FLOAT);
            kdTree.insert(ind);
            assertEquals(i + 1, kdTree.size());
        }
    }

    @Test
    public void testDelete() {
        int elements = 10;
        List<Double> digits = Arrays.asList(1.0, 0.0, 2.0, 3.0);

        KDTree kdTree = new KDTree(digits.size());
        List<List<Double>> lists = new ArrayList<>();
        for (int i = 0; i < elements; i++) {
            List<Double> thisList = new ArrayList<>(digits.size());
            for (int k = 0; k < digits.size(); k++) {
                thisList.add(digits.get(k) + i);
            }
            lists.add(thisList);
        }

        INDArray toDelete = Nd4j.empty(DataType.DOUBLE),
                 leafToDelete = Nd4j.empty(DataType.DOUBLE);
        for (int i = 0; i < elements; i++) {
            double[] features = Doubles.toArray(lists.get(i));
            INDArray ind = Nd4j.create(features, new long[]{1, features.length}, DataType.FLOAT);
            if (i == 1)
                toDelete = ind;
            if (i == elements - 1) {
                leafToDelete = ind;
            }
            kdTree.insert(ind);
            assertEquals(i + 1, kdTree.size());
        }

        kdTree.delete(toDelete);
        assertEquals(9, kdTree.size());
        kdTree.delete(leafToDelete);
        assertEquals(8, kdTree.size());
    }

    @Test
    public void testNN() {
        int n = 10;

        // make a KD-tree of dimension {#n}
        KDTree kdTree = new KDTree(n);
        for (int i = -1; i < n; i++) {
            // Insert a unit vector along each dimension
            List<Double> vec = new ArrayList<>(n);
            // i = -1 ensures the origin is in the Tree
            for (int k = 0; k < n; k++) {
                vec.add((k == i) ? 1.0 : 0.0);
            }
            INDArray indVec = Nd4j.create(Doubles.toArray(vec), new long[]{1, vec.size()}, DataType.FLOAT);
            kdTree.insert(indVec);
        }
        Random rand = new Random();

        // random point in the Hypercube
        List<Double> pt = new ArrayList(n);
        for (int k = 0; k < n; k++) {
            pt.add(rand.nextDouble());
        }
        Pair<Double, INDArray> result = kdTree.nn(Nd4j.create(Doubles.toArray(pt), new long[]{1, pt.size()}, DataType.FLOAT));

        // Always true for points in the unitary hypercube
        assertTrue(result.getKey() < Double.MAX_VALUE);

    }

    @Test
    public void testKNN() {
        int n = 10;
        // make a KD-tree of dimension {#n}
        KDTree kdTree = new KDTree(n);
        for (int i = -1; i < n; i++) {
            // Insert a unit vector along each dimension
            List<Double> vec = new ArrayList<>(n);
            // i = -1 ensures the origin is in the Tree
            for (int k = 0; k < n; k++) {
                vec.add((k == i) ? 1.0 : 0.0);
            }
            INDArray indVec = Nd4j.create(Nd4j.createBuffer(Doubles.toArray(vec)));
            kdTree.insert(indVec);
        }
        Random rand = new Random();
        // random point in the Hypercube
        List<Double> pt = new ArrayList(n);
        for (int k = 0; k < n; k++) {
            pt.add(rand.nextDouble() * 10.0);
        }
        List<Pair<Double, INDArray>> list = kdTree.knn(Nd4j.create(Nd4j.createBuffer(Doubles.toArray(pt))), 20.0);
    }

    @Test
    public void testKNN_Simple() {
        int n = 2;
        KDTree kdTree = new KDTree(n);

        double[] data = new double[]{3,3};
        kdTree.insert(Nd4j.createFromArray(data));
        data = new double[]{1,1};
        kdTree.insert(Nd4j.createFromArray(data));
        data = new double[]{2,2};
        kdTree.insert(Nd4j.createFromArray(data));

        data = new double[]{0,0};
        List<Pair<Double, INDArray>> result = kdTree.knn(Nd4j.createFromArray(data), 4.5);

        assertEquals(1.0, result.get(0).getSecond().getDouble(0), 1e-5);
        assertEquals(1.0, result.get(0).getSecond().getDouble(1), 1e-5);

        assertEquals(2.0, result.get(1).getSecond().getDouble(0), 1e-5);
        assertEquals(2.0, result.get(1).getSecond().getDouble(1), 1e-5);

        assertEquals(3.0, result.get(2).getSecond().getDouble(0), 1e-5);
        assertEquals(3.0, result.get(2).getSecond().getDouble(1), 1e-5);
    }

    @Test
    public void testKNN_1() {

        assertEquals(6, kdTree.size());

        double[] data = new double[]{8,1};
        List<Pair<Double, INDArray>> result = kdTree.knn(Nd4j.createFromArray(data), 10.0);
        assertEquals(8.0, result.get(0).getSecond().getDouble(0), 1e-5);
        assertEquals(1.0, result.get(0).getSecond().getDouble(1), 1e-5);
        assertEquals(7.0, result.get(1).getSecond().getDouble(0), 1e-5);
        assertEquals(2.0, result.get(1).getSecond().getDouble(1), 1e-5);
        assertEquals(5.0, result.get(2).getSecond().getDouble(0), 1e-5);
        assertEquals(4.0, result.get(2).getSecond().getDouble(1), 1e-5);
        assertEquals(9.0, result.get(3).getSecond().getDouble(0), 1e-5);
        assertEquals(6.0, result.get(3).getSecond().getDouble(1), 1e-5);
        assertEquals(2.0, result.get(4).getSecond().getDouble(0), 1e-5);
        assertEquals(3.0, result.get(4).getSecond().getDouble(1), 1e-5);
        assertEquals(4.0, result.get(5).getSecond().getDouble(0), 1e-5);
        assertEquals(7.0, result.get(5).getSecond().getDouble(1), 1e-5);
    }

    @Test
    public void testKNN_2() {
        double[] data = new double[]{8, 1};
        List<Pair<Double, INDArray>> result = kdTree.knn(Nd4j.createFromArray(data), 5.0);
        assertEquals(8.0, result.get(0).getSecond().getDouble(0), 1e-5);
        assertEquals(1.0, result.get(0).getSecond().getDouble(1), 1e-5);
        assertEquals(7.0, result.get(1).getSecond().getDouble(0), 1e-5);
        assertEquals(2.0, result.get(1).getSecond().getDouble(1), 1e-5);
        assertEquals(5.0, result.get(2).getSecond().getDouble(0), 1e-5);
        assertEquals(4.0, result.get(2).getSecond().getDouble(1), 1e-5);
    }

    @Test
    public void testKNN_3() {

        double[] data = new double[]{2, 3};
        val result = kdTree.knn(Nd4j.createFromArray(data), 10.0);
        assertEquals(2.0, result.get(0).getSecond().getDouble(0), 1e-5);
        assertEquals(3.0, result.get(0).getSecond().getDouble(1), 1e-5);
        assertEquals(5.0, result.get(1).getSecond().getDouble(0), 1e-5);
        assertEquals(4.0, result.get(1).getSecond().getDouble(1), 1e-5);
        assertEquals(4.0, result.get(2).getSecond().getDouble(0), 1e-5);
        assertEquals(7.0, result.get(2).getSecond().getDouble(1), 1e-5);
        assertEquals(7.0, result.get(3).getSecond().getDouble(0), 1e-5);
        assertEquals(2.0, result.get(3).getSecond().getDouble(1), 1e-5);
        assertEquals(8.0, result.get(4).getSecond().getDouble(0), 1e-5);
        assertEquals(1.0, result.get(4).getSecond().getDouble(1), 1e-5);
        assertEquals(9.0, result.get(5).getSecond().getDouble(0), 1e-5);
        assertEquals(6.0, result.get(5).getSecond().getDouble(1), 1e-5);
    }


    @Test
    public void testKNN_4() {
        double[] data = new double[]{2, 3};
        val result = kdTree.knn(Nd4j.createFromArray(data), 5.0);
        assertEquals(2.0, result.get(0).getSecond().getDouble(0), 1e-5);
        assertEquals(3.0, result.get(0).getSecond().getDouble(1), 1e-5);
        assertEquals(5.0, result.get(1).getSecond().getDouble(0), 1e-5);
        assertEquals(4.0, result.get(1).getSecond().getDouble(1), 1e-5);
        assertEquals(4.0, result.get(2).getSecond().getDouble(0), 1e-5);
        assertEquals(7.0, result.get(2).getSecond().getDouble(1), 1e-5);
    }

    @Test
    public void testKNN_5() {
        double[] data = new double[]{2, 3};
        val result = kdTree.knn(Nd4j.createFromArray(data), 20.0);
        assertEquals(2.0, result.get(0).getSecond().getDouble(0), 1e-5);
        assertEquals(3.0, result.get(0).getSecond().getDouble(1), 1e-5);
        assertEquals(5.0, result.get(1).getSecond().getDouble(0), 1e-5);
        assertEquals(4.0, result.get(1).getSecond().getDouble(1), 1e-5);
        assertEquals(4.0, result.get(2).getSecond().getDouble(0), 1e-5);
        assertEquals(7.0, result.get(2).getSecond().getDouble(1), 1e-5);
        assertEquals(7.0, result.get(3).getSecond().getDouble(0), 1e-5);
        assertEquals(2.0, result.get(3).getSecond().getDouble(1), 1e-5);
        assertEquals(8.0, result.get(4).getSecond().getDouble(0), 1e-5);
        assertEquals(1.0, result.get(4).getSecond().getDouble(1), 1e-5);
        assertEquals(9.0, result.get(5).getSecond().getDouble(0), 1e-5);
        assertEquals(6.0, result.get(5).getSecond().getDouble(1), 1e-5);
    }

    @Test
    public void test_KNN_6() {
        double[] data = new double[]{4, 6};
        val result = kdTree.knn(Nd4j.createFromArray(data), 10.0);
        assertEquals(4.0, result.get(0).getSecond().getDouble(0), 1e-5);
        assertEquals(7.0, result.get(0).getSecond().getDouble(1), 1e-5);
        assertEquals(5.0, result.get(1).getSecond().getDouble(0), 1e-5);
        assertEquals(4.0, result.get(1).getSecond().getDouble(1), 1e-5);
        assertEquals(2.0, result.get(2).getSecond().getDouble(0), 1e-5);
        assertEquals(3.0, result.get(2).getSecond().getDouble(1), 1e-5);
        assertEquals(7.0, result.get(3).getSecond().getDouble(0), 1e-5);
        assertEquals(2.0, result.get(3).getSecond().getDouble(1), 1e-5);
        assertEquals(9.0, result.get(4).getSecond().getDouble(0), 1e-5);
        assertEquals(6.0, result.get(4).getSecond().getDouble(1), 1e-5);
        assertEquals(8.0, result.get(5).getSecond().getDouble(0), 1e-5);
        assertEquals(1.0, result.get(5).getSecond().getDouble(1), 1e-5);
    }

    @Test
    public void test_KNN_7() {
        double[] data = new double[]{4, 6};
        val result = kdTree.knn(Nd4j.createFromArray(data), 5.0);
        assertEquals(4.0, result.get(0).getSecond().getDouble(0), 1e-5);
        assertEquals(7.0, result.get(0).getSecond().getDouble(1), 1e-5);
        assertEquals(5.0, result.get(1).getSecond().getDouble(0), 1e-5);
        assertEquals(4.0, result.get(1).getSecond().getDouble(1), 1e-5);
        assertEquals(2.0, result.get(2).getSecond().getDouble(0), 1e-5);
        assertEquals(3.0, result.get(2).getSecond().getDouble(1), 1e-5);
        assertEquals(7.0, result.get(3).getSecond().getDouble(0), 1e-5);
        assertEquals(2.0, result.get(3).getSecond().getDouble(1), 1e-5);
        assertEquals(9.0, result.get(4).getSecond().getDouble(0), 1e-5);
        assertEquals(6.0, result.get(4).getSecond().getDouble(1), 1e-5);
    }

    @Test
    public void test_KNN_8() {
        double[] data = new double[]{4, 6};
        val result = kdTree.knn(Nd4j.createFromArray(data), 20.0);
        assertEquals(4.0, result.get(0).getSecond().getDouble(0), 1e-5);
        assertEquals(7.0, result.get(0).getSecond().getDouble(1), 1e-5);
        assertEquals(5.0, result.get(1).getSecond().getDouble(0), 1e-5);
        assertEquals(4.0, result.get(1).getSecond().getDouble(1), 1e-5);
        assertEquals(2.0, result.get(2).getSecond().getDouble(0), 1e-5);
        assertEquals(3.0, result.get(2).getSecond().getDouble(1), 1e-5);
        assertEquals(7.0, result.get(3).getSecond().getDouble(0), 1e-5);
        assertEquals(2.0, result.get(3).getSecond().getDouble(1), 1e-5);
        assertEquals(9.0, result.get(4).getSecond().getDouble(0), 1e-5);
        assertEquals(6.0, result.get(4).getSecond().getDouble(1), 1e-5);
        assertEquals(8.0, result.get(5).getSecond().getDouble(0), 1e-5);
        assertEquals(1.0, result.get(5).getSecond().getDouble(1), 1e-5);
    }

    @Test
    public void testNoDuplicates() {
        int N = 100;
        KDTree bigTree = new KDTree(2);

        List<INDArray> points = new ArrayList<>();
        for (int i = 0; i < N; ++i) {
            double[] data = new double[]{i, i};
            points.add(Nd4j.createFromArray(data));
        }

        for (int i = 0; i < N; ++i) {
            bigTree.insert(points.get(i));
        }

        assertEquals(N, bigTree.size());

        INDArray node = Nd4j.empty(DataType.DOUBLE);
        for (int i = 0; i < N; ++i) {
            node = bigTree.delete(node.isEmpty() ? points.get(i) : node);
        }

        assertEquals(0, bigTree.size());
    }

    @Test
    public void performanceTest() {
        int n = 2;
        int num = 1000;
        // make a KD-tree of dimension {#n}
        long start = System.currentTimeMillis();
        KDTree kdTree = new KDTree(n);
        INDArray inputArrray = Nd4j.linspace(DataType.DOUBLE, 0.0, 1.0, num*n).reshape(num, n);
        for (int  i = 0 ; i < num; ++i) {
            kdTree.insert(inputArrray.getRow(i));
        }

        long end = System.currentTimeMillis();
        Duration duration = new Duration(start, end);
        System.out.println("Elapsed time for tree construction " + duration.getMillis());

        List<Double> pt = new ArrayList(num);
        for (int k = 0; k < n; k++) {
            pt.add((double)(num / 2));
        }
        start = System.currentTimeMillis();
        List<Pair<Double, INDArray>> list = kdTree.knn(Nd4j.create(Nd4j.createBuffer(Doubles.toArray(pt))), 20.0);
        end = System.currentTimeMillis();
        duration = new Duration(start, end);
        long elapsed = end - start;
        System.out.println("Elapsed time for tree search " + duration.getMillis());
    }
}
