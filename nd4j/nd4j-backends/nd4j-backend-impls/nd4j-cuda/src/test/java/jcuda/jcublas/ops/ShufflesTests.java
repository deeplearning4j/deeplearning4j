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

package jcuda.jcublas.ops;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
public class ShufflesTests {

    @Test
    public void testSimpleShuffle1() {
        INDArray array = Nd4j.zeros(10, 10);
        for (int x = 0; x < 10; x++) {
            array.getRow(x).assign(x);
        }

        System.out.println(array);

        OrderScanner2D scanner = new OrderScanner2D(array);

        assertArrayEquals(new float[]{0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f}, scanner.getMap(), 0.01f);

        System.out.println();

        Nd4j.shuffle(array, 1);

        System.out.println(array);

        ArrayUtil.argMin(new int[]{});

        assertTrue(scanner.compareRow(array));
    }

    @Test
    public void testSimpleShuffle2() {
        INDArray array = Nd4j.zeros(10, 10);
        for (int x = 0; x < 10; x++) {
            array.getColumn(x).assign(x);
        }

        System.out.println(array);

        OrderScanner2D scanner = new OrderScanner2D(array);

        assertArrayEquals(new float[]{0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f}, scanner.getMap(), 0.01f);

        System.out.println();

        Nd4j.shuffle(array, 0);

        System.out.println(array);

        assertTrue(scanner.compareColumn(array));
    }


    @Test
    public void testSymmetricShuffle1() {
        INDArray features = Nd4j.zeros(10, 10);
        INDArray labels = Nd4j.zeros(10, 3);
        for (int x = 0; x < 10; x++) {
            features.getRow(x).assign(x);
            labels.getRow(x).assign(x);
        }

        System.out.println(features);

        OrderScanner2D scanner = new OrderScanner2D(features);

        assertArrayEquals(new float[]{0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f}, scanner.getMap(), 0.01f);

        System.out.println();

        List<INDArray> list = new ArrayList<>();
        list.add(features);
        list.add(labels);

        Nd4j.shuffle(list, 1);

        System.out.println(features);

        System.out.println();

        System.out.println(labels);

        ArrayUtil.argMin(new int[]{});

        assertTrue(scanner.compareRow(features));

        for (int x = 0; x < 10; x++) {
            double val = features.getRow(x).getDouble(0);
            INDArray row = labels.getRow(x);

            for (int y = 0; y < row.length(); y++ ) {
                assertEquals(val, row.getDouble(y), 0.001);
            }
        }
    }

    @Test
    public void testSymmetricShuffle2() throws Exception {
        INDArray features = Nd4j.zeros(10, 10, 20);
        INDArray labels = Nd4j.zeros(10, 10, 3);

        for (int x = 0; x < 10; x++) {
            features.slice(x).assign(x);
            labels.slice(x).assign(x);
        }

        System.out.println(features);

        OrderScanner3D scannerFeatures = new OrderScanner3D(features);
        OrderScanner3D scannerLabels = new OrderScanner3D(labels);

        System.out.println();

        List<INDArray> list = new ArrayList<>();
        list.add(features);
        list.add(labels);

        Nd4j.shuffle(list, 1, 2);

        System.out.println(features);

        System.out.println("------------------");

        System.out.println(labels);

        assertTrue(scannerFeatures.compareSlice(features));
        assertTrue(scannerLabels.compareSlice(labels));

        for (int x = 0; x < 10; x++) {
            double val = features.slice(x).getDouble(0);
            INDArray row = labels.slice(x);

            for (int y = 0; y < row.length(); y++ ) {
                assertEquals(val, row.getDouble(y), 0.001);
            }
        }
    }

    @Test
    public void testSymmetricShuffle3() throws Exception {
        INDArray features = Nd4j.zeros(10, 10, 20);
        INDArray featuresMask = Nd4j.zeros(10, 20);
        INDArray labels = Nd4j.zeros(10, 10, 3);
        INDArray labelsMask = Nd4j.zeros(10, 3);

        for (int x = 0; x < 10; x++) {
            features.slice(x).assign(x);
            featuresMask.slice(x).assign(x);
            labels.slice(x).assign(x);
            labelsMask.slice(x).assign(x);
        }

        OrderScanner3D scannerFeatures = new OrderScanner3D(features);
        OrderScanner3D scannerLabels = new OrderScanner3D(labels);
        OrderScanner3D scannerFeaturesMask = new OrderScanner3D(featuresMask);
        OrderScanner3D scannerLabelsMask = new OrderScanner3D(labelsMask);


        List<INDArray> arrays = new ArrayList<>();
        arrays.add(features);
        arrays.add(labels);
        arrays.add(featuresMask);
        arrays.add(labelsMask);

        List<int[]> dimensions = new ArrayList<>();
        dimensions.add(ArrayUtil.range(1,features.rank()));
        dimensions.add(ArrayUtil.range(1,labels.rank()));
        dimensions.add(ArrayUtil.range(1,featuresMask.rank()));
        dimensions.add(ArrayUtil.range(1,labelsMask.rank()));

        Nd4j.shuffle(arrays, new Random(), dimensions);

        assertTrue(scannerFeatures.compareSlice(features));
        assertTrue(scannerLabels.compareSlice(labels));
        assertTrue(scannerFeaturesMask.compareSlice(featuresMask));
        assertTrue(scannerLabelsMask.compareSlice(labelsMask));


        for (int x = 0; x < 10; x++) {
            double val = features.slice(x).getDouble(0);
            INDArray sliceLabels = labels.slice(x);
            INDArray sliceLabelsMask = labelsMask.slice(x);
            INDArray sliceFeaturesMask = featuresMask.slice(x);

            for (int y = 0; y < sliceLabels.length(); y++ ) {
                assertEquals(val, sliceLabels.getDouble(y), 0.001);
            }

            for (int y = 0; y < sliceLabelsMask.length(); y++ ) {
                assertEquals(val, sliceLabelsMask.getDouble(y), 0.001);
            }

            for (int y = 0; y < sliceFeaturesMask.length(); y++ ) {
                assertEquals(val, sliceFeaturesMask.getDouble(y), 0.001);
            }
        }
    }


    @Test
    public void testSymmetricShuffle4() throws Exception {
        INDArray features = Nd4j.zeros(10, 3, 4, 2);
        INDArray labels = Nd4j.zeros(10, 5);

        for (int x = 0; x < 10; x++) {
            features.slice(x).assign(x);
            labels.slice(x).assign(x);
        }

        OrderScanner3D scannerFeatures = new OrderScanner3D(features);
        OrderScanner3D scannerLabels = new OrderScanner3D(labels);

        System.out.println(features);

        System.out.println();

        DataSet ds = new DataSet(features, labels);
        ds.shuffle();

        System.out.println(features);

        System.out.println("------------------");

        assertTrue(scannerFeatures.compareSlice(features));
        assertTrue(scannerLabels.compareSlice(labels));

        for (int x = 0; x < 10; x++) {
            double val = features.slice(x).getDouble(0);
            INDArray row = labels.slice(x);

            for (int y = 0; y < row.length(); y++ ) {
                assertEquals(val, row.getDouble(y), 0.001);
            }
        }
    }

    @Test
    public void testSymmetricShuffle4F() throws Exception {
        INDArray features = Nd4j.create(new int[] {10, 3, 4, 2} , 'f');
        INDArray labels = Nd4j.create(new int[] {10, 5}, 'f');

        for (int x = 0; x < 10; x++) {
            features.slice(x).assign(x);
            labels.slice(x).assign(x);
        }


        System.out.println("features.length: " + features.length());

        System.out.println(labels);

        System.out.println();

        DataSet ds = new DataSet(features, labels);
        ds.shuffle();

        System.out.println(labels);

        System.out.println("------------------");


        for (int x = 0; x < 10; x++) {
            double val = features.slice(x).getDouble(0);
            INDArray row = labels.slice(x);

            for (int y = 0; y < row.length(); y++ ) {
                assertEquals(val, row.getDouble(y), 0.001);
            }
        }
    }

    @Test
    public void testHalfVectors() throws Exception {
        int[] array = ArrayUtil.buildHalfVector(new Random(12), 11);

        System.out.println("HalfVec: " + Arrays.toString(array));
    }


    public static class OrderScanner3D {
        private float[] map;
        public OrderScanner3D(INDArray data) {
            map = measureState(data);
        }

        public float[] measureState(INDArray data) {
            // for 3D we save 0 element for each slice.
            float[] result = new float[data.shape()[0]];

            for (int x = 0; x < data.shape()[0]; x++) {
                result[x] = data.slice(x).getFloat(0);
            }

            return result;
        }

        public boolean compareSlice(INDArray data) {
            float[] newMap = measureState(data);

            if (newMap.length != map.length) {
                System.out.println("Different map lengths");
                return false;
            }

            if (Arrays.equals(map, newMap)) {
                System.out.println("Maps are equal");
                return false;
            }

            for (int x = 0; x < data.shape()[0]; x++) {
                INDArray slice = data.slice(x);

                for (int y = 0; y < slice.length(); y++) {
                    if (Math.abs(slice.getFloat(y) - newMap[x]) > Nd4j.EPS_THRESHOLD) {
                        System.out.print("Different data in a row");
                        return false;
                    }
                }
            }


            return true;
        }
    }


    public static class OrderScanner2D {
        private float[] map;

        public OrderScanner2D(INDArray data) {
            map = measureState(data);
        }

        public float[] measureState(INDArray data) {
            float[] result = new float[data.rows()];

            for (int x = 0; x < data.rows(); x++) {
                result[x] = data.getRow(x).getFloat(0);
            }

            return result;
        }

        public boolean compareRow(INDArray newData) {
            float[] newMap = measureState(newData);

            if (newMap.length != map.length) {
                System.out.println("Different map lengths");
                return false;
            }

            if (Arrays.equals(map, newMap)) {
                System.out.println("Maps are equal");
                return false;
            }

            for (int x = 0; x < newData.rows(); x++) {
                INDArray row = newData.getRow(x);
                for (int y = 0; y < row.lengthLong(); y++ ) {
                    if (Math.abs(row.getFloat(y) - newMap[x]) > Nd4j.EPS_THRESHOLD) {
                        System.out.print("Different data in a row");
                        return false;
                    }
                }
            }

            return true;
        }

        public boolean compareColumn(INDArray newData) {
            float[] newMap = measureState(newData);

            if (newMap.length != map.length) {
                System.out.println("Different map lengths");
                return false;
            }

            if (Arrays.equals(map, newMap)) {
                System.out.println("Maps are equal");
                return false;
            }

            for (int x = 0; x < newData.rows(); x++) {
                INDArray column = newData.getColumn(x);
                double val = column.getDouble(0);
                for (int y = 0; y < column.lengthLong(); y++ ) {
                    if (Math.abs(column.getFloat(y) - val) > Nd4j.EPS_THRESHOLD) {
                        System.out.print("Different data in a column: " + column.getFloat(y));
                        return false;
                    }
                }
            }

            return true;
        }

        public float[] getMap() {
            return map;
        }
    }
}
