/*
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

package org.nd4j.linalg.dataset;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.BaseNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.FeatureUtil;

import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.junit.Assert.*;

public class DataSetTest extends BaseNd4jTest {
    public DataSetTest() {
    }

    public DataSetTest(String name) {
        super(name);
    }

    public DataSetTest(String name, Nd4jBackend backend) {
        super(name, backend);
    }

    public DataSetTest(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testViewIterator() {
        DataSetIterator iter = new ViewIterator(new IrisDataSetIterator(150,150).next(),10);
        assertTrue(iter.hasNext());
        int count = 0;
        while(iter.hasNext()) {
            DataSet next = iter.next();
            count++;
            assertArrayEquals(new int[]{10,4},next.getFeatureMatrix().shape());
        }

        assertFalse(iter.hasNext());
        assertEquals(15,count);
        iter.reset();
        assertTrue(iter.hasNext());

    }



    @Test
    public void testPersist() {
        DataSet iris = new IrisDataSetIterator(150,150).next();
        iris.save(new File("iris.bin"));
        DataSet load = new DataSet();
        load.load(new File("iris.bin"));
        new File("iris.bin").deleteOnExit();
        assertEquals(iris,load);
    }

    @Test
    public void testSplitTestAndTrain() throws Exception{
        INDArray labels = FeatureUtil.toOutcomeMatrix(new int[]{0,0,0,0,0,0,0,0},1);
        DataSet data = new DataSet(Nd4j.rand(8,1),labels);

        SplitTestAndTrain train = data.splitTestAndTrain(6, new Random(1));
        assertEquals(train.getTrain().getLabels().length(),6);

        SplitTestAndTrain train2 = data.splitTestAndTrain(6, new Random(1));
        assertEquals(getFailureMessage(),train.getTrain().getFeatureMatrix(), train2.getTrain().getFeatureMatrix());

        DataSet x0 = new IrisDataSetIterator(150,150).next();
        SplitTestAndTrain testAndTrain = x0.splitTestAndTrain(10);
        assertArrayEquals(new int[]{10,4},testAndTrain.getTrain().getFeatureMatrix().shape());
        assertEquals(x0.getFeatureMatrix().getRows(ArrayUtil.range(0, 10)), testAndTrain.getTrain().getFeatureMatrix());
        assertEquals(x0.getLabels().getRows(ArrayUtil.range(0,10)),testAndTrain.getTrain().getLabels());


    }

    @Test
    public void testLabelCounts() {
        DataSet x0 = new IrisDataSetIterator(150,150).next();
        assertEquals(getFailureMessage(),0,x0.get(0).outcome());
        assertEquals(getFailureMessage(),0,x0.get(1).outcome());
        assertEquals(getFailureMessage(),2, x0.get(149).outcome());
        Map<Integer,Double> counts = x0.labelCounts();
        assertEquals(getFailureMessage(),50,counts.get(0),1e-1);
        assertEquals(getFailureMessage(),50,counts.get(1),1e-1);
        assertEquals(getFailureMessage(),50,counts.get(2),1e-1);

    }


    @Override
    public char ordering() {
        return 'f';
    }
}
