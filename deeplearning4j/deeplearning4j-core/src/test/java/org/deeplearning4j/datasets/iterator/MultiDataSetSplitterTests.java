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

package org.deeplearning4j.datasets.iterator;

import lombok.val;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.tools.DataSetGenerator;
import org.deeplearning4j.datasets.iterator.tools.MultiDataSetGenerator;
import org.junit.Test;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.exception.ND4JIllegalStateException;

import java.util.List;
import java.util.Random;

import static org.junit.Assert.*;

/**
 *
 * @author raver119@gmail.com
 */
public class MultiDataSetSplitterTests extends BaseDL4JTest {

    @Test
    public void testSplitter_1() throws Exception {
        val back = new MultiDataSetGenerator(1000, new int[]{32, 100}, new int[]{32, 5});

        val splitter = new MultiDataSetIteratorSplitter(back, 1000, 0.7);

        val train = splitter.getTrainIterator();
        val test = splitter.getTestIterator();
        val numEpochs = 10;

        int gcntTrain = 0;
        int gcntTest = 0;
        int global = 0;
        // emulating epochs here
        for (int e = 0; e < numEpochs; e++){
            int cnt = 0;
            while (train.hasNext()) {
                val data = train.next().getFeatures(0);

                assertEquals("Train failed on iteration " + cnt + "; epoch: " + e, (float) cnt++, data.getFloat(0), 1e-5);
                gcntTrain++;
                global++;
            }

            train.reset();


            while (test.hasNext()) {
                val data = test.next().getFeatures(0);
                assertEquals("Train failed on iteration " + cnt + "; epoch: " + e, (float) cnt++, data.getFloat(0), 1e-5);
                gcntTest++;
                global++;
            }

            test.reset();
        }

        assertEquals(1000 * numEpochs, global);
    }


    @Test
    public void testSplitter_2() throws Exception {
        val back = new MultiDataSetGenerator(1000, new int[]{32, 100}, new int[]{32, 5});

        val splitter = new MultiDataSetIteratorSplitter(back, 1000, 0.7);

        val train = splitter.getTrainIterator();
        val test = splitter.getTestIterator();
        val numEpochs = 10;

        int gcntTrain = 0;
        int gcntTest = 0;
        int global = 0;
        // emulating epochs here
        for (int e = 0; e < numEpochs; e++){
            int cnt = 0;
            while (train.hasNext()) {
                val data = train.next().getFeatures(0);

                assertEquals("Train failed on iteration " + cnt + "; epoch: " + e, (float) cnt++, data.getFloat(0), 1e-5);
                gcntTrain++;
                global++;
            }

            train.reset();

            if (e % 2 == 0)
                while (test.hasNext()) {
                    val data = test.next().getFeatures(0);
                    assertEquals("Train failed on iteration " + cnt + "; epoch: " + e, (float) cnt++, data.getFloat(0), 1e-5);
                    gcntTest++;
                    global++;
                }
        }

        assertEquals(700 * numEpochs + (300 * numEpochs / 2), global);
    }

    @Test(expected = ND4JIllegalStateException.class)
    public void testSplitter_3() throws Exception {
        val back = new MultiDataSetGenerator(1000, new int[]{32, 100}, new int[]{32, 5});

        val splitter = new MultiDataSetIteratorSplitter(back, 1000, 0.7);

        val train = splitter.getTrainIterator();
        val test = splitter.getTestIterator();
        val numEpochs = 10;

        int gcntTrain = 0;
        int gcntTest = 0;
        int global = 0;
        // emulating epochs here
        for (int e = 0; e < numEpochs; e++){
            int cnt = 0;
            while (train.hasNext()) {
                val data = train.next().getFeatures(0);

                assertEquals("Train failed on iteration " + cnt + "; epoch: " + e, (float) cnt++, data.getFloat(0), 1e-5);
                gcntTrain++;
                global++;
            }

            train.reset();


            while (test.hasNext()) {
                val data = test.next().getFeatures(0);
                assertEquals("Train failed on iteration " + cnt + "; epoch: " + e, (float) cnt++, data.getFloat(0), 1e-5);
                gcntTest++;
                global++;
            }

            // shifting underlying iterator by one
            train.hasNext();
            back.shift();
        }

        assertEquals(1000 * numEpochs, global);
    }

    @Test
    public void testMultiSplitter_1() {
        val back = new MultiDataSetGenerator(1000, new int[]{32, 100}, new int[]{32, 5});

        // we're going to mimic train+test+validation split
        val splitter = new MultiDataSetIteratorSplitter(back, new int[]{800, 100, 100});

        assertEquals(3, splitter.getIterators().size());

        val trainIter = splitter.getIterators().get(0);
        val testIter = splitter.getIterators().get(1);
        val validationIter = splitter.getIterators().get(2);

        // we're going to have multiple epochs
        int numEpochs = 10;
        for (int e = 0; e < numEpochs; e++) {
            int globalIter = 0;
            trainIter.reset();
            testIter.reset();
            validationIter.reset();

            boolean trained = false;
            while (trainIter.hasNext()) {
                trained = true;
                val ds = trainIter.next();
                assertNotNull(ds);

                for (int i = 0; i < ds.getFeatures().length; ++i) {
                    assertEquals("Failed at iteration [" + globalIter + "]", (double) globalIter, ds.getFeatures()[i].getDouble(0), 1e-5f);
                }
                globalIter++;
            }
            assertTrue("Failed at epoch [" + e + "]", trained);
            assertEquals(800, globalIter);


            // test set is used every epoch
            boolean tested = false;
            //testIter.reset();
            while (testIter.hasNext()) {
                tested = true;
                val ds = testIter.next();
                assertNotNull(ds);

                for (int i = 0; i < ds.getFeatures().length; ++i) {
                    assertEquals("Failed at iteration [" + globalIter + "]", (double) globalIter, ds.getFeatures()[i].getDouble(0), 1e-5f);
                }
                globalIter++;
            }
            assertTrue("Failed at epoch [" + e + "]", tested);
            assertEquals(900, globalIter);

            // validation set is used every 5 epochs
            if (e % 5 == 0) {
                boolean validated = false;
                //validationIter.reset();
                while (validationIter.hasNext()) {
                    validated = true;
                    val ds = validationIter.next();
                    assertNotNull(ds);

                    for (int i = 0; i < ds.getFeatures().length; ++i) {
                        assertEquals("Failed at iteration [" + globalIter + "]", (double) globalIter, ds.getFeatures()[i].getDouble(0), 1e-5f);
                    }
                    globalIter++;
                }
                assertTrue("Failed at epoch [" + e + "]", validated);
            }

            // all 3 iterators have exactly 1000 elements combined
            if (e % 5 == 0)
                assertEquals(1000, globalIter);
            else
                assertEquals(900, globalIter);
            trainIter.reset();
        }
    }

    @Test
    public void testSplitter_5() {
        val back = new MultiDataSetGenerator(1000, new int[]{32, 100}, new int[]{32, 5});

        val splitter = new MultiDataSetIteratorSplitter(back, new int[]{900, 100});

        List<MultiDataSetIterator> iteratorList = splitter.getIterators();
        val numEpochs = 10;

        int global = 0;
        // emulating epochs here
        for (int e = 0; e < numEpochs; e++) {
            int iterNo = 0;
            int perEpoch = 0;
            for (val partIterator : iteratorList) {
                partIterator.reset();
                while (partIterator.hasNext()) {
                    int cnt = 0;
                    val data = partIterator.next().getFeatures();

                    for (int i = 0; i < data.length; ++i) {
                        assertEquals("Train failed on iteration " + cnt + "; epoch: " + e,
                                (float) perEpoch, data[i].getFloat(0), 1e-5);
                    }
                    //gcntTrain++;
                    global++;
                    cnt++;
                    ++perEpoch;
                }
                ++iterNo;
            }
        }

        assertEquals(1000 * numEpochs, global);
    }

    @Test
    public void testSplitter_6() {
        val back = new MultiDataSetGenerator(1000, new int[]{32, 100}, new int[]{32, 5});

        // we're going to mimic train+test+validation split
        val splitter = new MultiDataSetIteratorSplitter(back, new int[]{800, 100, 100});

        assertEquals(3, splitter.getIterators().size());

        val trainIter = splitter.getIterators().get(0);
        val testIter = splitter.getIterators().get(1);
        val validationIter = splitter.getIterators().get(2);

        // we're going to have multiple epochs
        int numEpochs = 10;
        for (int e = 0; e < numEpochs; e++) {
            int globalIter = 0;
            trainIter.reset();
            testIter.reset();
            validationIter.reset();

            boolean trained = false;
            while (trainIter.hasNext()) {
                trained = true;
                val ds = trainIter.next();
                assertNotNull(ds);

                for (int i = 0; i < ds.getFeatures().length; ++i) {
                    assertEquals("Failed at iteration [" + globalIter + "]", (double) globalIter,
                            ds.getFeatures()[i].getDouble(0), 1e-5f);
                }
                globalIter++;
            }
            assertTrue("Failed at epoch [" + e + "]", trained);
            assertEquals(800, globalIter);


            // test set is used every epoch
            boolean tested = false;
            //testIter.reset();
            while (testIter.hasNext()) {
                tested = true;
                val ds = testIter.next();
                assertNotNull(ds);
                for (int i = 0; i < ds.getFeatures().length; ++i) {
                    assertEquals("Failed at iteration [" + globalIter + "]", (double) globalIter, ds.getFeatures()[i].getDouble(0), 1e-5f);
                }
                globalIter++;
            }
            assertTrue("Failed at epoch [" + e + "]", tested);
            assertEquals(900, globalIter);

            // validation set is used every 5 epochs
            if (e % 5 == 0) {
                boolean validated = false;
                //validationIter.reset();
                while (validationIter.hasNext()) {
                    validated = true;
                    val ds = validationIter.next();
                    assertNotNull(ds);

                    for (int i = 0; i < ds.getFeatures().length; ++i) {
                        assertEquals("Failed at iteration [" + globalIter + "]", (double) globalIter,
                                ds.getFeatures()[i].getDouble(0), 1e-5f);
                    }
                    globalIter++;
                }
                assertTrue("Failed at epoch [" + e + "]", validated);
            }

            // all 3 iterators have exactly 1000 elements combined
            if (e % 5 == 0)
                assertEquals(1000, globalIter);
            else
                assertEquals(900, globalIter);
            trainIter.reset();
        }
    }

    @Test
    public void testUnorderedSplitter_1() {
        val back = new MultiDataSetGenerator(1000, new int[]{32, 100}, new int[]{32, 5});

        val splitter = new MultiDataSetIteratorSplitter(back, new int[]{500, 500});

        List<MultiDataSetIterator> iteratorList = splitter.getIterators();
        val numEpochs = 10;

        int global = 0;
        // emulating epochs here
        for (int e = 0; e < numEpochs; e++) {

            // Get data from second part, then rewind for the first one.
            int cnt = 0;
            int partNumber = 1;
            while (iteratorList.get(partNumber).hasNext()) {
                int farCnt = (1000 / 2) * (partNumber) + cnt;
                val data = iteratorList.get(partNumber).next().getFeatures();
                for (int i = 0; i < data.length; ++i) {
                    assertEquals("Train failed on iteration " + cnt + "; epoch: " + e, (float) farCnt, data[i].getFloat(0), 1e-5);
                }
                cnt++;
                global++;
            }
            iteratorList.get(partNumber).reset();
            partNumber = 0;
            cnt = 0;
            while (iteratorList.get(0).hasNext()) {
                val data = iteratorList.get(0).next().getFeatures();
                for (int i = 0; i < data.length; ++i) {
                    assertEquals("Train failed on iteration " + cnt + "; epoch: " + e, (float) cnt++,
                            data[i].getFloat(0), 1e-5);
                }
                global++;
            }
        }
    }

    @Test
    public void testUnorderedSplitter_2() {
        val back = new MultiDataSetGenerator(1000, new int[]{32, 100}, new int[]{32, 5});

        val splitter = new MultiDataSetIteratorSplitter(back, new int[]{2});

        List<MultiDataSetIterator> iteratorList = splitter.getIterators();

        for (int partNumber = 0 ; partNumber < iteratorList.size(); ++partNumber) {
            int cnt = 0;
            while (iteratorList.get(partNumber).hasNext()) {
                val data = iteratorList.get(partNumber).next().getFeatures();
                for (int i = 0; i < data.length; ++i) {
                    assertEquals("Train failed on iteration " + cnt, (float) (500 * partNumber + cnt), data[i].getFloat(0), 1e-5);
                }
                cnt++;
            }
        }
    }

    @Test
    public void testUnorderedSplitter_3() {
        val back = new MultiDataSetGenerator(1000, new int[]{32, 100}, new int[]{32, 5});

        val splitter = new MultiDataSetIteratorSplitter(back, new int[]{10});

        List<MultiDataSetIterator> iteratorList = splitter.getIterators();
        Random random = new Random();
        int[] indexes = new int[iteratorList.size()];
        for (int i = 0; i < indexes.length; ++i) {
            indexes[i] = random.nextInt(iteratorList.size());
        }

        for (int partNumber : indexes) {
            int cnt = 0;
            while (iteratorList.get(partNumber).hasNext()) {
                val data = iteratorList.get(partNumber).next().getFeatures();
                for (int i = 0; i < data.length; ++i) {
                    assertEquals("Train failed on iteration " + cnt, (float) (500 * partNumber + cnt),
                            data[i].getFloat(0), 1e-5);
                }
                cnt++;
            }
        }
    }

    @Test
    public void testUnorderedSplitter_4() {
        val back = new MultiDataSetGenerator(1000, new int[]{32, 100}, new int[]{32, 5});

        // we're going to mimic train+test+validation split
        val splitter = new MultiDataSetIteratorSplitter(back, new int[]{80, 10, 5});

        assertEquals(3, splitter.getIterators().size());

        val trainIter = splitter.getIterators().get(0);  // 0..79
        val testIter = splitter.getIterators().get(1);   // 80 ..89
        val validationIter = splitter.getIterators().get(2); // 90..94

        // we're skipping train/test and go for validation first. we're that crazy, right.
        int valCnt = 0;
        while (validationIter.hasNext()) {
            val ds = validationIter.next();
            assertNotNull(ds);
            for (int i = 0; i < ds.getFeatures().length; ++i) {
                assertEquals("Validation failed on iteration " + valCnt, (float) valCnt + 90,
                        ds.getFeatures()[i].getFloat(0), 1e-5);
            }
            valCnt++;
        }
        assertEquals(5, valCnt);
    }
}
