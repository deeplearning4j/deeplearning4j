/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.datasets.iterator;

import lombok.val;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.tools.DataSetGenerator;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

@NativeTag
@Tag(TagNames.FILE_IO)
public class DataSetSplitterTests extends BaseDL4JTest {
    @Test
    public void testSplitter_1() throws Exception {
        val back = new DataSetGenerator(1000, new int[]{32, 100}, new int[]{32, 5});

        val splitter = new DataSetIteratorSplitter(back, 1000, 0.7);

        val train = splitter.getTrainIterator();
        val test = splitter.getTestIterator();
        val numEpochs = 10;

        int gcntTrain = 0;
        int gcntTest = 0;
        int global = 0;
        // emulating epochs here
        for (int e = 0; e < numEpochs; e++) {
            int cnt = 0;
            while (train.hasNext()) {
                val data = train.next().getFeatures();

                assertEquals( (float) cnt++, data.getFloat(0), 1e-5,"Train failed on iteration " + cnt + "; epoch: " + e);
                gcntTrain++;
                global++;
            }

            train.reset();


            while (test.hasNext()) {
                val data = test.next().getFeatures();
                assertEquals( (float) cnt++, data.getFloat(0), 1e-5,"Train failed on iteration " + cnt + "; epoch: " + e);
                gcntTest++;
                global++;
            }

            test.reset();
        }

        assertEquals(1000 * numEpochs, global);
    }

    @Test
    public void testSplitter_2() throws Exception {
        val back = new DataSetGenerator(1000, new int[]{32, 100}, new int[]{32, 5});

        val splitter = new DataSetIteratorSplitter(back, 1000, 0.7);

        val train = splitter.getTrainIterator();
        val test = splitter.getTestIterator();
        val numEpochs = 10;

        int gcntTrain = 0;
        int gcntTest = 0;
        int global = 0;
        // emulating epochs here
        for (int e = 0; e < numEpochs; e++) {
            int cnt = 0;
            while (train.hasNext()) {
                val data = train.next().getFeatures();

                assertEquals((float) cnt++, data.getFloat(0), 1e-5,"Train failed on iteration " + cnt + "; epoch: " + e);
                gcntTrain++;
                global++;
            }

            train.reset();

            if (e % 2 == 0)
                while (test.hasNext()) {
                    val data = test.next().getFeatures();
                    assertEquals( (float) cnt++, data.getFloat(0), 1e-5,"Train failed on iteration " + cnt + "; epoch: " + e);
                    gcntTest++;
                    global++;
                }
        }

        assertEquals(700 * numEpochs + (300 * numEpochs / 2), global);
    }

    @Test()
    public void testSplitter_3() throws Exception {
       assertThrows(ND4JIllegalStateException.class, () -> {
           val back = new DataSetGenerator(1000, new int[]{32, 100}, new int[]{32, 5});

           val splitter = new DataSetIteratorSplitter(back, 1000, 0.7);

           val train = splitter.getTrainIterator();
           val test = splitter.getTestIterator();
           val numEpochs = 10;

           int gcntTrain = 0;
           int gcntTest = 0;
           int global = 0;
           // emulating epochs here
           for (int e = 0; e < numEpochs; e++) {
               int cnt = 0;
               while (train.hasNext()) {
                   val data = train.next().getFeatures();

                   assertEquals((float) cnt++, data.getFloat(0), 1e-5,"Train failed on iteration " + cnt + "; epoch: " + e);
                   gcntTrain++;
                   global++;
               }

               train.reset();


               while (test.hasNext()) {
                   val data = test.next().getFeatures();
                   assertEquals((float) cnt++, data.getFloat(0), 1e-5,"Train failed on iteration " + cnt + "; epoch: " + e);
                   gcntTest++;
                   global++;
               }

               // shifting underlying iterator by one
               train.hasNext();
               back.shift();
           }

           assertEquals(1000 * numEpochs, global);
       });


    }

    @Test
    public void testSplitter_4() {
        val back = new DataSetGenerator(1000, new int[]{32, 100}, new int[]{32, 5});

        val splitter = new DataSetIteratorSplitter(back, 1000, new double[]{0.5, 0.3, 0.2});
        List<DataSetIterator> iteratorList = splitter.getIterators();
        val numEpochs = 10;
        int global = 0;
        // emulating epochs here
        for (int e = 0; e < numEpochs; e++) {
            int iterNo = 0;
            int perEpoch = 0;
            for (val partIterator : iteratorList) {
                int cnt = 0;
                partIterator.reset();
                while (partIterator.hasNext()) {
                    val data = partIterator.next().getFeatures();
                    assertEquals((float) perEpoch, data.getFloat(0), 1e-5,"Train failed on iteration " + cnt + "; epoch: " + e);
                    //gcntTrain++;
                    global++;
                    cnt++;
                    ++perEpoch;
                }
                ++iterNo;
            }
        }

        assertEquals(1000* numEpochs, global);
    }

    @Test
    public void testSplitter_5() {
        val back = new DataSetGenerator(1000, new int[]{32, 100}, new int[]{32, 5});

        val splitter = new DataSetIteratorSplitter(back, new int[]{900, 100});

        List<DataSetIterator> iteratorList = splitter.getIterators();
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

                    assertEquals((float) perEpoch, data.getFloat(0), 1e-5,"Train failed on iteration " + cnt + "; epoch: " + e);
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
        val back = new DataSetGenerator(1000, new int[]{32, 100}, new int[]{32, 5});

        // we're going to mimic train+test+validation split
        val splitter = new DataSetIteratorSplitter(back, new int[]{800, 100, 100});

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

                assertEquals(globalIter, ds.getFeatures().getDouble(0), 1e-5f,"Failed at iteration [" + globalIter + "]");
                globalIter++;
            }
            assertTrue(trained,"Failed at epoch [" + e + "]");
            assertEquals(800, globalIter);


            // test set is used every epoch
            boolean tested = false;
            //testIter.reset();
            while (testIter.hasNext()) {
                tested = true;
                val ds = testIter.next();
                assertNotNull(ds);

                assertEquals(globalIter, ds.getFeatures().getDouble(0), 1e-5f,"Failed at iteration [" + globalIter + "]");
                globalIter++;
            }
            assertTrue(tested,"Failed at epoch [" + e + "]");
            assertEquals(900, globalIter);

            // validation set is used every 5 epochs
            if (e % 5 == 0) {
                boolean validated = false;
                //validationIter.reset();
                while (validationIter.hasNext()) {
                    validated = true;
                    val ds = validationIter.next();
                    assertNotNull(ds);

                    assertEquals(globalIter, ds.getFeatures().getDouble(0), 1e-5f,"Failed at iteration [" + globalIter + "]");
                    globalIter++;
                }
                assertTrue(validated,"Failed at epoch [" + e + "]");
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
        val back = new DataSetGenerator(1000, new int[]{32, 100}, new int[]{32, 5});

        val splitter = new DataSetIteratorSplitter(back, new int[]{500, 500});

        List<DataSetIterator> iteratorList = splitter.getIterators();
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

                assertEquals((float) farCnt, data.getFloat(0), 1e-5,"Train failed on iteration " + cnt + "; epoch: " + e);
                cnt++;
                global++;
            }
            iteratorList.get(partNumber).reset();
            partNumber = 0;
            cnt = 0;
            while (iteratorList.get(0).hasNext()) {
                val data = iteratorList.get(0).next().getFeatures();

                assertEquals((float) cnt++, data.getFloat(0), 1e-5,"Train failed on iteration " + cnt + "; epoch: " + e);
                global++;
            }
        }
    }

    @Test
    public void testUnorderedSplitter_2() {
        val back = new DataSetGenerator(1000, new int[]{32, 100}, new int[]{32, 5});

        val splitter = new DataSetIteratorSplitter(back, new int[]{2});

        List<DataSetIterator> iteratorList = splitter.getIterators();

        for (int partNumber = 0 ; partNumber < iteratorList.size(); ++partNumber) {
            int cnt = 0;
            while (iteratorList.get(partNumber).hasNext()) {
                val data = iteratorList.get(partNumber).next().getFeatures();

                assertEquals( (float) (500*partNumber + cnt), data.getFloat(0), 1e-5,"Train failed on iteration " + cnt);
                cnt++;
            }
        }
    }

    @Test
    public void testUnorderedSplitter_3() {
        val back = new DataSetGenerator(1000, new int[]{32, 100}, new int[]{32, 5});

        val splitter = new DataSetIteratorSplitter(back, new int[]{10});

        List<DataSetIterator> iteratorList = splitter.getIterators();
        Random random = new Random();
        int[] indexes = new int[iteratorList.size()];
        for (int i = 0; i < indexes.length; ++i) {
            indexes[i] = random.nextInt(iteratorList.size());
        }

        for (int partNumber : indexes) {
            int cnt = 0;
            while (iteratorList.get(partNumber).hasNext()) {
                val data = iteratorList.get(partNumber).next().getFeatures();

                assertEquals( (float) (500*partNumber + cnt), data.getFloat(0), 1e-5,"Train failed on iteration " + cnt);
                cnt++;
            }
        }
    }

    @Test
    public void testUnorderedSplitter_4() {
        val back = new DataSetGenerator(1000, new int[]{32, 100}, new int[]{32, 5});

        // we're going to mimic train+test+validation split
        val splitter = new DataSetIteratorSplitter(back, new int[]{80, 10, 5});

        assertEquals(3, splitter.getIterators().size());

        val trainIter = splitter.getIterators().get(0);  // 0..79
        val testIter = splitter.getIterators().get(1);   // 80 ..89
        val validationIter = splitter.getIterators().get(2); // 90..94

        // we're skipping train/test and go for validation first. we're that crazy, right.
        int valCnt = 0;
        while (validationIter.hasNext()) {
            val ds = validationIter.next();
            assertNotNull(ds);

            assertEquals((float) valCnt + 90, ds.getFeatures().getFloat(0), 1e-5,"Validation failed on iteration " + valCnt);
            valCnt++;
        }
        assertEquals(5, valCnt);
    }
}
