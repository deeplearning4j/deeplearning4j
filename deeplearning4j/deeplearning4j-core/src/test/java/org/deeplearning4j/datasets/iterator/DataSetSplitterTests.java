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
import org.junit.Test;
import org.nd4j.linalg.exception.ND4JIllegalStateException;

import static org.junit.Assert.assertEquals;

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
        for (int e = 0; e < numEpochs; e++){
            int cnt = 0;
            while (train.hasNext()) {
                val data = train.next().getFeatures();

                assertEquals("Train failed on iteration " + cnt + "; epoch: " + e, (float) cnt++, data.getFloat(0), 1e-5);
                gcntTrain++;
                global++;
            }

            train.reset();


            while (test.hasNext()) {
                val data = test.next().getFeatures();
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
        val back = new DataSetGenerator(1000, new int[]{32, 100}, new int[]{32, 5});

        val splitter = new DataSetIteratorSplitter(back, 1000, 0.7);

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
                val data = train.next().getFeatures();

                assertEquals("Train failed on iteration " + cnt + "; epoch: " + e, (float) cnt++, data.getFloat(0), 1e-5);
                gcntTrain++;
                global++;
            }

            train.reset();

            if (e % 2 == 0)
                while (test.hasNext()) {
                    val data = test.next().getFeatures();
                    assertEquals("Train failed on iteration " + cnt + "; epoch: " + e, (float) cnt++, data.getFloat(0), 1e-5);
                    gcntTest++;
                    global++;
                }
        }

        assertEquals(700 * numEpochs + (300 * numEpochs / 2), global);
    }

    @Test(expected = ND4JIllegalStateException.class)
    public void testSplitter_3() throws Exception {
        val back = new DataSetGenerator(1000, new int[]{32, 100}, new int[]{32, 5});

        val splitter = new DataSetIteratorSplitter(back, 1000, 0.7);

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
                val data = train.next().getFeatures();

                assertEquals("Train failed on iteration " + cnt + "; epoch: " + e, (float) cnt++, data.getFloat(0), 1e-5);
                gcntTrain++;
                global++;
            }

            train.reset();


            while (test.hasNext()) {
                val data = test.next().getFeatures();
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
}
