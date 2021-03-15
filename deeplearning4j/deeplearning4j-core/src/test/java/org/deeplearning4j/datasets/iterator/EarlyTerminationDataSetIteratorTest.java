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

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.junit.Rule;
import org.junit.jupiter.api.Test;
import org.junit.rules.ExpectedException;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.extension.ExtendWith;

@DisplayName("Early Termination Data Set Iterator Test")
class EarlyTerminationDataSetIteratorTest extends BaseDL4JTest {

    int minibatchSize = 10;

    int numExamples = 105;

    @Rule
    public final ExpectedException exception = ExpectedException.none();

    @Test
    @DisplayName("Test Next And Reset")
    void testNextAndReset() throws Exception {
        int terminateAfter = 2;
        DataSetIterator iter = new MnistDataSetIterator(minibatchSize, numExamples);
        EarlyTerminationDataSetIterator earlyEndIter = new EarlyTerminationDataSetIterator(iter, terminateAfter);
        assertTrue(earlyEndIter.hasNext());
        int batchesSeen = 0;
        List<DataSet> seenData = new ArrayList<>();
        while (earlyEndIter.hasNext()) {
            DataSet path = earlyEndIter.next();
            assertFalse(path == null);
            seenData.add(path);
            batchesSeen++;
        }
        assertEquals(batchesSeen, terminateAfter);
        // check data is repeated after reset
        earlyEndIter.reset();
        batchesSeen = 0;
        while (earlyEndIter.hasNext()) {
            DataSet path = earlyEndIter.next();
            assertEquals(seenData.get(batchesSeen).getFeatures(), path.getFeatures());
            assertEquals(seenData.get(batchesSeen).getLabels(), path.getLabels());
            batchesSeen++;
        }
    }

    @Test
    @DisplayName("Test Next Num")
    void testNextNum() throws IOException {
        int terminateAfter = 1;
        DataSetIterator iter = new MnistDataSetIterator(minibatchSize, numExamples);
        EarlyTerminationDataSetIterator earlyEndIter = new EarlyTerminationDataSetIterator(iter, terminateAfter);
        earlyEndIter.next(10);
        assertEquals(false, earlyEndIter.hasNext());
        earlyEndIter.reset();
        assertEquals(true, earlyEndIter.hasNext());
    }

    @Test
    @DisplayName("Test Callsto Next Not Allowed")
    void testCallstoNextNotAllowed() throws IOException {
        int terminateAfter = 1;
        DataSetIterator iter = new MnistDataSetIterator(minibatchSize, numExamples);
        EarlyTerminationDataSetIterator earlyEndIter = new EarlyTerminationDataSetIterator(iter, terminateAfter);
        earlyEndIter.next(10);
        iter.reset();
        exception.expect(RuntimeException.class);
        earlyEndIter.next(10);
    }
}
