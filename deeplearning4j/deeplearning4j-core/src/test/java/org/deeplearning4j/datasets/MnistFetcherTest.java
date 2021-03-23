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
package org.deeplearning4j.datasets;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.base.MnistFetcher;
import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.junit.jupiter.api.*;
import org.junit.jupiter.api.io.TempDir;

import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.reduce.longer.MatchCondition;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Conditions;
import java.io.File;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.Set;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.extension.ExtendWith;

@DisplayName("Mnist Fetcher Test")
@NativeTag
@Tag(TagNames.FILE_IO)
@Tag(TagNames.NDARRAY_ETL)
class MnistFetcherTest extends BaseDL4JTest {

    @TempDir public static Path tempPath;

    @BeforeAll
    static void setup() throws Exception {
        DL4JResources.setBaseDirectory(tempPath.toFile());
    }

    @AfterAll
    static void after() throws Exception {
        DL4JResources.resetBaseDirectoryLocation();
    }

    @Test
    @DisplayName("Test Mnist")
    void testMnist() throws Exception {
        MnistDataSetIterator iter = new MnistDataSetIterator(32, 60000, false, true, false, -1);
        int count = 0;
        while (iter.hasNext()) {
            DataSet ds = iter.next();
            INDArray arr = ds.getFeatures().sum(1);
            int countMatch = Nd4j.getExecutioner().execAndReturn(new MatchCondition(arr, Conditions.equals(0))).z().getInt(0);
            assertEquals(0, countMatch);
            count++;
        }
        assertEquals(60000 / 32, count);
        count = 0;
        iter = new MnistDataSetIterator(32, false, 12345);
        while (iter.hasNext()) {
            DataSet ds = iter.next();
            INDArray arr = ds.getFeatures().sum(1);
            int countMatch = Nd4j.getExecutioner().execAndReturn(new MatchCondition(arr, Conditions.equals(0))).z().getInt(0);
            assertEquals(0, countMatch);
            count++;
        }
        assertEquals((int) Math.ceil(10000 / 32.0), count);
        iter.close();
    }

    @Test
    @DisplayName("Test Mnist Data Fetcher")
    void testMnistDataFetcher() throws Exception {
        MnistFetcher mnistFetcher = new MnistFetcher();
        File mnistDir = mnistFetcher.downloadAndUntar();
        assertTrue(mnistDir.isDirectory());

    }

    @Test
    public void testMnistSubset() throws Exception {
        final int numExamples = 100;
        MnistDataSetIterator iter1 = new MnistDataSetIterator(10, numExamples, false, true, true, 123);
        int examples1 = 0;
        int itCount1 = 0;
        while (iter1.hasNext()) {
            itCount1++;
            examples1 += iter1.next().numExamples();
        }
        assertEquals(10, itCount1);
        assertEquals(100, examples1);
        iter1.close();
        MnistDataSetIterator iter2 = new MnistDataSetIterator(10, numExamples, false, true, true, 123);
        iter2.close();
        int examples2 = 0;
        int itCount2 = 0;
        for (int i = 0; i < 10; i++) {
            itCount2++;
            examples2 += iter2.next().numExamples();
        }
        assertFalse(iter2.hasNext());
        assertEquals(10, itCount2);
        assertEquals(100, examples2);
        MnistDataSetIterator iter3 = new MnistDataSetIterator(19, numExamples, false, true, true, 123);
        iter3.close();
        int examples3 = 0;
        int itCount3 = 0;
        while (iter3.hasNext()) {
            itCount3++;
            examples3 += iter3.next().numExamples();
        }
        assertEquals(100, examples3);
        assertEquals((int) Math.ceil(100 / 19.0), itCount3);
        MnistDataSetIterator iter4 = new MnistDataSetIterator(32, true, 12345);
        int count4 = 0;
        while (iter4.hasNext()) {
            count4 += iter4.next().numExamples();
        }
        assertEquals(60000, count4);
        iter4.close();
        iter1.close();
    }

    @Test
    @DisplayName("Test Subset Repeatability")
    void testSubsetRepeatability() throws Exception {
        MnistDataSetIterator it = new MnistDataSetIterator(1, 1, false, false, true, 0);
        DataSet d1 = it.next();
        for (int i = 0; i < 10; i++) {
            it.reset();
            DataSet d2 = it.next();
            assertEquals(d1.get(0).getFeatures(), d2.get(0).getFeatures());
        }
        it.close();
        // Check larger number:
        it = new MnistDataSetIterator(8, 32, false, false, true, 12345);
        Set<String> featureLabelSet = new HashSet<>();
        while (it.hasNext()) {
            DataSet ds = it.next();
            INDArray f = ds.getFeatures();
            INDArray l = ds.getLabels();
            for (int i = 0; i < f.size(0); i++) {
                featureLabelSet.add(f.getRow(i).toString() + "\t" + l.getRow(i).toString());
            }
        }
        assertEquals(32, featureLabelSet.size());
        it.close();
        for (int i = 0; i < 3; i++) {
            it.reset();
            Set<String> flSet2 = new HashSet<>();
            while (it.hasNext()) {
                DataSet ds = it.next();
                INDArray f = ds.getFeatures();
                INDArray l = ds.getLabels();
                for (int j = 0; j < f.size(0); j++) {
                    flSet2.add(f.getRow(j).toString() + "\t" + l.getRow(j).toString());
                }
            }
            assertEquals(featureLabelSet, flSet2);
        }

    }
}
