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

package org.nd4j.linalg.dataset;


import org.apache.commons.io.FileUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.CachingDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.SamplingDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.cache.DataSetCache;
import org.nd4j.linalg.dataset.api.iterator.cache.InFileDataSetCache;
import org.nd4j.linalg.dataset.api.iterator.cache.InMemoryDataSetCache;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.*;

/**
 * Created by anton on 7/18/16.
 */
@RunWith(Parameterized.class)
public class CachingDataSetIteratorTest extends BaseNd4jTest {

    public CachingDataSetIteratorTest(Nd4jBackend backend) {
        super(backend);
    }

    @Override
    public char ordering() {
        return 'f';
    }

    @Test
    public void testInMemory() {
        DataSetCache cache = new InMemoryDataSetCache();

        runDataSetTest(cache);
    }

    @Test
    public void testInFile() throws IOException {
        Path cacheDir = Files.createTempDirectory("nd4j-data-set-cache-test");
        DataSetCache cache = new InFileDataSetCache(cacheDir);

        runDataSetTest(cache);

        FileUtils.deleteDirectory(cacheDir.toFile());
    }

    private void runDataSetTest(DataSetCache cache) {
        int rows = 500;
        int inputColumns = 100;
        int outputColumns = 2;
        DataSet dataSet = new DataSet(Nd4j.ones(rows, inputColumns), Nd4j.zeros(rows, outputColumns));

        int batchSize = 10;
        int totalNumberOfSamples = 50;
        int expectedNumberOfDataSets = totalNumberOfSamples / batchSize;
        DataSetIterator it = new SamplingDataSetIterator(dataSet, batchSize, totalNumberOfSamples);

        String namespace = "test-namespace";

        CachingDataSetIterator cachedIt = new CachingDataSetIterator(it, cache, namespace);
        PreProcessor preProcessor = new PreProcessor();
        cachedIt.setPreProcessor(preProcessor);

        assertDataSetCacheGetsCompleted(cache, namespace, cachedIt);

        assertPreProcessingGetsCached(expectedNumberOfDataSets, it, cachedIt, preProcessor);

        assertCachingDataSetIteratorHasAllTheData(rows, inputColumns, outputColumns, dataSet, it, cachedIt);
    }

    private void assertDataSetCacheGetsCompleted(DataSetCache cache, String namespace, DataSetIterator cachedIt) {
        while (cachedIt.hasNext()) {
            assertFalse(cache.isComplete(namespace));
            cachedIt.next();
        }

        assertTrue(cache.isComplete(namespace));
    }

    private void assertPreProcessingGetsCached(int expectedNumberOfDataSets, DataSetIterator it,
                    CachingDataSetIterator cachedIt, PreProcessor preProcessor) {

        assertSame(preProcessor, cachedIt.getPreProcessor());
        assertSame(preProcessor, it.getPreProcessor());

        cachedIt.reset();
        it.reset();

        while (cachedIt.hasNext()) {
            cachedIt.next();
        }

        assertEquals(expectedNumberOfDataSets, preProcessor.getCallCount());

        cachedIt.reset();
        it.reset();

        while (cachedIt.hasNext()) {
            cachedIt.next();
        }

        assertEquals(expectedNumberOfDataSets, preProcessor.getCallCount());
    }

    private void assertCachingDataSetIteratorHasAllTheData(int rows, int inputColumns, int outputColumns,
                    DataSet dataSet, DataSetIterator it, CachingDataSetIterator cachedIt) {
        cachedIt.reset();
        it.reset();

        dataSet.setFeatures(Nd4j.zeros(rows, inputColumns));
        dataSet.setLabels(Nd4j.ones(rows, outputColumns));

        while (it.hasNext()) {
            assertTrue(cachedIt.hasNext());

            DataSet cachedDs = cachedIt.next();
            assertEquals(1000.0, cachedDs.getFeatures().sumNumber());
            assertEquals(0.0, cachedDs.getLabels().sumNumber());

            DataSet ds = it.next();
            assertEquals(0.0, ds.getFeatures().sumNumber());
            assertEquals(20.0, ds.getLabels().sumNumber());
        }

        assertFalse(cachedIt.hasNext());
        assertFalse(it.hasNext());
    }

    private class PreProcessor implements DataSetPreProcessor {

        private int callCount;

        @Override
        public void preProcess(org.nd4j.linalg.dataset.api.DataSet toPreProcess) {
            callCount++;
        }

        public int getCallCount() {
            return callCount;
        }
    }
}
