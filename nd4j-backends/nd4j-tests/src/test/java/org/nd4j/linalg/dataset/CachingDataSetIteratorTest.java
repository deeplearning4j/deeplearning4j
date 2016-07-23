package org.nd4j.linalg.dataset;


import org.apache.commons.io.FileUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.dataset.api.iterator.CachingDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.SamplingDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.cache.DataSetCache;
import org.nd4j.linalg.dataset.api.iterator.cache.InFileDataSetCache;
import org.nd4j.linalg.dataset.api.iterator.cache.InMemoryDataSetCache;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

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
        DataSet dataSet = new DataSet(Nd4j.ones(500, 100), Nd4j.zeros(500, 2));

        DataSetIterator it = new SamplingDataSetIterator(dataSet, 10, 50);

        String namespace = "test-namespace";

        DataSetIterator cachedIt = new CachingDataSetIterator(it, cache, namespace);

        while (cachedIt.hasNext()) {
            assertFalse(cache.isComplete(namespace));
            cachedIt.next();
        }

        assertTrue(cache.isComplete(namespace));

        cachedIt.reset();
        it.reset();

        dataSet.setFeatures(Nd4j.zeros(500, 100));
        dataSet.setLabels(Nd4j.ones(500, 2));

        while (it.hasNext()) {
            assertTrue(cachedIt.hasNext());

            DataSet cachedDs = cachedIt.next();
            assertEquals(1000.0, cachedDs.getFeatureMatrix().sumNumber());
            assertEquals(0.0, cachedDs.getLabels().sumNumber());

            DataSet ds = it.next();
            assertEquals(0.0, ds.getFeatureMatrix().sumNumber());
            assertEquals(20.0, ds.getLabels().sumNumber());
        }

        assertFalse(cachedIt.hasNext());
        assertFalse(it.hasNext());
    }
}