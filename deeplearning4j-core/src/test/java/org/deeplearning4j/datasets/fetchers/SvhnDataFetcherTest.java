package org.deeplearning4j.datasets.fetchers;

import org.junit.Test;

import java.io.File;

import static org.junit.Assert.assertTrue;

/**
 * @author saudet
 */
public class SvhnDataFetcherTest {

    @Test
    public void testMnistDataFetcher() throws Exception {
        SvhnDataFetcher fetch = new SvhnDataFetcher();
        File path = fetch.getDataSetPath(DataSetType.TRAIN);
        File path2 = fetch.getDataSetPath(DataSetType.TEST);
        File path3 = fetch.getDataSetPath(DataSetType.VALIDATION);

        assertTrue(path.isDirectory());
        assertTrue(path2.isDirectory());
        assertTrue(path3.isDirectory());
    }
}
