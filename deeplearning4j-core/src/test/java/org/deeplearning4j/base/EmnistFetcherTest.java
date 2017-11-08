package org.deeplearning4j.base;

import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator;
import org.junit.Test;

import java.io.File;

/**
 * @author Max Pumperla
 */
public class EmnistFetcherTest {

    @Test
    public void testEMnistDataFetcher() throws Exception {
        EmnistFetcher emnistFetcher = new EmnistFetcher(EmnistDataSetIterator.Set.MNIST);
        File emnistDir = emnistFetcher.downloadAndUntar();

        assert (emnistDir.isDirectory());
    }
}
