package org.deeplearning4j.base;

import org.deeplearning4j.BaseDL4JTest;
import org.junit.Test;

import java.io.File;

/**
 * @author Justin Long (crockpotveggies)
 */
public class MnistFetcherTest extends BaseDL4JTest {

    @Test
    public void testMnistDataFetcher() throws Exception {
        MnistFetcher mnistFetcher = new MnistFetcher();
        File mnistDir = mnistFetcher.downloadAndUntar();

        assert (mnistDir.isDirectory());
    }
}
