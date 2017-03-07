package org.deeplearning4j.datasets.fetchers;

import org.junit.Test;

/**
 * @author Justin Long (crockpotveggies)
 */
public class MnistFetcherTest {

    @Test
    public void testIrisDataFetcher() throws Exception {
        MnistFetcher mnistFetcher = new MnistFetcher();
        mnistFetcher.downloadAndUntar();

    }

}
