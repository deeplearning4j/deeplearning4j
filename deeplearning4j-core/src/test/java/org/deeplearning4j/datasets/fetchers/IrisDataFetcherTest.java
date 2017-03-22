package org.deeplearning4j.datasets.fetchers;

import org.junit.Test;

/**
 * @author Adam Gibson
 */
public class IrisDataFetcherTest {

    @Test
    public void testIrisDataFetcher() throws Exception {
        IrisDataFetcher irisFetcher = new IrisDataFetcher();
        irisFetcher.fetch(10);

    }

}
