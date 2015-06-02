package org.deeplearning4j.datasets.fetchers;

import org.junit.Test;
import org.nd4j.linalg.dataset.api.iterator.fetcher.DataSetFetcher;

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
