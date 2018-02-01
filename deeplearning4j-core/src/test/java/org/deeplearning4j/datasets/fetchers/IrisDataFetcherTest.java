package org.deeplearning4j.datasets.fetchers;

import org.deeplearning4j.BaseDL4JTest;
import org.junit.Test;

/**
 * @author Adam Gibson
 */
public class IrisDataFetcherTest extends BaseDL4JTest {

    @Test
    public void testIrisDataFetcher() throws Exception {
        IrisDataFetcher irisFetcher = new IrisDataFetcher();
        irisFetcher.fetch(10);

    }

}
