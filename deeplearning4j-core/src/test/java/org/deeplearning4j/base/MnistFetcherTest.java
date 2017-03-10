package org.deeplearning4j.base;

import org.junit.Test;
import java.io.File;

/**
 * @author Justin Long (crockpotveggies)
 */
public class MnistFetcherTest {

    @Test
    public void testIrisDataFetcher() throws Exception {
        MnistFetcher mnistFetcher = new MnistFetcher();
        File mnistDir = mnistFetcher.downloadAndUntar();

        assert (mnistDir.isDirectory());
    }

}
