
package org.nd4j.linalg.api.ndarray;

import lombok.extern.slf4j.Slf4j;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.nd4j.linalg.api.ndarray.TestNdArrReadWriteTxt.compareArrays;

/**
 * Created by susaneraly on 6/18/16.
 */
@Slf4j
@RunWith(Parameterized.class)
public class TestNdArrReadWriteTxtC extends BaseNd4jTest {

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    public TestNdArrReadWriteTxtC(Nd4jBackend backend) {

        super(backend);
    }

    @Test
    public void compareAfterWrite() throws Exception {
        int[] ranksToCheck = new int[]{0, 1, 2, 3, 4};
        for (int i = 0; i < ranksToCheck.length; i++) {
            log.info("Checking read write arrays with rank " + ranksToCheck[i]);
            compareArrays(ranksToCheck[i], ordering(), testDir);
        }
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
