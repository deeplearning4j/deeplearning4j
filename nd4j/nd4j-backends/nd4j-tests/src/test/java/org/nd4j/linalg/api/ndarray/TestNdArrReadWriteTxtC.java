
package org.nd4j.linalg.api.ndarray;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
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

    public TestNdArrReadWriteTxtC(Nd4jBackend backend) {

        super(backend);
    }

    @Test
    public void compareAfterWrite() {
        int[] ranksToCheck = new int[]{0, 1, 2, 3, 4};
        for (int i = 0; i < ranksToCheck.length; i++) {
            log.info("Checking read write arrays with rank " + ranksToCheck[i]);
            compareArrays(ranksToCheck[i], ordering());
        }
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
