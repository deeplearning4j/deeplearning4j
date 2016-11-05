package org.nd4j.linalg.crash;

import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.factory.Nd4jBackend;

/**
 * This set of test launches different ops in different order, to check for possible data corruption cases
 *
 * @author raver119@gmail.com
 */
@RunWith(Parameterized.class)
public class CrashTest extends BaseNd4jTest {
    public CrashTest(Nd4jBackend backend) {
        super(backend);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
