package org.nd4j.linalg.api.rng.distribution;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author Adam Gibson
 */
public class DistributionTestsC extends BaseNd4jTest {
    @Test
    public void testBinomial() {
        Nd4j.getDistributions().createBinomial(1,0).sample(new int[]{784,600});

    }

    @Override
    public char ordering() {
        return 'c';
    }
}
