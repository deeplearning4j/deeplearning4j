package org.nd4j.linalg.api.rng.distribution;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

/**
 * @author Adam Gibson
 */
public class DistributionTestsC extends BaseNd4jTest {
    public DistributionTestsC() {
    }

    public DistributionTestsC(String name) {
        super(name);
    }

    public DistributionTestsC(String name, Nd4jBackend backend) {
        super(name, backend);
    }

    public DistributionTestsC(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testBinomial() {
        Nd4j.getDistributions().createBinomial(1,0).sample(new int[]{784,600});

    }

    @Override
    public char ordering() {
        return 'c';
    }
}
