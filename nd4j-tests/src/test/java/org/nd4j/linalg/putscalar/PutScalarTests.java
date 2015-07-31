package org.nd4j.linalg.putscalar;

import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.factory.Nd4jBackend;

/**
 * @author Adam Gibson
 */
public class PutScalarTests extends BaseNd4jTest {
    public PutScalarTests(String name, Nd4jBackend backend) {
        super(name, backend);
    }

    @Override
    public char ordering() {
        return 'f';
    }
}
