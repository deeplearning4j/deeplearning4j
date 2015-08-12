package org.nd4j.linalg.api.indexing;

import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.factory.Nd4jBackend;

/**
 * @author Adam Gibson
 */
public class ShapeResolutionTests extends BaseNd4jTest {

    public ShapeResolutionTests(String name, Nd4jBackend backend) {
        super(name, backend);
    }

    @Override
    public char ordering() {
        return 'f';
    }
}
