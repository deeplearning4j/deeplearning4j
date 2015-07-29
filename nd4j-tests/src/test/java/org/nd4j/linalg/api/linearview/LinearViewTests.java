package org.nd4j.linalg.api.linearview;

import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.factory.Nd4jBackend;

/**
 * @author Adam Gibson
 */
public class LinearViewTests extends BaseNd4jTest {
    public LinearViewTests() {
        super();
    }

    public LinearViewTests(String name) {
        super(name);
    }

    public LinearViewTests(String name, Nd4jBackend backend) {
        super(name, backend);
    }

    public LinearViewTests(Nd4jBackend backend) {
        super(backend);
    }

    @Override
    public char ordering() {
        return 'f';
    }
}
