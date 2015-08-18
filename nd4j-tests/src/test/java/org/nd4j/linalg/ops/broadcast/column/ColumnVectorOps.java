package org.nd4j.linalg.ops.broadcast.column;

import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.factory.Nd4jBackend;

/**
 * @author Adam Gibson
 */
public class ColumnVectorOps extends BaseNd4jTest {
    public ColumnVectorOps(String name, Nd4jBackend backend) {
        super(name, backend);
    }

    @Override
    public char ordering() {
        return 'f';
    }
}
