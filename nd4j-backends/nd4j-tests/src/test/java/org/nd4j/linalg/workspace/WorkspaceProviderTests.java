package org.nd4j.linalg.workspace;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

/**
 * @author raver119@gmail.com
 */
@Slf4j
@RunWith(Parameterized.class)
public class WorkspaceProviderTests extends BaseNd4jTest {
    DataBuffer.Type initialType;

    public WorkspaceProviderTests(Nd4jBackend backend) {
        super(backend);
        this.initialType = Nd4j.dataType();
    }

    @Test
    public void testNestedWorkspaces1() throws Exception {

    }

    @Override
    public char ordering() {
        return 'c';
    }
}
