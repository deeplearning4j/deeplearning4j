package org.nd4j.linalg.workspace;

import lombok.extern.slf4j.Slf4j;
import org.junit.After;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.concurrent.atomic.AtomicLong;

import static org.junit.Assert.assertEquals;

/**
 * THESE TESTS SHOULD NOT BE EXECUTED IN CI ENVIRONMENT - THEY ARE ENDLESS
 * @author raver119@gmail.com
 */
@Ignore
@Slf4j
@RunWith(Parameterized.class)
public class EndlessWorkspaceTests extends BaseNd4jTest {
    DataBuffer.Type initialType;

    public EndlessWorkspaceTests(Nd4jBackend backend) {
        super(backend);
        this.initialType = Nd4j.dataType();
    }

    @After
    public void shutUp() throws Exception {
        Nd4j.getMemoryManager().setCurrentWorkspace(null);
        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
        Nd4j.setDataType(this.initialType);
    }

    /**
     * This test checks for allocations within single workspace, without any spills
     *
     * @throws Exception
     */
    @Test
    public void endlessTest1() throws Exception {

        Nd4j.getWorkspaceManager().setDefaultWorkspaceConfiguration(WorkspaceConfiguration.builder().initialSize(100 * 1024L * 1024L).build());

        AtomicLong counter = new AtomicLong(0);
        while (true) {
            try (MemoryWorkspace workspace = Nd4j.getWorkspaceManager().getAndActivateWorkspace()){
                INDArray array = Nd4j.create(1024 * 1024);
                array.addi(1.0f);
                assertEquals(1.0f, array.meanNumber().floatValue(), 0.1f);

                if (counter.incrementAndGet() % 1000 == 0)
                    log.info("{} iterations passed...", counter.get());
            }
        }
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
