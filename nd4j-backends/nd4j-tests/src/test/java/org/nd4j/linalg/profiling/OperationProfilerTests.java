package org.nd4j.linalg.profiling;

import lombok.extern.slf4j.Slf4j;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.profiler.OpProfiler;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class OperationProfilerTests {
    @Before
    public void setUp() {
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.OPERATIONS);
        OpProfiler.getInstance().reset();
    }

    @After
    public void tearDown() {
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.DISABLED);
    }

    @Test
    public void testCounter1() {
        INDArray array = Nd4j.createUninitialized(100);

        array.assign(10f);
        array.divi(2f);

        Assert.assertEquals(2, OpProfiler.getInstance().getInvocationsCount());
    }


    @Test
    public void testStack1() throws Exception {
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.METHODS);

        INDArray array = Nd4j.createUninitialized(100);

        array.assign(10f);
        array.assign(20f);
        array.assign(30f);

        OpProfiler.getInstance().printOutDashboard();
    }
}
