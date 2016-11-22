package org.nd4j.linalg.profiling;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.SystemUtils;
import org.apache.commons.math3.util.Pair;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.profiler.OpProfiler;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

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

        assertEquals(2, OpProfiler.getInstance().getInvocationsCount());
    }


    @Test
    public void testStack1() throws Exception {

        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.ALL);

        INDArray array = Nd4j.createUninitialized(100);

        array.assign(10f);
        array.assign(20f);
        array.assign(30f);

        assertEquals(3, OpProfiler.getInstance().getInvocationsCount());

        OpProfiler.getInstance().printOutDashboard();
    }


    @Test
    public void testBadCombos1() throws Exception {
        INDArray x = Nd4j.create(100);
        INDArray y = Nd4j.create(100);

        OpProfiler.PenaltyCause[] causes = OpProfiler.getInstance().processOperands(x,y);

        assertEquals(1, causes.length);
        assertTrue(ArrayUtils.contains(causes, OpProfiler.PenaltyCause.NONE));
    }

    @Test
    public void testBadCombos2() throws Exception {
        INDArray x = Nd4j.create(100).reshape('f', 10, 10);
        INDArray y = Nd4j.create(100).reshape('c', 10, 10);

        OpProfiler.PenaltyCause[] causes = OpProfiler.getInstance().processOperands(x,y);

        assertEquals(1, causes.length);
        assertTrue(ArrayUtils.contains(causes, OpProfiler.PenaltyCause.MIXED_ORDER));
    }

    @Test
    public void testBadCombos3() throws Exception {
        INDArray x = Nd4j.create(27).reshape('c', 3, 3, 3).tensorAlongDimension(0, 1, 2);
        INDArray y = Nd4j.create(100).reshape('f', 10, 10);

        OpProfiler.PenaltyCause[] causes = OpProfiler.getInstance().processOperands(x,y);

        log.info("Causes: {}", Arrays.toString(causes));
        assertEquals(2, causes.length);
        assertTrue(ArrayUtils.contains(causes, OpProfiler.PenaltyCause.MIXED_ORDER));
        assertTrue(ArrayUtils.contains(causes, OpProfiler.PenaltyCause.NON_EWS_ACCESS));
    }

    @Test
    public void testBadCombos4() throws Exception {
        INDArray x = Nd4j.create(27).reshape('c', 3, 3, 3).tensorAlongDimension(0, 1, 2);
        INDArray y = Nd4j.create(100).reshape('f', 10, 10);
        INDArray z = Nd4j.create(100).reshape('f', 10, 10);

        OpProfiler.PenaltyCause[] causes = OpProfiler.getInstance().processOperands(x,y,z);

        log.info("Causes: {}", Arrays.toString(causes));
        assertEquals(2, causes.length);
        assertTrue(ArrayUtils.contains(causes, OpProfiler.PenaltyCause.MIXED_ORDER));
        assertTrue(ArrayUtils.contains(causes, OpProfiler.PenaltyCause.NON_EWS_ACCESS));
    }

    @Test
    public void testBadCombos5() throws Exception {
        INDArray w = Nd4j.create(100).reshape('c', 10, 10);
        INDArray x = Nd4j.create(100).reshape('c', 10, 10);
        INDArray y = Nd4j.create(100).reshape('f', 10, 10);
        INDArray z = Nd4j.create(100).reshape('c', 10, 10);

        OpProfiler.PenaltyCause[] causes = OpProfiler.getInstance().processOperands(w, x, y, z);

        log.info("Causes: {}", Arrays.toString(causes));
        assertEquals(1, causes.length);
        assertTrue(ArrayUtils.contains(causes, OpProfiler.PenaltyCause.MIXED_ORDER));
    }

    @Test
    public void testBadCombos6() throws Exception {
        INDArray x = Nd4j.create(27).reshape('f', 3, 3, 3).slice(1);
        INDArray y = Nd4j.create(100).reshape('f', 10, 10);

        OpProfiler.PenaltyCause[] causes = OpProfiler.getInstance().processOperands(x,y);

        log.info("Causes: {}", Arrays.toString(causes));
        assertEquals(1, causes.length);
        assertTrue(ArrayUtils.contains(causes, OpProfiler.PenaltyCause.STRIDED_ACCESS));
    }

    @Test
    public void testBadTad1() throws Exception {
        INDArray x = Nd4j.create(2, 4, 5, 6);

        Pair<DataBuffer, DataBuffer> pair = Nd4j.getExecutioner().getTADManager().getTADOnlyShapeInfo(x,new int[]{0, 2});

        OpProfiler.PenaltyCause[] causes = OpProfiler.getInstance().processTADOperands(pair.getFirst());

        log.info("Causes: {}", Arrays.toString(causes));
        assertEquals(1, causes.length);
        assertTrue(ArrayUtils.contains(causes, OpProfiler.PenaltyCause.TAD_NON_EWS_ACCESS));
    }

    @Test
    public void testBadTad2() throws Exception {
        INDArray x = Nd4j.create(2, 4, 5, 6);

        Pair<DataBuffer, DataBuffer> pair = Nd4j.getExecutioner().getTADManager().getTADOnlyShapeInfo(x,new int[]{2, 3});

        OpProfiler.PenaltyCause[] causes = OpProfiler.getInstance().processTADOperands(pair.getFirst());

        log.info("Causes: {}", Arrays.toString(causes));
        assertEquals(1, causes.length);
        assertTrue(ArrayUtils.contains(causes, OpProfiler.PenaltyCause.TAD_NON_EWS_ACCESS));
    }



    @Test
    public void testBadTad3() throws Exception {
        INDArray x = Nd4j.create(new int[] {2, 4, 5, 6, 7}, 'f');

        Pair<DataBuffer, DataBuffer> pair = Nd4j.getExecutioner().getTADManager().getTADOnlyShapeInfo(x,new int[]{0, 2, 4});

        OpProfiler.PenaltyCause[] causes = OpProfiler.getInstance().processTADOperands(pair.getFirst());

        log.info("Causes: {}", Arrays.toString(causes));
        assertEquals(1, causes.length);
        assertTrue(ArrayUtils.contains(causes, OpProfiler.PenaltyCause.TAD_NON_EWS_ACCESS));
    }

    @Test
    public void testBadTad4() throws Exception {
        INDArray x = Nd4j.create(2, 4, 5, 6);

        Pair<DataBuffer, DataBuffer> pair = Nd4j.getExecutioner().getTADManager().getTADOnlyShapeInfo(x,new int[]{3});

        OpProfiler.PenaltyCause[] causes = OpProfiler.getInstance().processTADOperands(pair.getFirst());

        log.info("TAD: {}", Arrays.toString(pair.getFirst().asInt()));
        log.info("Causes: {}", Arrays.toString(causes));
        assertEquals(1, causes.length);
        assertTrue(ArrayUtils.contains(causes, OpProfiler.PenaltyCause.NONE));
    }

    @Test
    public void testBadTad5() throws Exception {
        INDArray x = Nd4j.create(new int[] {2, 4, 5, 6, 7}, 'f');

        Pair<DataBuffer, DataBuffer> pair = Nd4j.getExecutioner().getTADManager().getTADOnlyShapeInfo(x,new int[]{4});

        OpProfiler.PenaltyCause[] causes = OpProfiler.getInstance().processTADOperands(pair.getFirst());

        log.info("TAD: {}", Arrays.toString(pair.getFirst().asInt()));
        log.info("Causes: {}", Arrays.toString(causes));
        assertEquals(1, causes.length);
        assertTrue(ArrayUtils.contains(causes, OpProfiler.PenaltyCause.TAD_STRIDED_ACCESS));
    }
}
