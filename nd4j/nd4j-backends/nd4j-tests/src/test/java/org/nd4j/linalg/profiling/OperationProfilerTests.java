/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.profiling;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.primitives.Pair;
import org.junit.After;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.profiler.OpProfiler;

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

        OpProfiler.PenaltyCause[] causes = OpProfiler.getInstance().processOperands(x, y);

        assertEquals(1, causes.length);
        assertTrue(ArrayUtils.contains(causes, OpProfiler.PenaltyCause.NONE));
    }

    @Test
    public void testBadCombos2() throws Exception {
        INDArray x = Nd4j.create(100).reshape('f', 10, 10);
        INDArray y = Nd4j.create(100).reshape('c', 10, 10);

        OpProfiler.PenaltyCause[] causes = OpProfiler.getInstance().processOperands(x, y);

        assertEquals(1, causes.length);
        assertTrue(ArrayUtils.contains(causes, OpProfiler.PenaltyCause.MIXED_ORDER));
    }

    @Test
    public void testBadCombos3() throws Exception {
        INDArray x = Nd4j.create(27).reshape('c', 3, 3, 3).tensorAlongDimension(0, 1, 2);
        INDArray y = Nd4j.create(100).reshape('f', 10, 10);

        OpProfiler.PenaltyCause[] causes = OpProfiler.getInstance().processOperands(x, y);

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

        OpProfiler.PenaltyCause[] causes = OpProfiler.getInstance().processOperands(x, y, z);

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
    @Ignore
    public void testBadCombos6() throws Exception {
        INDArray x = Nd4j.create(27).reshape('f', 3, 3, 3).slice(1);
        INDArray y = Nd4j.create(100).reshape('f', 10, 10);

        OpProfiler.PenaltyCause[] causes = OpProfiler.getInstance().processOperands(x, y);

        log.info("Causes: {}", Arrays.toString(causes));
        assertEquals(1, causes.length);
        assertTrue(ArrayUtils.contains(causes, OpProfiler.PenaltyCause.STRIDED_ACCESS));
    }

    @Test
    public void testBadTad1() throws Exception {
        INDArray x = Nd4j.create(2, 4, 5, 6);

        Pair<DataBuffer, DataBuffer> pair =
                        Nd4j.getExecutioner().getTADManager().getTADOnlyShapeInfo(x, new int[] {0, 2});

        OpProfiler.PenaltyCause[] causes = OpProfiler.getInstance().processTADOperands(pair.getFirst());

        log.info("Causes: {}", Arrays.toString(causes));
        assertEquals(1, causes.length);
        assertTrue(ArrayUtils.contains(causes, OpProfiler.PenaltyCause.TAD_NON_EWS_ACCESS));
    }

    @Test
    public void testBadTad2() throws Exception {
        INDArray x = Nd4j.create(2, 4, 5, 6);

        Pair<DataBuffer, DataBuffer> pair =
                        Nd4j.getExecutioner().getTADManager().getTADOnlyShapeInfo(x, new int[] {2, 3});

        OpProfiler.PenaltyCause[] causes = OpProfiler.getInstance().processTADOperands(pair.getFirst());

        log.info("Causes: {}", Arrays.toString(causes));
        assertEquals(1, causes.length);
        assertTrue(ArrayUtils.contains(causes, OpProfiler.PenaltyCause.TAD_NON_EWS_ACCESS));
    }



    @Test
    public void testBadTad3() throws Exception {
        INDArray x = Nd4j.create(new int[] {2, 4, 5, 6, 7}, 'f');

        Pair<DataBuffer, DataBuffer> pair =
                        Nd4j.getExecutioner().getTADManager().getTADOnlyShapeInfo(x, new int[] {0, 2, 4});

        OpProfiler.PenaltyCause[] causes = OpProfiler.getInstance().processTADOperands(pair.getFirst());

        log.info("Causes: {}", Arrays.toString(causes));
        assertEquals(1, causes.length);
        assertTrue(ArrayUtils.contains(causes, OpProfiler.PenaltyCause.TAD_NON_EWS_ACCESS));
    }

    @Test
    @Ignore
    public void testBadTad4() throws Exception {
        INDArray x = Nd4j.create(2, 4, 5, 6);

        Pair<DataBuffer, DataBuffer> pair = Nd4j.getExecutioner().getTADManager().getTADOnlyShapeInfo(x, new int[] {3});

        OpProfiler.PenaltyCause[] causes = OpProfiler.getInstance().processTADOperands(pair.getFirst());

        log.info("TAD: {}", Arrays.toString(pair.getFirst().asInt()));
        log.info("Causes: {}", Arrays.toString(causes));
        assertEquals(1, causes.length);
        assertTrue(ArrayUtils.contains(causes, OpProfiler.PenaltyCause.NONE));
    }

    @Test
    public void testBadTad5() throws Exception {
        INDArray x = Nd4j.create(new int[] {2, 4, 5, 6, 7}, 'f');

        Pair<DataBuffer, DataBuffer> pair = Nd4j.getExecutioner().getTADManager().getTADOnlyShapeInfo(x, new int[] {4});

        OpProfiler.PenaltyCause[] causes = OpProfiler.getInstance().processTADOperands(pair.getFirst());

        log.info("TAD: {}", Arrays.toString(pair.getFirst().asInt()));
        log.info("Causes: {}", Arrays.toString(causes));
        assertEquals(1, causes.length);
        assertTrue(ArrayUtils.contains(causes, OpProfiler.PenaltyCause.TAD_STRIDED_ACCESS));
    }


    @Test
    public void testCxFxF1() throws Exception {
        INDArray a = Nd4j.create(10, 10).reshape('f', 10, 10);
        INDArray b = Nd4j.create(10, 10).reshape('c', 10, 10);
        INDArray c = Nd4j.create(10, 10).reshape('f', 10, 10);

        String ret = OpProfiler.getInstance().processOrders(a, b, c);
        assertEquals("F x C x F", ret);
    }

    @Test
    public void testCxFxF2() throws Exception {
        INDArray a = Nd4j.create(10, 10).reshape('c', 10, 10);
        INDArray b = Nd4j.create(10, 10).reshape('c', 10, 10);
        INDArray c = Nd4j.create(10, 10).reshape('f', 10, 10);

        String ret = OpProfiler.getInstance().processOrders(a, b, c);
        assertEquals("C x C x F", ret);
    }

    @Test
    public void testCxFxF3() throws Exception {
        INDArray a = Nd4j.create(10, 10).reshape('c', 10, 10);
        INDArray b = Nd4j.create(10, 10).reshape('c', 10, 10);
        INDArray c = Nd4j.create(10, 10).reshape('c', 10, 10);

        String ret = OpProfiler.getInstance().processOrders(a, b, c);
        assertEquals("C x C x C", ret);
    }


    @Test
    public void testBlasFF() throws Exception {
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.ALL);

        INDArray a = Nd4j.create(10, 10).reshape('f', 10, 10);
        INDArray b = Nd4j.create(10, 10).reshape('f', 10, 10);

        a.mmul(b);

        OpProfiler.getInstance().printOutDashboard();
    }


    @Test(expected = ND4JIllegalStateException.class)
    public void testNaNPanic1() throws Exception {
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.NAN_PANIC);

        INDArray a = Nd4j.create(new float[] {1f, 2f, 3f, Float.NaN});

        a.muli(3f);
    }

    @Test(expected = ND4JIllegalStateException.class)
    public void testNaNPanic2() throws Exception {
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.INF_PANIC);

        INDArray a = Nd4j.create(new float[] {1f, 2f, 3f, Float.POSITIVE_INFINITY});

        a.muli(3f);
    }

    @Test(expected = ND4JIllegalStateException.class)
    public void testNaNPanic3() throws Exception {
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.ANY_PANIC);

        INDArray a = Nd4j.create(new float[] {1f, 2f, 3f, Float.NEGATIVE_INFINITY});

        a.muli(3f);
    }


    @Test(expected = ND4JIllegalStateException.class)
    public void testScopePanic1() throws Exception {
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.SCOPE_PANIC);

        INDArray array;

        try (MemoryWorkspace workspace = Nd4j.getWorkspaceManager().getAndActivateWorkspace("WS119")) {
            array = Nd4j.create(10);

            assertTrue(array.isAttached());
        }

        array.add(1.0);
    }


    @Test(expected = ND4JIllegalStateException.class)
    public void testScopePanic2() throws Exception {
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.SCOPE_PANIC);

        INDArray array;

        try (MemoryWorkspace workspace = Nd4j.getWorkspaceManager().getAndActivateWorkspace("WS120")) {
            array = Nd4j.create(10);
            assertTrue(array.isAttached());

            assertEquals(1, workspace.getGenerationId());
        }


        try (MemoryWorkspace workspace = Nd4j.getWorkspaceManager().getAndActivateWorkspace("WS120")) {
            assertEquals(2, workspace.getGenerationId());

            array.add(1.0);

            assertTrue(array.isAttached());
        }
    }


    @Test
    public void testScopePanic3() throws Exception {
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.SCOPE_PANIC);


        INDArray array;

        try (MemoryWorkspace workspace = Nd4j.getWorkspaceManager().getAndActivateWorkspace("WS121")) {
            array = Nd4j.create(10);
            assertTrue(array.isAttached());

            assertEquals(1, workspace.getGenerationId());


            try (MemoryWorkspace workspaceInner = Nd4j.getWorkspaceManager().getAndActivateWorkspace("WS122")) {
                array.add(1.0);
            }
        }
    }

    @Test
    public void testScopePanicPerf() {
        try (MemoryWorkspace workspace = Nd4j.getWorkspaceManager().getAndActivateWorkspace("WS121")) {
            INDArray x = Nd4j.create(1000, 1000).assign(1.0);
            INDArray y = Nd4j.create(1000, 1000).assign(1.0);

            for (int e = 0; e < 10000; e++) {
                x.addi(y);
            }

            Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.SCOPE_PANIC);

            val nanosC = System.nanoTime();
            for (int e = 0; e < 10000; e++) {
                x.addi(y);
            }
            val nanosD = System.nanoTime();

            val avgB = (nanosD - nanosC) / 10000;


            Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.DISABLED);

            val nanosA = System.nanoTime();
            for (int e = 0; e < 10000; e++) {
                x.addi(y);
            }
            val nanosB = System.nanoTime();

            val avgA = (nanosB - nanosA) / 10000;


            log.info("A: {}; B: {}", avgA, avgB);
        }
    }


}
