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
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.ops.performance.PerformanceTracker;
import org.nd4j.linalg.api.ops.performance.primitives.AveragingTransactionsHolder;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.memory.MemcpyDirection;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author raver119@gmail.com
 */
@Slf4j
@RunWith(Parameterized.class)
public class PerformanceTrackerTests extends BaseNd4jTest {
    public PerformanceTrackerTests(Nd4jBackend backend) {
        super(backend);
    }

    @Before
    public void setUp() throws Exception {
        PerformanceTracker.getInstance().clear();
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.BANDWIDTH);
    }

    @After
    public void tearDown() throws Exception {
        PerformanceTracker.getInstance().clear();
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.SCOPE_PANIC);
    }

    @Test
    public void testAveragedHolder_1() {
        val holder = new AveragingTransactionsHolder();

        holder.addValue(MemcpyDirection.HOST_TO_HOST,50L);
        holder.addValue(MemcpyDirection.HOST_TO_HOST,150L);

        assertEquals(100L, holder.getAverageValue(MemcpyDirection.HOST_TO_HOST).longValue());
    }

    @Test
    public void testAveragedHolder_2() {
        val holder = new AveragingTransactionsHolder();

        holder.addValue(MemcpyDirection.HOST_TO_HOST, 50L);
        holder.addValue(MemcpyDirection.HOST_TO_HOST,150L);
        holder.addValue(MemcpyDirection.HOST_TO_HOST,100L);

        assertEquals(100L, holder.getAverageValue(MemcpyDirection.HOST_TO_HOST).longValue());
    }

    @Test
    public void testPerformanceTracker_1() {
        val perf = PerformanceTracker.getInstance();

        // 100 nanoseconds spent for 5000 bytes. result should be around 50000 bytes per microsecond
        val res = perf.addMemoryTransaction(0, 100, 5000);
        assertEquals(50000, res);
    }

    @Test
    public void testPerformanceTracker_2() {
        val perf = PerformanceTracker.getInstance();

        // 10 nanoseconds spent for 5000 bytes. result should be around 500000 bytes per microsecond
        val res = perf.addMemoryTransaction(0, 10, 5000, MemcpyDirection.HOST_TO_HOST);
        assertEquals(500000, res);
    }

    @Test
    public void testPerformanceTracker_3() {
        val perf = PerformanceTracker.getInstance();

        // 10000 nanoseconds spent for 5000 bytes. result should be around 500 bytes per microsecond
        val res = perf.addMemoryTransaction(0, 10000, 5000);
        assertEquals(500, res);
    }

    @Test
    public void testTrackerCpu_1() {
        if (!Nd4j.getExecutioner().getClass().getCanonicalName().toLowerCase().contains("native"))
            return;

        val fa = new float[100000000];
        val array = Nd4j.create(fa, new int[]{10000, 10000});

        val map = PerformanceTracker.getInstance().getCurrentBandwidth();

        // getting H2H bandwidth
        val bw = map.get(0).get(MemcpyDirection.HOST_TO_HOST);
        log.info("H2H bandwidth: {}", map);

        assertTrue(bw > 0);
    }

    @Test
    public void testTrackerGpu_1() {
        if (!Nd4j.getExecutioner().getClass().getCanonicalName().toLowerCase().contains("cuda"))
            return;

        val fa = new float[100000000];
        val array = Nd4j.create(fa, new int[]{10000, 10000});

        val map = PerformanceTracker.getInstance().getCurrentBandwidth();

        // getting H2D bandwidth for device 0
        val bw = map.get(0).get(MemcpyDirection.HOST_TO_DEVICE);
        log.info("H2D bandwidth: {}", map);

        assertTrue(bw > 0);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
