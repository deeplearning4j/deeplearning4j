/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.profiling;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.ops.performance.PerformanceTracker;
import org.nd4j.linalg.api.ops.performance.primitives.AveragingTransactionsHolder;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.api.memory.MemcpyDirection;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

@Slf4j
@NativeTag
public class PerformanceTrackerTests extends BaseNd4jTestWithBackends {

    @BeforeEach
    public void setUp() {
        PerformanceTracker.getInstance().clear();
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.BANDWIDTH);
    }

    @AfterEach
    public void tearDown() {
        PerformanceTracker.getInstance().clear();
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.SCOPE_PANIC);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAveragedHolder_1(Nd4jBackend backend) {
        val holder = new AveragingTransactionsHolder();

        holder.addValue(MemcpyDirection.HOST_TO_HOST,50L);
        holder.addValue(MemcpyDirection.HOST_TO_HOST,150L);

        assertEquals(100L, holder.getAverageValue(MemcpyDirection.HOST_TO_HOST).longValue());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAveragedHolder_2(Nd4jBackend backend) {
        val holder = new AveragingTransactionsHolder();

        holder.addValue(MemcpyDirection.HOST_TO_HOST, 50L);
        holder.addValue(MemcpyDirection.HOST_TO_HOST,150L);
        holder.addValue(MemcpyDirection.HOST_TO_HOST,100L);

        assertEquals(100L, holder.getAverageValue(MemcpyDirection.HOST_TO_HOST).longValue());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPerformanceTracker_1(Nd4jBackend backend) {
        val perf = PerformanceTracker.getInstance();

        // 100 nanoseconds spent for 5000 bytes. result should be around 50000 bytes per microsecond
        val res = perf.addMemoryTransaction(0, 100, 5000);
        assertEquals(50000, res);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPerformanceTracker_2(Nd4jBackend backend) {
        val perf = PerformanceTracker.getInstance();

        // 10 nanoseconds spent for 5000 bytes. result should be around 500000 bytes per microsecond
        val res = perf.addMemoryTransaction(0, 10, 5000, MemcpyDirection.HOST_TO_HOST);
        assertEquals(500000, res);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPerformanceTracker_3(Nd4jBackend backend) {
        val perf = PerformanceTracker.getInstance();

        // 10000 nanoseconds spent for 5000 bytes. result should be around 500 bytes per microsecond
        val res = perf.addMemoryTransaction(0, 10000, 5000);
        assertEquals(500, res);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Disabled
    public void testTrackerCpu_1(Nd4jBackend backend) {
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

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Disabled("useless these days")
    public void testTrackerGpu_1(Nd4jBackend backend) {
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
