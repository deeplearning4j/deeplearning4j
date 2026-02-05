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
 *  ******************************************************************************
 */

package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.InferenceFactory;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.autodiff.samediff.internal.memory.ArrayCacheMemoryMgr;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

@NativeTag
@Tag(TagNames.SAMEDIFF)
public class DynamicShapePlanPoolingTest extends BaseNd4jTestWithBackends {

    private static final String DYNAMIC_SHAPE_PROP = "org.nd4j.inference.dynamicShapePlan";

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDynamicShapePlanEnablesCacheAndGrowth(Nd4jBackend backend) {
        InferenceFactory prevFactory = SameDiff.getInferenceFactory();
        String prevDynamicProp = System.getProperty(DYNAMIC_SHAPE_PROP);
        boolean prevCache = ArrayCacheMemoryMgr.isCacheEnabled();
        double prevGrowth = ArrayCacheMemoryMgr.getGrowthFactor().get();
        boolean prevDynamicEnabled = InferenceSession.isDynamicShapePlanEnabled();

        try {
            System.setProperty(DYNAMIC_SHAPE_PROP, "true");
            InferenceSession.setDynamicShapePlanEnabled(true);
            ArrayCacheMemoryMgr.setEnableCache(false);
            ArrayCacheMemoryMgr.setGrowthFactor(1.0);

            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 4);
            SDVariable y = x.add("y", 1.0);

            INDArray input = Nd4j.ones(DataType.FLOAT, 1, 4);
            Map<String, INDArray> placeholders = new LinkedHashMap<>();
            placeholders.put("x", input);
            Map<String, INDArray> outputs = sd.output(placeholders, "y");
            assertEquals(input.add(1.0), outputs.get("y"));

            assertTrue(ArrayCacheMemoryMgr.isCacheEnabled(), "DynamicShapePlan should force-enable cache");
            assertEquals(1.1, ArrayCacheMemoryMgr.getGrowthFactor().get(), 1e-6,
                    "DynamicShapePlan should set growthFactor to 1.1 when <= 1.0");
        } finally {
            restoreProperty(DYNAMIC_SHAPE_PROP, prevDynamicProp);
            ArrayCacheMemoryMgr.setEnableCache(prevCache);
            ArrayCacheMemoryMgr.setGrowthFactor(prevGrowth);
            InferenceSession.setDynamicShapePlanEnabled(prevDynamicEnabled);
            SameDiff.bindInferenceFactory(prevFactory);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDynamicShapePlanReusesBuffersWithinRun(Nd4jBackend backend) {
        InferenceFactory prevFactory = SameDiff.getInferenceFactory();
        String prevDynamicProp = System.getProperty(DYNAMIC_SHAPE_PROP);
        boolean prevCache = ArrayCacheMemoryMgr.isCacheEnabled();
        double prevGrowth = ArrayCacheMemoryMgr.getGrowthFactor().get();
        boolean prevDynamicEnabled = InferenceSession.isDynamicShapePlanEnabled();

        CountingInferenceFactory factory = new CountingInferenceFactory();
        try {
            System.setProperty(DYNAMIC_SHAPE_PROP, "true");
            InferenceSession.setDynamicShapePlanEnabled(true);
            ArrayCacheMemoryMgr.setEnableCache(false);
            ArrayCacheMemoryMgr.setGrowthFactor(1.0);
            SameDiff.bindInferenceFactory(factory);

            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 4);
            SDVariable a = x.add("a", 1.0);
            SDVariable b = a.mul("b", 2.0);
            SDVariable c = b.add("c", 3.0);

            INDArray input = Nd4j.linspace(1.0, 8.0, 8, DataType.FLOAT).reshape(2, 4);
            Map<String, INDArray> placeholders = new LinkedHashMap<>();
            placeholders.put("x", input);
            Map<String, INDArray> outputs = sd.output(placeholders, "c");
            assertEquals(input.add(1.0).mul(2.0).add(3.0), outputs.get("c"));

            CountingArrayCacheMemoryMgr memMgr = factory.getLastMemMgr();
            assertNotNull(memMgr, "Expected CountingArrayCacheMemoryMgr from factory");
            int allocCount = memMgr.getAllocateCount();
            assertTrue(allocCount <= 2, "Expected buffer reuse within run (allocations=" + allocCount + ")");
        } finally {
            restoreProperty(DYNAMIC_SHAPE_PROP, prevDynamicProp);
            ArrayCacheMemoryMgr.setEnableCache(prevCache);
            ArrayCacheMemoryMgr.setGrowthFactor(prevGrowth);
            InferenceSession.setDynamicShapePlanEnabled(prevDynamicEnabled);
            SameDiff.bindInferenceFactory(prevFactory);
        }
    }

    @Override
    public char ordering() {
        return 'c';
    }

    private static void restoreProperty(String key, String previous) {
        if (previous == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, previous);
        }
    }

    private static final class CountingInferenceFactory implements InferenceFactory {
        private CountingArrayCacheMemoryMgr lastMemMgr;

        @Override
        public InferenceSession create(SameDiff sameDiff) {
            lastMemMgr = new CountingArrayCacheMemoryMgr();
            return new InferenceSession(sameDiff, lastMemMgr);
        }

        CountingArrayCacheMemoryMgr getLastMemMgr() {
            return lastMemMgr;
        }
    }

    private static final class CountingArrayCacheMemoryMgr extends ArrayCacheMemoryMgr {
        private final AtomicInteger allocateCount = new AtomicInteger();

        @Override
        public INDArray allocate(boolean detached, DataType dataType, long... shape) {
            allocateCount.incrementAndGet();
            return super.allocate(detached, dataType, shape);
        }

        @Override
        public INDArray allocate(boolean detached, LongShapeDescriptor descriptor) {
            allocateCount.incrementAndGet();
            return super.allocate(detached, descriptor);
        }

        int getAllocateCount() {
            return allocateCount.get();
        }
    }
}
