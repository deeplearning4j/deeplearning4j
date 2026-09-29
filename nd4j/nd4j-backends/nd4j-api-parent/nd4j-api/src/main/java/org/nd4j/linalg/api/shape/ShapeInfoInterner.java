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

package org.nd4j.linalg.api.shape;

import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Hands out one shared, constant shape information buffer per (device, exact shape info content).
 * <p>
 * Shape information buffers are immutable once built: arrays share them and never close them, and
 * address-keyed caches (TAD packs, DSP plans) treat the buffer address as the shape's identity. A
 * constant buffer is never returned to the deallocator, so building a fresh one per call leaks
 * native (and on CUDA, device) memory on every op that infers an output shape. Interning bounds that
 * cost by the number of distinct shapes, the same bound the native shape cache already has.
 * <p>
 * The content is interned exactly as given, with no normalization, so callers get back a buffer whose
 * values equal their input. The buffers are Java-owned copies rather than views of the native
 * {@code ConstantShapeHelper} cache, because that cache can be cleared and a view would then dangle.
 * The key includes the device because on CUDA each buffer carries a device allocation.
 */
public class ShapeInfoInterner {

    private static final Map<Key, DataBuffer> CACHE = new ConcurrentHashMap<>();

    private ShapeInfoInterner() {
    }

    /**
     * Returns the shared constant buffer for the given shape info content on the current device.
     *
     * @param shapeInfo shape info; values past {@code Shape.shapeInfoLength(shapeInfo[0])} are ignored
     * @return a constant buffer holding exactly the shape info content; callers must not modify or close it
     */
    public static DataBuffer intern(long[] shapeInfo) {
        Preconditions.checkArgument(shapeInfo != null && shapeInfo.length > 0, "Shape info must not be null or empty");
        int length = Shape.shapeInfoLength(shapeInfo[0]);
        Preconditions.checkArgument(shapeInfo.length >= length,
                "Shape info of rank %s needs %s values but has %s", shapeInfo[0], length, shapeInfo.length);

        Key key = new Key(Nd4j.getDeviceIdProvider().getDeviceId(), Arrays.copyOf(shapeInfo, length));
        DataBuffer cached = CACHE.get(key);
        return cached != null ? cached : CACHE.computeIfAbsent(key, ShapeInfoInterner::create);
    }

    private static DataBuffer create(Key key) {
        // The native shape cache rejects invalid ranks and dimensions, so every distinct shape is validated once.
        Nd4j.getNativeOps().cacheAndStoreShapeBuffer(key.shapeInfo);

        // Shape info outlives any workspace cycle, so it must not be allocated from workspace memory.
        DataBuffer buffer;
        try (MemoryWorkspace ignored = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
            buffer = Nd4j.createBuffer(key.shapeInfo);
        }
        buffer.setConstant(true);
        return buffer;
    }

    private static final class Key {
        private final int deviceId;
        private final long[] shapeInfo;
        private final int hash;

        private Key(int deviceId, long[] shapeInfo) {
            this.deviceId = deviceId;
            this.shapeInfo = shapeInfo;
            this.hash = 31 * deviceId + Arrays.hashCode(shapeInfo);
        }

        @Override
        public boolean equals(Object o) {
            if (this == o)
                return true;
            if (!(o instanceof Key))
                return false;
            Key other = (Key) o;
            return deviceId == other.deviceId && Arrays.equals(shapeInfo, other.shapeInfo);
        }

        @Override
        public int hashCode() {
            return hash;
        }
    }
}
