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

package org.nd4j.linalg.cpu.nativecpu;

import lombok.NonNull;
import lombok.val;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cache.ConstantHandler;
import org.nd4j.linalg.cache.TADManager;
import org.nd4j.linalg.cache.TadDescriptor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.primitives.Pair;
import org.nd4j.nativeblas.NativeOps;

import java.util.Arrays;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

public class CpuTADManager implements TADManager {
    private Map<TadDescriptor, Pair<DataBuffer, DataBuffer>> cache = new ConcurrentHashMap<>();
    private AtomicLong bytes = new AtomicLong(0);

    public CpuTADManager() {
        //
    }

    public void init(@NonNull NativeOps nativeOps, @NonNull ConstantHandler constantHandler) {
    }

    /**
     * This method removes all cached shape buffers
     */
    @Override
    public void purgeBuffers() {
        cache = new ConcurrentHashMap<>();
    }

    @Override
    public Pair<DataBuffer, DataBuffer> getTADOnlyShapeInfo(INDArray array, long... dimension) {
        if (dimension != null && dimension.length > 1)
            Arrays.sort(dimension);

        if (dimension == null)
            dimension = new long[] {Integer.MAX_VALUE};

        val pack = Nd4j.getExecutioner().tadShapeInfoAndOffsets(array, dimension);

        // now we need to copy this buffer to either device global memory or device cache

        return new Pair<>(pack.getTadShapeInfo(), pack.getTadOffsets());

    }

    @Override
    public long getCachedBytes() {
        return bytes.get();
    }
}
