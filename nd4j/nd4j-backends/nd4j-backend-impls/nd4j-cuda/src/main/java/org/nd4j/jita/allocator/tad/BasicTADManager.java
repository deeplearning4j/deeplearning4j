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

package org.nd4j.jita.allocator.tad;

import lombok.val;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cache.TADManager;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author raver119@gmail.com
 */
public class BasicTADManager implements TADManager {
    protected NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
    private static Logger logger = LoggerFactory.getLogger(BasicTADManager.class);
    protected AtomicLong bytes = new AtomicLong(0);

    @Override
    public Pair<DataBuffer, DataBuffer> getTADOnlyShapeInfo(INDArray array, int[] dimension) {
        if (dimension != null && dimension.length > 1)
            Arrays.sort(dimension);

        if (dimension == null)
            dimension = new int[] {Integer.MAX_VALUE};

        val pack = Nd4j.getExecutioner().tadShapeInfoAndOffsets(array, dimension);

        //   logger.info("TAD shapeInfo after construction: {}", Arrays.toString(TadDescriptor.dataBufferToArray(outputBuffer)));
        // now we need to copy this buffer to either device global memory or device cache

        return new Pair<>(pack.getTadShapeInfo(), pack.getTadOffsets());
    }

    /**
     * This method removes all cached shape buffers
     */
    @Override
    public void purgeBuffers() {
        // no-op
    }

    @Override
    public long getCachedBytes() {
        return bytes.get();
    }
}
