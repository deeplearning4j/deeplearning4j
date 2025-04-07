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

package org.nd4j.jita.allocator.tad;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.common.primitives.Pair;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cache.TadDescriptor;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Semaphore;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class DeviceTADManager extends BasicTADManager {


    public DeviceTADManager() {

    }

    /**
     * This method removes all cached shape buffers
     */
    @Override
    public void purgeBuffers() {
        log.info("Purging TAD buffers...");
        super.purgeBuffers();
    }

    @Override
    public Pair<DataBuffer, DataBuffer> getTADOnlyShapeInfo(INDArray array, long... dimension) {
        if (dimension != null && dimension.length > 1)
            Arrays.sort(dimension);

        if (dimension == null)
            dimension = new long[] {Integer.MAX_VALUE};

        val pack = Nd4j.getExecutioner().tadShapeInfoAndOffsets(array, dimension);
        DataBuffer tadShapeInfo = pack.getTadShapeInfo();
        DataBuffer offsets = pack.getTadOffsets();
        return new Pair<>(tadShapeInfo, offsets);
    }
}
