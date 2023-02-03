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

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.AllocationsTracker;
import org.nd4j.linalg.api.memory.enums.AllocationKind;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.BaseShapeInfoProvider;
import org.nd4j.linalg.api.shape.ShapeDescriptor;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

@Slf4j
public class DirectShapeInfoProvider extends BaseShapeInfoProvider {
    private Map<LongShapeDescriptor, Pair<DataBuffer, long[]>> longCache = new ConcurrentHashMap<>();
    private AtomicInteger counter = new AtomicInteger(0);
    private static final int MAX_ENTRIES = 1000;

    public Pair<DataBuffer, long[]> createShapeInformation(long[] shape, long[] stride,  long elementWiseStride, char order, DataType dataType) {
        long extras = 0;
        extras = ArrayOptionsHelper.setOptionBit(extras, dataType);
        return createShapeInformation(shape, stride, elementWiseStride, order, extras);
    }

    @Override
    public Pair<DataBuffer, long[]> createShapeInformation(long[] shape, long[] stride,  long elementWiseStride, char order, long extras) {
        return super.createShapeInformation(shape, stride, elementWiseStride, order, extras);
    }

    @Override
    public void purgeCache() {
    }
}
