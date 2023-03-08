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

package org.nd4j.jita.constant;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.api.shape.options.ArrayType;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.BaseShapeInfoProvider;
import org.nd4j.linalg.factory.Nd4j;

import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicLong;

/**
 *
 * Thread safe cache for providing
 * shape info based on a long shape descriptor.
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class ProtectedCachedShapeInfoProvider extends BaseShapeInfoProvider {


    private AtomicLong cacheHit = new AtomicLong(1);
    private AtomicLong cacheMiss = new AtomicLong(1);

    private Semaphore lock = new Semaphore(1);

    protected static final ConstantProtector protector = ConstantProtector.getInstance();

    private static ProtectedCachedShapeInfoProvider ourInstance = new ProtectedCachedShapeInfoProvider();


    public ProtectedCachedShapeInfoProvider() {

    }

    /**
     * This method forces cache purge, if cache is available for specific implementation
     */
    @Override
    public void purgeCache() {
        protector.purgeProtector();
    }

    public static ProtectedCachedShapeInfoProvider getInstance() {
        return ourInstance;
    }


    @Override
    public Pair<DataBuffer, long[]> createShapeInformation(long[] shape, long[] stride, long elementWiseStride, char order, DataType type, boolean empty) {
        long extras = ArrayOptionsHelper.setOptionBit(0L, type);
        if (empty)
            extras = ArrayOptionsHelper.setOptionBit(extras, ArrayType.EMPTY);

        return createShapeInformation(shape, stride, elementWiseStride, order, extras);
    }

    @Override
    public Pair<DataBuffer, long[]> createShapeInformation(long[] shape, long[] stride, long elementWiseStride, char order, long extras) {
        // We enforce offset to 0 in shapeBuffer, since we need it for cache efficiency + we don't actually use offset value @ native side
        long offset = 0;
        if (elementWiseStride < 0)
            elementWiseStride = 0;

        Integer deviceId = Nd4j.getDeviceIdProvider().getDeviceId();

        LongShapeDescriptor descriptor = new LongShapeDescriptor(shape, stride, offset, elementWiseStride, order, extras);

        if (!protector.containsDataBuffer(deviceId, descriptor)) {
            Pair<DataBuffer, long[]> buffer = null;
            synchronized (this) {
                if (!protector.containsDataBuffer(deviceId, descriptor)) {
                    buffer = super.createShapeInformation(shape, stride, elementWiseStride, order, extras);
                    buffer.getFirst().setConstant(true);


                    protector.persistDataBuffer(deviceId, descriptor, buffer);

                    bytes.addAndGet(buffer.getFirst().length() * 8 * 2);

                    cacheMiss.incrementAndGet();
                } else {
                    buffer = protector.getDataBuffer(deviceId, descriptor);
                }
            }
            return buffer;
        } else {
            cacheHit.incrementAndGet();
        }

        return protector.getDataBuffer(deviceId, descriptor);
    }
}
