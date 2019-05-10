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

package org.nd4j.jita.constant;

import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.shape.ShapeDescriptor;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * This class implements storage singleton, to guarantee constant buffers persistence
 *
 * @author raver119@gmail.com
 */
public class ConstantProtector {
    private static ConstantProtector ourInstance = new ConstantProtector();

    public static ConstantProtector getInstance() {
        return ourInstance;
    }

    private List<DataBuffer> protectorLegacy = new CopyOnWriteArrayList<>();
    private List<Pair<DataBuffer, long[]>> protector = new CopyOnWriteArrayList<>();
    private List<Map<LongShapeDescriptor, Pair<DataBuffer, long[]>>> deviceCache = new ArrayList<>();


    private ConstantProtector() {
        purgeProtector();
    }

    public void purgeProtector() {
        protector = new CopyOnWriteArrayList<>();
        deviceCache = new ArrayList<>();

        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();

        for (int i = 0; i < numDevices; i++) {
            deviceCache.add(i, new ConcurrentHashMap<LongShapeDescriptor, Pair<DataBuffer, long[]>>());
        }
    }

    public void persistDataBuffer(DataBuffer buffer) {
        protectorLegacy.add(buffer);
    }

    public void persistDataBuffer(Pair<DataBuffer, long[]> buffer) {
        protector.add(buffer);
    }

    public void persistDataBuffer(int deviceId, ShapeDescriptor descriptor, Pair<DataBuffer, long[]> buffer) {
        deviceCache.get(deviceId).put(LongShapeDescriptor.fromShapeDescriptor(descriptor), buffer);
    }

    public void persistDataBuffer(int deviceId, LongShapeDescriptor descriptor, Pair<DataBuffer, long[]> buffer) {
        deviceCache.get(deviceId).put(descriptor, buffer);
    }

    public Pair<DataBuffer, long[]> getDataBuffer(int deviceId, ShapeDescriptor descriptor) {
        return deviceCache.get(deviceId).get(LongShapeDescriptor.fromShapeDescriptor(descriptor));
    }

    public Pair<DataBuffer, long[]> getDataBuffer(int deviceId, LongShapeDescriptor descriptor) {
        return deviceCache.get(deviceId).get(descriptor);
    }

    public boolean containsDataBuffer(int deviceId, ShapeDescriptor descriptor) {
        return deviceCache.get(deviceId).containsKey(LongShapeDescriptor.fromShapeDescriptor(descriptor));
    }

    public boolean containsDataBuffer(int deviceId, LongShapeDescriptor descriptor) {
        return deviceCache.get(deviceId).containsKey(descriptor);
    }


}
