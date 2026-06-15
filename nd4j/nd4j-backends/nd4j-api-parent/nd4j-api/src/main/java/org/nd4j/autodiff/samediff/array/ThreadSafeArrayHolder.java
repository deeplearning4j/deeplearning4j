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

package org.nd4j.autodiff.samediff.array;

import lombok.NonNull;
import org.nd4j.autodiff.samediff.ArrayHolder;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.DeviceLocalNDArray;

import java.util.Collection;
import java.util.Collections;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class ThreadSafeArrayHolder implements ArrayHolder {

    private final Map<String, DeviceLocalNDArray> map = new ConcurrentHashMap<>();
    private final boolean lazyInit;

    /**
     * @param lazyInit If true: use lazy initialization for {@link DeviceLocalNDArray}
     */
    public ThreadSafeArrayHolder(boolean lazyInit) {
        this.lazyInit = lazyInit;
    }

    @Override
    public boolean hasArray(@NonNull String name) {
        return map.containsKey(name);
    }

    @Override
    public INDArray getArray(@NonNull String name) {
        if(!map.containsKey(name))
            return null;
        return map.get(name).get();
    }

    @Override
    public void setArray(@NonNull String name, @NonNull INDArray array) {
        // Check if source array is constant - we'll propagate this flag to copies.
        // This is important because ThreadSafeArrayHolder is used for BOTH constantArrays
        // (which SHOULD be constant) and variablesArrays (which should NOT be constant).
        // The caller (SameDiff.setArrayForVariable) is responsible for marking constants.
        boolean sourceIsConstant = array.data() != null && array.data().isConstant();

        if (array.isView()) {
            try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                array = array.dup();    //Device local doesn't support views
            }
            // Only mark view duplicates as constant if the source was constant.
            // This preserves the distinction between CONSTANT and VARIABLE arrays.
            if (sourceIsConstant) {
                array = Nd4j.getDeallocatorService().registerPendingConstant(array);
                try {
                    if (array.data() != null) {
                        array.data().setConstant(true);
                    }
                    if (array.shapeInfoDataBuffer() != null) {
                        array.shapeInfoDataBuffer().setConstant(true);
                    }
                    array.setCloseable(false);
                } finally {
                    Nd4j.getDeallocatorService().releasePendingConstant(array);
                }
            }
        }
        if (!map.containsKey(name)) {
            INDArray toBroadcast;
            try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                toBroadcast = array.detach();
            }

            if (sourceIsConstant) {
                toBroadcast = Nd4j.getDeallocatorService().registerPendingConstant(toBroadcast);
                try {
                    if (toBroadcast.data() != null) {
                        toBroadcast.data().setConstant(true);
                    }
                    if (toBroadcast.shapeInfoDataBuffer() != null) {
                        toBroadcast.shapeInfoDataBuffer().setConstant(true);
                    }
                    toBroadcast.setCloseable(false);
                } finally {
                    Nd4j.getDeallocatorService().releasePendingConstant(toBroadcast);
                }
            }

            DeviceLocalNDArray dla = new DeviceLocalNDArray(toBroadcast, lazyInit);
            map.put(name, dla);
        } else {
            DeviceLocalNDArray dla = map.get(name);
            INDArray toUpdate;
            try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                toUpdate = array.detach();
            }

            // Propagate constant flag for update path too.
            if (sourceIsConstant) {
                toUpdate = Nd4j.getDeallocatorService().registerPendingConstant(toUpdate);
                try {
                    if (toUpdate.data() != null) {
                        toUpdate.data().setConstant(true);
                    }
                    if (toUpdate.shapeInfoDataBuffer() != null) {
                        toUpdate.shapeInfoDataBuffer().setConstant(true);
                    }
                    toUpdate.setCloseable(false);
                } finally {
                    Nd4j.getDeallocatorService().releasePendingConstant(toUpdate);
                }
            }

            dla.update(toUpdate);
        }
    }

    @Override
    public INDArray removeArray(@NonNull String name) {
        DeviceLocalNDArray arr = map.remove(name);
        if (arr == null)
            return null;
        return arr.get();
    }

    @Override
    public int size() {
        return map.size();
    }

    @Override
    public void initFrom(ArrayHolder arrayHolder) {
        map.clear();
        Collection<String> names = arrayHolder.arrayNames();
        for (String n : names) {
            setArray(n, arrayHolder.getArray(n));
        }
    }

    @Override
    public Collection<String> arrayNames() {
        return Collections.unmodifiableCollection(map.keySet());
    }

    @Override
    public void rename(@NonNull String from, @NonNull String to) {
        DeviceLocalNDArray dl = map.remove(from);
        map.put(to, dl);
    }
}
