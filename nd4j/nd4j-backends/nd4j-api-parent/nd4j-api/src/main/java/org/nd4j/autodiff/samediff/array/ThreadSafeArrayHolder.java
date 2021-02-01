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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.DeviceLocalNDArray;

import java.util.Collection;
import java.util.Collections;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * An {@link ArrayHolder} that uses the thread safe {@link DeviceLocalNDArray} internally
 *
 * @author Alex Black
 */
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
        return map.get(name).get();
    }

    @Override
    public void setArray(@NonNull String name, @NonNull INDArray array) {
        if (array.isView())
            array = array.dup();    //Device local doesn't support views
        if (!map.containsKey(name)) {
            INDArray toBroadcast = array.dataType() == DataType.UTF8 ? array.dup() : array;
            DeviceLocalNDArray dla = new DeviceLocalNDArray(toBroadcast, lazyInit);
            map.put(name, dla);
        } else {
            DeviceLocalNDArray dla = map.get(name);
            dla.update(array);
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
