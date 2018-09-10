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

package org.nd4j.parameterserver.distributed.logic.storage;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.parameterserver.distributed.logic.Storage;

import java.util.concurrent.ConcurrentHashMap;

/**
 * @author raver119@gmail.com
 */
@Slf4j
@Deprecated
public abstract class BaseStorage implements Storage {

    private ConcurrentHashMap<Integer, INDArray> storage = new ConcurrentHashMap<>();


    @Override
    public INDArray getArray(@NonNull Integer key) {
        return storage.get(key);
    }

    @Override
    public void setArray(@NonNull Integer key, @NonNull INDArray array) {
        storage.put(key, array);
    }

    @Override
    public boolean arrayExists(@NonNull Integer key) {
        return storage.containsKey(key);
    }

    @Override
    public void shutdown() {
        storage.clear();
    }
}
