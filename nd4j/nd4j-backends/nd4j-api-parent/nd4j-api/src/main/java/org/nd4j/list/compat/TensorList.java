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

package org.nd4j.list.compat;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

/**
 * This class implements basic storage for NDArrays
 */
@Slf4j
public class TensorList {
    private final String name;
    private ArrayList<INDArray> list = new ArrayList<>();

    public TensorList(@NonNull String name) {
        this.name = name;
    }

    public TensorList(@NonNull String name, @NonNull INDArray source) {
        this.name = name;
    }

    public INDArray get(int index) {
        return list.get(index);
    }

    public void put(int index, @NonNull INDArray array) {
        // TODO: if we want to validate shape - we should do it here

        list.ensureCapacity(index + 1);
        list.add(index, array.dup());
    }

    public INDArray stack() {
        return Nd4j.pile(list);
    }

    public INDArray gather(INDArray indices) {
        if(indices.length() == 1 && indices.getInt(0) == -1){
            return stack();
        }
        val idxs = indices.reshape(indices.length()).toIntVector();
        ArrayList<INDArray> newList = new ArrayList<>();
        for(val id : idxs){
            newList.add(list.get(id));
        }
        return Nd4j.pile(newList);
    }

    public INDArray concat(){
       return Nd4j.concat(0, (INDArray[]) list.toArray());
    }

    public int size() {
        return list.size();
    }

    public String getName() {
        return name;
    }
}
