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

package org.datavec.api.transform.ops;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.Getter;
import lombok.NonNull;
import org.datavec.api.writable.Writable;

import java.util.ArrayList;
import java.util.List;

/**
 * This class transforms a list of {@link IAggregableReduceOp} on one single field, each returning a {@link Writable}
 * and transforms it into an operation on that single column, that returns a {@link Writable} list.
 *
 * It is used to execute many reduction operations in parallel on the same column, datavec#238
 *
 * Created by huitseeker on 5/8/17.
 */
@AllArgsConstructor
@Data
public class AggregableMultiOp<T> implements IAggregableReduceOp<T, List<Writable>> {

    @Getter
    @NonNull
    private List<IAggregableReduceOp<T, Writable>> operations;

    public void accept(T t) {
        for (int i = 0; i < operations.size(); i++) {
            operations.get(i).accept(t);
        }
    }

    public <U extends IAggregableReduceOp<T, List<Writable>>> void combine(U accu) {
        if (accu instanceof AggregableMultiOp) {
            AggregableMultiOp<T> accumulator = (AggregableMultiOp<T>) accu;
            List<IAggregableReduceOp<T, Writable>> otherAccumulators = accumulator.getOperations();
            if (operations.size() != otherAccumulators.size())
                throw new IllegalArgumentException("Tried to combine() incompatible " + this.getClass().getName()
                                + " operators: received " + otherAccumulators.size() + " operations, expected "
                                + operations.size());
            for (int i = 0; i < operations.size(); i++) {
                operations.get(i).combine(otherAccumulators.get(i));
            }
        } else
            throw new UnsupportedOperationException("Tried to combine() incompatible " + accu.getClass().getName()
                            + " operator where " + this.getClass().getName() + " expected");
    }

    public List<Writable> get() {
        List<Writable> res = new ArrayList<>(operations.size());
        for (int i = 0; i < operations.size(); i++) {
            res.add(operations.get(i).get());
        }
        return res;
    }

}
