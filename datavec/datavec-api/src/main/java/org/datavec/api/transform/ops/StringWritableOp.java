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
import org.datavec.api.writable.Writable;

/**
 * This class converts an {@link IAggregableReduceOp} operating on a String to one operating
 * on {@link Writable} instances. It's expected this will only work if that {@link Writable}
 * supports a conversion to TextWritable.
 * Created by huitseeker on 5/14/17.
 */
@AllArgsConstructor
@Data
public class StringWritableOp<T> implements IAggregableReduceOp<Writable, T> {

    @Getter
    private IAggregableReduceOp<String, T> operation;

    @Override
    public <W extends IAggregableReduceOp<Writable, T>> void combine(W accu) {
        if (accu instanceof StringWritableOp)
            operation.combine(((StringWritableOp) accu).getOperation());
        else
            throw new UnsupportedOperationException("Tried to combine() incompatible " + accu.getClass().getName()
                            + " operator where " + this.getClass().getName() + " expected");
    }

    @Override
    public void accept(Writable writable) {
        operation.accept(writable.toString());
    }

    @Override
    public T get() {
        return operation.get();
    }
}
