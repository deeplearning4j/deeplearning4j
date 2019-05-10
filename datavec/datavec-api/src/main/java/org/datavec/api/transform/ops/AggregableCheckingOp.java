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
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.writable.Writable;

/**
 * A variant of {@link IAggregableReduceOp} exercised on a {@link Writable} that takes schema metadata
 * in its constructor, and checks the input {@link Writable} against the schema before accepting it.
 *
 * Created by huitseeker on 5/8/17.
 */
@AllArgsConstructor
@Data
public class AggregableCheckingOp<T> implements IAggregableReduceOp<Writable, T> {

    @Getter
    private IAggregableReduceOp<Writable, T> operation;
    @Getter
    private ColumnMetaData metaData;

    @Override
    public <W extends IAggregableReduceOp<Writable, T>> void combine(W accu) {
        if (accu instanceof AggregableCheckingOp) {
            AggregableCheckingOp<T> accumulator = (AggregableCheckingOp) accu;
            if (metaData.getColumnType() != accumulator.getMetaData().getColumnType())
                throw new IllegalArgumentException(
                                "Invalid merge with operation on " + accumulator.getMetaData().getName() + " of type "
                                                + accumulator.getMetaData().getColumnType() + " expected "
                                                + metaData.getName() + " of type " + metaData.getColumnType());
            else
                operation.combine(accumulator);
        } else
            throw new UnsupportedOperationException("Tried to combine() incompatible " + accu.getClass().getName()
                            + " operator where " + this.getClass().getName() + " expected");
    }

    @Override
    public void accept(Writable writable) {
        if (metaData.isValid(writable))
            operation.accept(writable);
    }

    @Override
    public T get() {
        return operation.get();
    }
}
