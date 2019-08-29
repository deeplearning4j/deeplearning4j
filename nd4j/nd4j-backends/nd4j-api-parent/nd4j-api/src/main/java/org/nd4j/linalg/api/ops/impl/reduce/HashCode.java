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

package org.nd4j.linalg.api.ops.impl.reduce;

import lombok.NonNull;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;

import java.util.Collections;
import java.util.List;

/**
 * This is hashCode op wrapper. Basically - simple parallel hash implementation.
 *
 * @author raver119@gmail.com
 */
public class HashCode extends DynamicCustomOp {
    public HashCode() {
        //
    }

    public HashCode(@NonNull INDArray array) {
        this.inputArguments.add(array);
    }

    public HashCode(@NonNull INDArray array, @NonNull INDArray result) {
        this(array);
        Preconditions.checkArgument(result.dataType() == DataType.LONG && result.isScalar(), "HashCode op expects LONG scalar as output");

        this.outputArguments.add(result);
    }

    @Override
    public String opName() {
        return "hashcode";
    }
}
