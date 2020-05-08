/*******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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

package org.nd4j.linalg.api.ops.compression;

import lombok.NonNull;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Bitmap encoding op wrapper. Used in gradients sharing.
 * @author raver119@gmail.com
 */
public class EncodeBitmap extends DynamicCustomOp {
    protected float threshold = 1e-3f;

    public EncodeBitmap() {
        //
    }

    public EncodeBitmap(@NonNull INDArray updates, float threshold) {
        this(updates, Nd4j.create(DataType.INT32, updates.length() / 16 + 5), Nd4j.scalar(DataType.INT32, 0), threshold);
    }

    public EncodeBitmap(@NonNull INDArray updates, @NonNull INDArray encoded, @NonNull INDArray counter, float threshold) {
        addInputArgument(updates);
        addOutputArgument(updates, encoded, counter);
        addTArgument(threshold);

        this.threshold = threshold;

        // this op ALWAYS modifies updates array
        setInPlace(true);
    }

    @Override
    public String opName() {
        return "encode_bitmap";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        return Arrays.asList(inputArguments.get(0).dataType(), DataType.INT32, DataType.INT32);
    }
}
