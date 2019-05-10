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

package org.datavec.api.transform.metadata;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.datavec.api.transform.ColumnType;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Arrays;

/**
 * Meta data class for NDArray columns
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(callSuper = true)
@JsonIgnoreProperties("allowVarLength")
public class NDArrayMetaData extends BaseColumnMetaData {

    private long[] shape;
    private boolean allowVarLength;


    /**
     * @param name  Name of the NDArray column
     * @param shape shape of the NDArray column. Use -1 in entries to specify as "variable length" in that dimension
     */
    public NDArrayMetaData(@JsonProperty("name") String name, @JsonProperty("shape") long[] shape) {
        super(name);
        this.shape = shape;
        for (long i : shape) {
            if (i < 0) {
                allowVarLength = true;
                break;
            }
        }
    }

    @Override
    public ColumnType getColumnType() {
        return ColumnType.NDArray;
    }

    @Override
    public boolean isValid(Writable writable) {
        if (!(writable instanceof NDArrayWritable)) {
            return false;
        }
        INDArray arr = ((NDArrayWritable) writable).get();
        if (arr == null) {
            return false;
        }
        if (allowVarLength) {
            for (int i = 0; i < shape.length; i++) {
                if (shape[i] < 0) {
                    continue;
                }
                if (shape[i] != arr.size(i)) {
                    return false;
                }
            }
            return true;
        } else {
            return Arrays.equals(shape, arr.shape());
        }
    }

    @Override
    public boolean isValid(Object input) {
        if (input == null) {
            return false;
        } else if (input instanceof Writable) {
            return isValid((Writable) input);
        } else if (input instanceof INDArray) {
            return isValid(new NDArrayWritable((INDArray) input));
        } else {
            throw new UnsupportedOperationException("Unknown object type: " + input.getClass());
        }
    }

    @Override
    public NDArrayMetaData clone() {
        return new NDArrayMetaData(name, shape.clone());
    }

}
