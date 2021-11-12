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
package org.nd4j.linalg.api.ops.impl.transforms.dtype;

import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

/**
 * Returns the min or max (0 or 1 in int arguments)
 * for a given data type.
 * The first data type is passed in as an int
 * represented in {@link DataType#fromInt(int)}
 *
 * The second value (as described above) is min or max
 * representing the value.
 * This value is returned as a scalar.
 */
public class MinMaxDataType extends DynamicCustomOp {
    public MinMaxDataType() {
        System.out.println();
    }

    public MinMaxDataType(SameDiff sd, int datatype, int minOrMax) {
        super(sd,null,false);
        addIArgument(datatype,minOrMax);
    }

    public MinMaxDataType(int datatype, int minOrMax) {
        super(null,null,null,false);
        addIArgument(datatype,minOrMax);
    }

    @Override
    public String opName() {
        return "min_max_datatype";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        //All scalar ops: output type is same as input type
        DataType dataType = DataType.fromInt(getIArgument(0).intValue());
        return Collections.singletonList(dataType);
    }

}
