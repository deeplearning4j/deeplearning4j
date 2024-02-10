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
package org.nd4j.linalg.profiler.data.array.event;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;
import java.util.regex.Pattern;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class NDArrayMetaData implements Serializable {
    private String data;
    private DataType dataType;
    private long[] jvmShapeInfo;
    private long id;


    public boolean dataHasDeallocatioValues() {
        //detect patterns in data like e-323 (very small or large numbers) exponents with 3 digits
        //need to detect both negative and positive exponents
        //
        return Pattern.compile(".*e-\\d{3}.*").matcher(data).groupCount() > 0
                || Pattern.compile(".*e\\+\\d{3}.*").matcher(data).groupCount() > 0;
    }

    public static NDArrayMetaData empty() {
        return NDArrayMetaData.builder().build();
    }

    public static NDArrayMetaData from(INDArray arr) {
        return NDArrayMetaData.builder()
                .data(arr.toStringFull())
                .dataType(arr.dataType())
                .jvmShapeInfo(arr.shapeInfoJava())
                .id(arr.getId())
                .build();
    }
}
