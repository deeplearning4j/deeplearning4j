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
import org.nd4j.linalg.api.memory.WorkspaceUseMetaData;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
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
    private StackTraceElement[] allocationTrace;
    private WorkspaceUseMetaData workspaceUseMetaData;
    private String dataBuffer;

    public boolean dataHasDeallocationValues() {
        //detect patterns in data like e-323 (very small or large numbers) exponents with 3 digits
        //need to detect both negative and positive exponents
        //
        return Pattern.compile(".*e-\\d{3}.*").matcher(data).groupCount() > 0
                || Pattern.compile(".*e\\+\\d{3}.*").matcher(data).groupCount() > 0;
    }

    public static NDArrayMetaData empty() {
        return NDArrayMetaData.builder().build();
    }


    /**
     * Create an array of {@link NDArrayMetaData}
     * from the given list
     * @param arr the array to create the metadata from
     * @return
     */
    public static NDArrayMetaData[] fromArr(List<INDArray> arr) {
        List<INDArray> convert = new ArrayList<>();
        for(int i = 0; i < arr.size(); i++) {
            if(arr != null) {
                convert.add(arr.get(i));
            }
        }

        NDArrayMetaData[] ret = new NDArrayMetaData[convert.size()];
        for(int i = 0; i < convert.size(); i++) {
            ret[i] = from(convert.get(i));
        }
        return ret;
    }

    /**
     * Creates a singular array of {@link NDArrayMetaData}
     * from the given array
     * @param arr the array to create the metadata from
     * @return the array of metadata
     */
    public static NDArrayMetaData[] fromArr(INDArray arr) {
        return new NDArrayMetaData[] {from(arr)};
    }

    /**
     * Create an {@link NDArrayMetaData} from an {@link INDArray}
     * note that when creating this data all data will be stored on heap.
     * This logging is very expensive and is mainly for use to track down subtle
     * issues like underlying views changing.
     * @param arr the array to create the metadata from
     * @return
     */
    public static NDArrayMetaData from(INDArray arr) {
        return NDArrayMetaData.builder()
                .workspaceUseMetaData(WorkspaceUseMetaData.from(arr.getWorkspace()))
                .allocationTrace(arr.allocationTrace())
                .data(arr.isEmpty() ? "[]" : Nd4j.getEnvironment().isTruncateNDArrayLogStrings() ? arr.toString() : arr.toStringFull())
                .dataType(arr.dataType())
                .dataBuffer(arr.isEmpty() ? "[]" : arr.data().toString())
                .jvmShapeInfo(arr.shapeInfoJava())
                .id(arr.getId())
                .build();
    }
}
