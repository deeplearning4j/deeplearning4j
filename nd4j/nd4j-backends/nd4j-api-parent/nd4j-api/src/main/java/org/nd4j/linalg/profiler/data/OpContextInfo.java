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
package org.nd4j.linalg.profiler.data;

import lombok.Builder;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Metadata containing information about an op context.
 * Used with {@link org.nd4j.linalg.profiler.OpContextTracker}
 * to track op context information about the inputs and outputs.
 * This information includes:
 * 1. data type
 * 2. size in bytes
 *
 * @author Adam Gibson
 */
@Slf4j
@Data
@Builder
public class OpContextInfo {

    @Builder.Default
    private List<DataType> inputTypes = new ArrayList<>();
    @Builder.Default
    private List<DataType> outputTypes = new ArrayList<>();
    @Builder.Default
    private List<Long> inputByteWidths = new ArrayList<>();
    @Builder.Default
    private List<Long> outputByteWidths = new ArrayList<>();
    @Builder.Default
    private List<Boolean> outputCreatedByCtx = new ArrayList<>();
    private long id;

    @Builder.Default
    private boolean allocated = true;





    public void purge() {
        inputTypes.clear();
        outputTypes.clear();
        outputByteWidths.clear();
        inputByteWidths.clear();
        outputCreatedByCtx.clear();
    }


    public void addOutput(INDArray input,boolean createdByOutput) {
       if(input == null)
           return;
        addOutputType(input.dataType());
        outputCreatedByCtx.add(createdByOutput);
        if(!input.isEmpty())
            addOutputByteWidth(input.data().length() * input.data().dataType().width());

        else  addOutputByteWidth(0L);

    }

    public void addInput(INDArray input) {
      if(input == null)
          return;
        addInputType(input.dataType());
        if(!input.isEmpty())
            addInputByteWidth(input.data().length() * input.data().dataType().width());
        else  addInputByteWidth(0L);

    }

    public void addInputType(DataType input) {
        inputTypes.add(input);
    }

    public void addOutputType(DataType output) {
        outputTypes.add(output);
    }


    public void addOutputByteWidth(Long outputByteWidth) {
        outputByteWidths.add(outputByteWidth);
    }

    public void addInputByteWidth(Long inputByteWidth) {
        inputByteWidths.add(inputByteWidth);
    }


    public void deallocate() {
        allocated = false;
    }


    public String inputAt(int index) {
        return "Type: " + inputTypes.get(index) + " Size in bytes:" + inputByteWidths.get(index);
    }

    public String outputAt(int index) {
        return "Type: " + outputTypes.get(index) + " Size in bytes: " + outputByteWidths.get(index) + " Allocated by ctx: " + outputCreatedByCtx.get(index);
    }


    public long allocatedInputBytes() {
        return inputByteWidths.stream().collect(Collectors.summingLong(Long::longValue));
    }

    public long allocatedOutputBytes() {
        return outputByteWidths.stream().collect(Collectors.summingLong(Long::longValue));
    }

    public String toString() {
        StringBuilder ret = new StringBuilder();
        ret.append("---------------------------\n");
        ret.append("id: " + id + " ");
        ret.append("Allocated: " + allocated + "\n");
        ret.append("Inputs:\n");
        for(int i = 0; i < inputByteWidths.size(); i++) {
            ret.append(inputAt(i) + "\n");
        }
        ret.append("\n");
        ret.append("Outputs:\n");
        for(int i = 0;  i < outputByteWidths.size(); i++) {
            ret.append(outputAt(i) + "\n");
        }

        ret.append("---------------------------\n");
        return ret.toString();

    }

}
