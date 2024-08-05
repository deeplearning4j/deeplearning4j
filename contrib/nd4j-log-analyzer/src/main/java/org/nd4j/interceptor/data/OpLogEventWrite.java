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
package org.nd4j.interceptor.data;

import lombok.*;
import org.nd4j.shade.jackson.databind.annotation.JsonSerialize;

import java.util.LinkedHashMap;
import java.util.Map;

@NoArgsConstructor
@AllArgsConstructor
@Builder
@Getter
@Setter
@ToString
@JsonSerialize(using = OpLogEventWriteSerializer.class)
public class OpLogEventWrite {

    public String opName;

    @Builder.Default
    @JsonSerialize(using = OpLogEvent.InputOutputSerializer.class)
    public Map<Integer,String> inputs = new LinkedHashMap<>();

    @Builder.Default
    @JsonSerialize(using = OpLogEvent.InputOutputSerializer.class)
    public Map<Integer,String> outputs = new LinkedHashMap<>();

    @Builder.Default
    @JsonSerialize(using = OpLogEvent.StackTraceSerializer.class)
    public String[] stackTrace = new String[0];

    public String firstNonExecutionCodeLine;


    public long eventId;

    public OpLogEventWrite(OpLogEvent opLogEvent) {
        this.opName = opLogEvent.getOpName();
        this.inputs = opLogEvent.getInputs();
        this.outputs = opLogEvent.getOutputs();
        this.stackTrace = opLogEvent.getStackTrace().split("\n");
        this.eventId = opLogEvent.getEventId();
        this.firstNonExecutionCodeLine = opLogEvent.getFirstNonExecutionCodeLine();
    }


}