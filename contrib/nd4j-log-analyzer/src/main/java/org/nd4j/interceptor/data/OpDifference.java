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


import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import org.nd4j.shade.jackson.databind.annotation.JsonSerialize;

import java.util.*;
import java.util.stream.Collectors;

@Getter
@Setter
@AllArgsConstructor
@Builder
public class OpDifference {
    @JsonSerialize(using = OpLogEventWriteSerializer.class)
    private OpLogEvent opLog1;
    @JsonSerialize(using = OpLogEventWriteSerializer.class)
    private OpLogEvent opLog2;
    private String differenceType;
    private int opDifference;
    private String differenceValue1;
    private String differenceValue2;
    public static List<String> skipOps = Arrays.asList(
            "set_scalar",
            "old_assign",
            "assign"
    );

    public long getEarliestEventTime() {
        return Math.min(opLog1.getEventId(), opLog2.getEventId());
    }

    public static OpDifference earliestDifference(Map<String,OpDifference> differenceList) {
        Map<String,OpDifference> opLog1 = new HashMap<>();
        for(Map.Entry<String,OpDifference> opDifference : differenceList.entrySet()) {
            if(skipOps.contains(opDifference.getValue().getOpLog1().opName) || opDifference.getValue().getOpLog1() == null
        || opDifference.getValue().getOpLog2() == null || opDifference.getValue().getOpLog2().opName == null || opDifference.getValue().getOpLog1().opName == null)
                continue;
            opLog1.put(opDifference.getKey(), opDifference.getValue());
        }



        List<OpDifference> opLog1List = new ArrayList<>(opLog1.values());
        //find the earliest event in oplog1
        Collections.sort(opLog1List, Comparator.comparingLong(OpDifference::getEarliestEventTime));

        return opLog1List.get(0);
    }

}