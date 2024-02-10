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
package org.nd4j.linalg.profiler.data.array.watch;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.profiler.data.stacktrace.StackTraceQuery;
import org.nd4j.linalg.profiler.data.array.event.NDArrayEvent;
import org.nd4j.linalg.profiler.data.array.event.NDArrayEventType;

import java.io.Serializable;
import java.util.Arrays;

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class WatchCriteria implements Serializable  {

    private String className;
    private String methodName;
    private int lineNumber;
    private int lineNumberBegin;
    private int lineNumberEnd;
    private int occursWithinLineCount;
    private boolean exactMatch;
    private Enum targetWorkspaceType;
    private NDArrayEventType ndArrayEventType;

    /**
     * Returns true if the given event
     * fulfills the criteria
     * Criteria is based on
     * {@link StackTraceQuery}
     * and performs an or on other parameters fulfilled such as
     * {@link NDArrayEvent#getNdArrayEventType()}
     * and {@link NDArrayEvent#getParentWorkspace()}
     * @param event
     * @return
     */
    public boolean fulfillsCriteria(NDArrayEvent event) {
        StackTraceQuery stackTraceQuery = StackTraceQuery
                .builder()
                .className(className)
                .methodName(methodName)
                .lineNumber(lineNumber)
                .lineNumberBegin(lineNumberBegin)
                .lineNumberEnd(lineNumberEnd)
                .occursWithinLineCount(occursWithinLineCount)
                .exactMatch(exactMatch)
                .build();
        if(StackTraceQuery.stackTraceFillsAnyCriteria(Arrays.asList(stackTraceQuery),event.getStackTrace())) {
            return true;
        }


        if(targetWorkspaceType != null && event.getParentWorkspace() != null &&  event.getParentWorkspace().getAssociatedEnum() != null && event.getParentWorkspace().getAssociatedEnum().equals(targetWorkspaceType)) {
            return true;
        }


        if(targetWorkspaceType != null && event.getChildWorkspaceUseMetaData() != null &&  event.getChildWorkspaceUseMetaData().getAssociatedEnum() != null
                && event.getChildWorkspaceUseMetaData().getAssociatedEnum().equals(targetWorkspaceType)) {
            return true;
        }

        if(ndArrayEventType != null && event.getNdArrayEventType() != null && event.getNdArrayEventType().equals(ndArrayEventType)) {
            return true;
        }

        return false;

    }
}
