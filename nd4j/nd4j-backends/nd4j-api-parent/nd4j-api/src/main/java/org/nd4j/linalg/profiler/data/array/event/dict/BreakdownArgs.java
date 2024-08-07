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
package org.nd4j.linalg.profiler.data.array.event.dict;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.profiler.data.array.event.NDArrayEvent;
import org.nd4j.linalg.profiler.data.stacktrace.StackTraceLookupKey;
import org.nd4j.linalg.profiler.data.stacktrace.StackTraceQuery;

import java.util.ArrayList;
import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class BreakdownArgs {
    private StackTraceLookupKey pointOfOrigin;
    private StackTraceLookupKey compPointOfOrigin;
    private StackTraceLookupKey commonPointOfInvocation;
    private StackTraceLookupKey commonParentOfInvocation;
    @Builder.Default
    private List<StackTraceQuery> eventsToExclude = new ArrayList<>();
    @Builder.Default
    private List<StackTraceQuery> eventsToInclude = new ArrayList<>();

    /**
     * Filter the events if needed
     * based on the {@link #eventsToExclude}
     * and {@link #eventsToInclude}
     *  using {@link StackTraceQuery#stackTraceFillsAnyCriteria(List, StackTraceElement[])}
     * @param toFilter the events to filter
     * @return
     */
    public List<NDArrayEvent> filterIfNeeded(List<NDArrayEvent> toFilter) {
        List<NDArrayEvent> ret = new ArrayList<>();
        for(NDArrayEvent event : toFilter) {
            if(eventsToExclude.isEmpty() && eventsToInclude.isEmpty()) {
                ret.add(event);
            } else if(eventsToExclude.isEmpty() && !eventsToInclude.isEmpty()) {
                if(StackTraceQuery.stackTraceFillsAnyCriteria(eventsToExclude, event.getStackTrace())) {
                    ret.add(event);
                }
            } else if(!eventsToExclude.isEmpty() && eventsToInclude.isEmpty()) {
                if(!StackTraceQuery.stackTraceFillsAnyCriteria(eventsToExclude, event.getStackTrace())) {
                    ret.add(event);
                }
            } else {
                if(StackTraceQuery.stackTraceFillsAnyCriteria(eventsToInclude, event.getStackTrace())
                        && !StackTraceQuery.stackTraceFillsAnyCriteria(eventsToExclude, event.getStackTrace())) {
                    ret.add(event);
                }
            }
        }
        return ret;
    }

}
