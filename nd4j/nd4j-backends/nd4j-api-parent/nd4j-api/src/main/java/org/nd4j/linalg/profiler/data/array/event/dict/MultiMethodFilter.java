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
import org.nd4j.linalg.profiler.data.stacktrace.StackTraceQuery;

import java.util.List;

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class MultiMethodFilter {
    private List<StackTraceQuery> pointOfOriginFilters;
    private List<StackTraceQuery> pointOfInvocationFilters;
    private List<StackTraceQuery> parentPointOfInvocationFilters;
    private boolean onlyIncludeDifferences;
    private boolean inclusionFilter;

    /**
     * Returns true if the filter is empty
     * "Empty" is defined as having no filters for point of origin, point of invocation, or parent point of invocation
     * or being null
     * @param filter the filter to check
     * @return
     */
    public static boolean isEmpty(MultiMethodFilter filter) {
        return filter == null || (filter.getPointOfOriginFilters() == null && filter.getPointOfInvocationFilters() == null && filter.getParentPointOfInvocationFilters() == null);
    }

}
