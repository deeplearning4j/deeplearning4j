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
package org.nd4j.linalg.profiler.data.stacktrace;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;
import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class StackTraceQueryFilters implements Serializable {

    private List<StackTraceQuery> include;
    private List<StackTraceQuery> exclude;

    /**
     * Returns true if the stack trace element should be filtered
     * @param stackTraceElement the stack trace element to filter
     * @return true if the stack trace element should be filtered, false otherwise
     */
    public boolean filter(StackTraceElement stackTraceElement) {
        if (exclude != null && !exclude.isEmpty()) {
            for (StackTraceQuery query : exclude) {
                if (query.filter(stackTraceElement)) {
                    return true;
                }
            }
        }

        if (include != null && !include.isEmpty()) {
            for (StackTraceQuery query : include) {
                if (query.filter(stackTraceElement)) {
                    return false;
                }
            }
            return false;
        }
        return false;
    }

    /**
     * Returns true if the stack trace element should be filtered
     * @param stackTraceElement the stack trace element to filter
     * @param stackTraceQueryFilters the filters to apply
     * @return true if the stack trace element should be filtered, false otherwise
     */
    public static boolean shouldFilter(StackTraceElement stackTraceElement[],
                                       StackTraceQueryFilters stackTraceQueryFilters) {
        if(stackTraceQueryFilters == null || stackTraceElement == null) {
            return false;
        }

        for(StackTraceElement stackTraceElement1 : stackTraceElement) {
            if(stackTraceElement1 == null)
                continue;
            if (stackTraceQueryFilters.filter(stackTraceElement1)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Returns true if the stack trace element should be filtered
     * @param stackTraceElement the stack trace element to filter
     * @param stackTraceQueryFilters the filters to apply
     * @return
     */
    public static boolean shouldFilter(StackTraceElement stackTraceElement,
                                       StackTraceQueryFilters stackTraceQueryFilters) {
        if(stackTraceQueryFilters == null || stackTraceElement == null)
            return false;
        return stackTraceQueryFilters.filter(stackTraceElement);
    }


}
