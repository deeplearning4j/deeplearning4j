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

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class StackTraceLookupKey implements Serializable  {

    private String className;
    private String methodName;
    private int lineNumber;


    public static StackTraceElement stackTraceElementOf(StackTraceLookupKey key) {
        return StackTraceElementCache.lookup(key);
    }

    public static StackTraceLookupKey of(String className, String methodName, int lineNumber) {
        return StackTraceLookupKey.builder()
                .className(className)
                .methodName(methodName)
                .lineNumber(lineNumber)
                .build();
    }
}
