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


@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class StackTraceLineSkip {

    private String className;
    private String methodName;
    private String packageName;
    @Builder.Default
    private  int lineNumber = -1;



    public static boolean stackTraceSkip(String stackTraceElement,String skip) {
        return stackTraceElement.contains(skip);
    }

    public static boolean matchesLineSkip(StackTraceElement stackTraceElement, StackTraceLineSkip stackTraceLineSkip) {
        if(stackTraceLineSkip.getClassName() != null && !stackTraceSkip(stackTraceElement.getClassName(),stackTraceLineSkip.getClassName())) {
            return false;
        }
        if(stackTraceLineSkip.getMethodName() != null && !stackTraceSkip(stackTraceElement.getMethodName(),stackTraceLineSkip.getMethodName())) {
            return false;
        }
        if(stackTraceLineSkip.getLineNumber() != -1 && stackTraceElement.getLineNumber() != stackTraceLineSkip.getLineNumber()) {
            return false;
        }

        //get the package name from a fully qualified java class name
        String packageName = stackTraceElement.getClassName().substring(0,stackTraceElement.getClassName().lastIndexOf("."));

        if(stackTraceLineSkip.getPackageName() != null && !stackTraceSkip(packageName,stackTraceLineSkip.getPackageName())) {
            return false;
        }

        return true;
    }

}
