/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.heartbeat.reports;

import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;

/**
 * Bean/POJO that describes current jvm/node
 *
 * @author raver119@gmail.com
 */
@Data
@NoArgsConstructor
public class Environment implements Serializable {
    private long serialVersionID;

    /*
        number of cores available to jvm
     */
    private int numCores;

    /*
        memory available within current process
     */
    private long availableMemory;

    /*
        System.getPriority("java.specification.version");
     */
    private String javaVersion;


    /*
        System.getProperty("os.opName");
     */
    private String osName;

    /*
        System.getProperty("os.arch");
        however, in 97% of cases it will be amd64. it will be better to have JNI call for that in future
     */
    private String osArch;

    /*
        Nd4j backend being used within current JVM session
    */
    private String backendUsed;


    public String toCompactString() {
        StringBuilder builder = new StringBuilder();

        /*
         new format is:
         Backend ( cores, ram, jvm, Linux, arch)
         */
        /*
            builder.append(numCores).append("cores/");
            builder.append(availableMemory / 1024 / 1024 / 1024).append("GB/");
            builder.append("jvm").append(javaVersion).append("/");
            builder.append(osName).append("/");
            builder.append(osArch).append("/");
            builder.append(backendUsed).append(" ");
        */

        builder.append(backendUsed).append(" (").append(numCores).append(" cores ")
                        .append(Math.max(availableMemory / 1024 / 1024 / 1024, 1)).append("GB ").append(osName)
                        .append(" ").append(osArch).append(")");

        return builder.toString();
    }
}
