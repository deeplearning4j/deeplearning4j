/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.systeminfo;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class GPUInfo {

    public static final String fGpu = "  %-30s %-5s %24s %24s %24s";

    private String name;
    private long totalMemory;
    private long freeMemory;
    int major;
    int minor;

    @Override
    public String toString(){
        return String.format(fGpu, name, major + "." + minor, SystemInfo.fBytes(totalMemory),
                SystemInfo.fBytes(totalMemory - freeMemory), SystemInfo.fBytes(freeMemory));
    }
}
