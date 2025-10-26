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

package org.nd4j.autodiff.samediff;

import org.nd4j.linalg.api.buffer.DataType;

// Inner classes for holding information
public  class VariableInfo {
    public final String name;
    public final VariableType type;
    public final DataType dataType;
    public String frame;
    public int iteration;
    public String parentFrame;

    public VariableInfo(String name, VariableType type, DataType dataType) {
        this.name = name;
        this.type = type;
        this.dataType = dataType;
    }
}
