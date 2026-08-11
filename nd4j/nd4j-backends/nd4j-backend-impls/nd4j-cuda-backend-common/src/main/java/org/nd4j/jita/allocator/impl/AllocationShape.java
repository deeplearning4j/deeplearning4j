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

package org.nd4j.jita.allocator.impl;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.api.buffer.DataType;

/**
 * @author raver119@gmail.com
 */
@Data
@NoArgsConstructor
public class AllocationShape {
    private long length = 0;
    private byte elementSize = 0;
    private DataType dataType = DataType.FLOAT;

    /*
    public AllocationShape(long length, int elementSize) {
        this.length = length;
        this.elementSize = elementSize;
    }
    */
    public AllocationShape(long length, int elementSize, DataType dataType) {
        this.length = length;
        this.elementSize = (byte) elementSize;
        this.dataType = dataType;
    }

    public int getElementSize() {
        return elementSize;
    }

    public void setElementSize(int elementSize) {
        this.elementSize = (byte) elementSize;
    }


    public long getNumberOfBytes() {
        return this.length * this.elementSize;
    }
}
