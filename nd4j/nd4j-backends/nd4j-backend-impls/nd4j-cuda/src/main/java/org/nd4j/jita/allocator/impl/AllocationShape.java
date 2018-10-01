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

package org.nd4j.jita.allocator.impl;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.api.buffer.DataBuffer;

/**
 * @author raver119@gmail.com
 */
@Data
@NoArgsConstructor
public class AllocationShape {
    private long offset = 0;
    private long length = 0;
    private int stride = 1;
    private int elementSize = 0;
    private DataType dataType = DataType.FLOAT;

    /*
    public AllocationShape(long length, int elementSize) {
        this.length = length;
        this.elementSize = elementSize;
    }
    */
    public AllocationShape(long length, int elementSize, DataType dataType) {
        this.length = length;
        this.elementSize = elementSize;
        this.dataType = dataType;
    }


    public long getNumberOfBytes() {
        return this.length * this.elementSize;
    }
}
