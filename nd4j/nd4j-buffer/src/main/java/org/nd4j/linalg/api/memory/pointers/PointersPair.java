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

package org.nd4j.linalg.api.memory.pointers;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * @author raver119@gmail.com
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class PointersPair {
    private Long allocationCycle;
    private Long requiredMemory;
    private PagedPointer hostPointer;
    private PagedPointer devicePointer;

    public PointersPair(PagedPointer hostPointer, PagedPointer devicePointer) {
        if (hostPointer == null && devicePointer == null)
            throw new RuntimeException("Both pointers can't be null");

        this.hostPointer = hostPointer;
        this.devicePointer = devicePointer;
    }
}
