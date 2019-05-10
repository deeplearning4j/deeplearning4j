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

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Pointer;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class ImmortalFloatPointer extends FloatPointer {
    private Pointer pointer;

    public ImmortalFloatPointer(PagedPointer pointer) {
        this.pointer = pointer;

        this.address = pointer.address();
        this.capacity = pointer.capacity();
        this.limit = pointer.limit();
        this.position = 0;
    }
}
