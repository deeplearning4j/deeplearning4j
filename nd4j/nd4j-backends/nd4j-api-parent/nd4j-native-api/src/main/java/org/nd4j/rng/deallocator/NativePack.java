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

package org.nd4j.rng.deallocator;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.bytedeco.javacpp.Pointer;

/**
 * Simple wrapper for state pointer, to avoid enqueue of non-initialized objects
 *
 * @author raver119@gmail.com
 */
@Data
@AllArgsConstructor
public class NativePack {
    private Long address;
    private Pointer statePointer;
}
