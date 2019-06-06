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

import lombok.Getter;
import org.bytedeco.javacpp.Pointer;

import java.lang.ref.ReferenceQueue;
import java.lang.ref.WeakReference;

/**
 * Weak reference for NativeRandom garbage collector
 *
 * @author raver119@gmail.com
 */
public class GarbageStateReference extends WeakReference<NativePack> {
    @Getter
    private Pointer statePointer;

    public GarbageStateReference(NativePack referent, ReferenceQueue<? super NativePack> queue) {
        super(referent, queue);
        this.statePointer = referent.getStatePointer();
        if (this.statePointer == null)
            throw new IllegalStateException("statePointer shouldn't be NULL");
    }
}
