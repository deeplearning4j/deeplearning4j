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

package org.nd4j.nativeblas;

import org.bytedeco.javacpp.Pointer;

/**
 *
 * @author saudet
 */
public class OpaqueContext extends Pointer {
    private NativeBufferOwner backendOwner;
    private volatile boolean closed;

    public OpaqueContext(Pointer p) {
        super(p);
    }

    /** Creates a graph context through the selected backend authority. */
    public static OpaqueContext create(NativeBufferOwner owner, int nodeId) {
        if (owner == null) {
            throw new IllegalArgumentException("NativeBufferOwner cannot be null");
        }
        OpaqueContext context = owner.nativeOps().createGraphContext(nodeId);
        if (context == null || context.isNull()) {
            throw new IllegalStateException("Backend failed to create graph context for node " + nodeId);
        }
        return context.attachOwner(owner);
    }

    /** Attaches the backend that created this native context. */
    public OpaqueContext attachOwner(NativeBufferOwner owner) {
        if (owner == null) {
            throw new IllegalArgumentException("NativeBufferOwner cannot be null");
        }
        if (closed) {
            throw new IllegalStateException("Cannot attach an owner to a closed graph context");
        }
        if (backendOwner != null && backendOwner.nativeOps() != owner.nativeOps()) {
            throw new IllegalStateException("OpaqueContext already belongs to a different backend");
        }
        backendOwner = owner;
        return this;
    }

    public NativeBufferOwner backendOwner() {
        if (backendOwner == null) {
            throw new IllegalStateException(
                    "OpaqueContext has no backend owner; create it with OpaqueContext.create(owner, nodeId)");
        }
        return backendOwner;
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        synchronized (this) {
            if (closed) {
                return;
            }
            if (!isNull()) {
                backendOwner().nativeOps().deleteGraphContext(this);
                setNull();
            }
            closed = true;
        }
    }
}
