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

package org.nd4j.linalg.api.memory;

import org.nd4j.linalg.profiler.data.eventlogger.LogEvent;

public interface Deallocator {
    /**
     * This method does actual deallocation
     */
    void deallocate();

    /**
     * Log event for a deallocation.
     * Only used when {@link org.nd4j.linalg.profiler.data.eventlogger.EventLogger#enabled}
     * is true. We store events on deallocators to retain metadata
     * about a be to be deleted buffer without the need to retain a reference
     * to the deallocatable object. This is to avoid conflict with {@link java.lang.ref.WeakReference}
     *
     * @return
     */
    LogEvent logEvent();

    /**
     * Returns whether the deallocator
     * is constant or not.
     *
     * @return
     */
    boolean isConstant();

    /**
     * Sets whether this deallocator is constant or not.
     * This is needed for when something like a databuffer changes its state.
     * @param constant
     */
    default void setConstant(boolean constant) {
        //default is no op. Only databuffer deallocators really need to update
        //their state as constant or not.
    }
}
