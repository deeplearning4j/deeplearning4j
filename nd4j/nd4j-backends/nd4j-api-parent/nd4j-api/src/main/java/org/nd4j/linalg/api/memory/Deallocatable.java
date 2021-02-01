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

/**
 * This interface describes resource trackable via unified deallocation system
 *
 * @author raver119@gmail.com
 */
public interface Deallocatable {
    /**
     * This method returns unique ID for this instance
     * @return
     */
    String getUniqueId();

    /**
     * This method returns deallocator associated with this instance
     * @return
     */
    Deallocator deallocator();


    /**
     * This method returns deviceId it's affined with, so deallocator thread will be guaranteed to match it
     */
    int targetDevice();
}
