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

package org.nd4j.jita.balance;

import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AllocationShape;
import org.nd4j.jita.conf.Configuration;

/**
 * @author raver119@gmail.com
 */
@Deprecated
public interface Balancer {

    /**
     *
     * This method initializes this Balancer instance
     *
     * @param configuration
     */
    void init(Configuration configuration);

    /**
     * This method checks, if it's worth moving some memory region to device
     *
     * @param deviceId
     * @param point
     * @param shape
     * @return
     */
    AllocationStatus makePromoteDecision(Integer deviceId, AllocationPoint point, AllocationShape shape);

    /**
     * This method checks, if it's worth moving some memory region to host
     *
     * @param deviceId
     * @param point
     * @param shape
     * @return
     */
    AllocationStatus makeDemoteDecision(Integer deviceId, AllocationPoint point, AllocationShape shape);
}
