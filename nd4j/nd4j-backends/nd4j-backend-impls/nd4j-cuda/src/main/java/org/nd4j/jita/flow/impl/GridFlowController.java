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

package org.nd4j.jita.flow.impl;

import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * FlowController implementation suitable for CudaGridExecutioner
 *
 * Main difference here, is delayed execution support and forced execution trigger in special cases
 *
 * @author raver119@gmail.com
 */
public class GridFlowController extends SynchronousFlowController {

    private static Logger logger = LoggerFactory.getLogger(GridFlowController.class);

    /**
     * This method makes sure HOST memory contains latest data from GPU
     *
     * Additionally, this method checks, that there's no ops pending execution for this array
     *
     * @param point
     */
    @Override
    public void synchronizeToHost(AllocationPoint point) {
        if (!point.isConstant() && point.isEnqueued()) {
            waitTillFinished(point);
        }

        super.synchronizeToHost(point);
    }

    /**
     *
     * Additionally, this method checks, that there's no ops pending execution for this array
     * @param point
     */
    @Override
    public void waitTillFinished(AllocationPoint point) {
        if (!point.isConstant() && point.isEnqueued())
            Nd4j.getExecutioner().commit();

        super.waitTillFinished(point);
    }

    /**
     *
     * Additionally, this method checks, that there's no ops pending execution for this array
     *
     * @param point
     */
    @Override
    public void waitTillReleased(AllocationPoint point) {
        /**
         * We don't really need special hook here, because if op is enqueued - it's still holding all arrays
         */

        super.waitTillReleased(point);
    }
}
