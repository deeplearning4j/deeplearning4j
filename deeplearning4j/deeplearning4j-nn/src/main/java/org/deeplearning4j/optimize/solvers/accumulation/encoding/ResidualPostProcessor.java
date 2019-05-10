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

package org.deeplearning4j.optimize.solvers.accumulation.encoding;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

/**
 * ResidualPostProcessor: is (as the name suggests) is used to post process the residual vector for DL4J's gradient
 * sharing implementation. The motivation for post processing the residual vector is to avoid it getting too large:
 * a large residual can take many steps to communicate, which may lead to stale gradient issues.
 * Thus most ResidualPostProcessor implementations will simply decay or clip the residual vector to keep values from
 * getting too large relative to the current threshold.
 *
 * @author Alex Black
 */
public interface ResidualPostProcessor extends Serializable, Cloneable {

    /**
     * @param iteration      Current iteration
     * @param epoch          Current epoch
     * @param lastThreshold  Last threshold that was used
     * @param residualVector The current residual vector. Should be modified in-place
     */
    void processResidual(int iteration, int epoch, double lastThreshold, INDArray residualVector);

    ResidualPostProcessor clone();
}
