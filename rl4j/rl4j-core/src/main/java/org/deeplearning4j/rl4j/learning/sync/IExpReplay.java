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

package org.deeplearning4j.rl4j.learning.sync;

import java.util.ArrayList;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/6/16.
 *
 * Common Interface for Experience replays
 *
 * A prioritized Exp Replay could be implemented by changing the interface
 * and integrating the TD-error in the transition for ranking
 * Not a high priority feature right now
 *
 * The memory is optimised by using array of INDArray in the transitions
 * such that two same INDArrays are not allocated twice
 */
public interface IExpReplay<A> {

    /**
     * @return a batch of uniformly sampled transitions
     */
    ArrayList<Transition<A>> getBatch();

    /**
     *
     * @param transition a new transition to store
     */
    void store(Transition<A> transition);

}
