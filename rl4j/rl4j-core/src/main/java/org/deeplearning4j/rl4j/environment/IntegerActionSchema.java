/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.deeplearning4j.rl4j.environment;

import lombok.Getter;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;

// Work in progress
public class IntegerActionSchema implements IActionSchema<Integer> {

    @Getter
    private final int actionSpaceSize;

    private final int noOpAction;
    private final Random rnd;

    public IntegerActionSchema(int numActions, int noOpAction) {
        this(numActions, noOpAction, Nd4j.getRandom());
    }

    public IntegerActionSchema(int numActions, int noOpAction, Random rnd) {
        this.actionSpaceSize = numActions;
        this.noOpAction = noOpAction;
        this.rnd = rnd;
    }

    @Override
    public Integer getNoOp() {
        return noOpAction;
    }

    @Override
    public Integer getRandomAction() {
        return rnd.nextInt(actionSpaceSize);
    }
}
