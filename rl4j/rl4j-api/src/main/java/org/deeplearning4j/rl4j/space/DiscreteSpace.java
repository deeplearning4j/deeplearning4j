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

package org.deeplearning4j.rl4j.space;

import lombok.Getter;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;

public class DiscreteSpace implements ActionSpace<Integer> {

    //size of the space also defined as the number of different actions
    @Getter
    final protected int size;
    protected final Random rnd;

    public DiscreteSpace(int size) {
        this(size, Nd4j.getRandom());
    }

    public DiscreteSpace(int size, Random rnd) {
        this.size = size;
        this.rnd = rnd;
    }

    public Integer randomAction() {
        return rnd.nextInt(size);
    }

    public Object encode(Integer a) {
        return a;
    }

    public Integer noOp() {
        return 0;
    }

}
