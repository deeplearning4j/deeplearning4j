/*
 *  ******************************************************************************
 *  *
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

package org.deeplearning4j.rl4j.support;

import org.deeplearning4j.rl4j.space.ObservationSpace;
import org.nd4j.linalg.api.ndarray.INDArray;

public class MockObservationSpace implements ObservationSpace {

    private final int[] shape;

    public MockObservationSpace() {
        this(new int[] { 1 });
    }

    public MockObservationSpace(int[] shape) {
        this.shape = shape;
    }

    @Override
    public String getName() {
        return null;
    }

    @Override
    public int[] getShape() {
        return shape;
    }

    @Override
    public INDArray getLow() {
        return null;
    }

    @Override
    public INDArray getHigh() {
        return null;
    }
}