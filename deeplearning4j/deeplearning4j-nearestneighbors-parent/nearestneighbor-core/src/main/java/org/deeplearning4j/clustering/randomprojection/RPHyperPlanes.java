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

package org.deeplearning4j.clustering.randomprojection;

import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

@Data
public class RPHyperPlanes {
    private int dim;
    private INDArray wholeHyperPlane;

    public RPHyperPlanes(int dim) {
        this.dim = dim;
    }

    public INDArray getHyperPlaneAt(int depth) {
        if(wholeHyperPlane.isVector())
            return wholeHyperPlane;
        return wholeHyperPlane.slice(depth);
    }


    /**
     * Add a new random element to the hyper plane.
     */
    public void addRandomHyperPlane() {
        INDArray newPlane = Nd4j.randn(new int[] {1,dim});
        newPlane.divi(newPlane.normmaxNumber());
        if(wholeHyperPlane == null)
            wholeHyperPlane = newPlane;
        else {
            wholeHyperPlane = Nd4j.concat(0,wholeHyperPlane,newPlane);
        }
    }


}
