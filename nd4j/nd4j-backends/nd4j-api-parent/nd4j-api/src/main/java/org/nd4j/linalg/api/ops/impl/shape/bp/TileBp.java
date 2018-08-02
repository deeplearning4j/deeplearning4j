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

package org.nd4j.linalg.api.ops.impl.shape.bp;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.*;

/**
 * Tile backprop function
 *
 * @author Alex Black
 */
public class TileBp extends DynamicCustomOp {

    private int[] repeat;

    public TileBp(SameDiff sameDiff, SDVariable in, SDVariable grad, int[] repeat) {
        super(null,sameDiff, new SDVariable[]{in, grad}, false);
        this.repeat = repeat;
        addArguments();
    }

    public TileBp() {}

    private void addArguments() {
        addIArgument(repeat);
    }


    @Override
    public String opName() {
        return "tile_bp";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        throw new UnsupportedOperationException("Backprop of gradient op not supported");
    }

}
