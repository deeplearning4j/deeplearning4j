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

package org.nd4j.linalg.api.ops.impl.shape;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.ArrayList;
import java.util.List;

public class MeshGrid extends DynamicCustomOp {

    /**
     *
     * @param sd
     * @param cartesian If true: broadcast dimensions for first two dimensions are swapped
     * @param inputs
     */
    public MeshGrid(SameDiff sd, boolean cartesian, SDVariable... inputs){
        super(null, sd, inputs, false);
        addIArgument(cartesian ? 1 : 0);
    }

    public MeshGrid(){ }

    @Override
    public String opName(){
        return "meshgrid";
    }

    public List<SDVariable> doDiff(List<SDVariable> gradients){
        SDVariable[] args = args();
        List<SDVariable> out = new ArrayList<>(args.length);
        for( int i=0; i<args.length; i++ ){
            int[] dims = new int[args.length-1];
            int x=0;
            for( int j=0; j<args.length; j++){
                if(i == j)
                    continue;
                dims[x++] = j;
            }
            out.add(gradients.get(0).sum(dims));
        }
        return out;
    }

    @Override
    public int getNumOutputs(){
        return args().length;
    }

}
