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

package org.nd4j.linalg.api.ops.impl.transforms.custom;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.List;
import java.util.Map;

public class Unique extends DynamicCustomOp {

    public Unique(){ }

    public Unique(SameDiff sd, SDVariable in){
        super(sd, new SDVariable[]{in}, false);
    }

    @Override
    public String opName(){
        return "unique";
    }

    @Override
    public String tensorflowName() {
        return "Unique";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        throw new UnsupportedOperationException("Not implemented yet");
    }

    @Override
    public int numOutputArguments(){
        return 2;
    }
}
