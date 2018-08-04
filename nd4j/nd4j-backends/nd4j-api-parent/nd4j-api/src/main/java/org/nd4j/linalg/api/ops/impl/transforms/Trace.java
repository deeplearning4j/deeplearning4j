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

package org.nd4j.linalg.api.ops.impl.transforms;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

/**
 * Matrix trace operation
 *
 * @author Alex Black
 */
public class Trace extends DynamicCustomOp {

    public Trace(SameDiff sd, SDVariable in){
        super(null, sd, new SDVariable[]{in});
    }

    public Trace(){ }

    @Override
    public String opName(){
        return "trace";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradAtOutput){
//        SDVariable eye = sameDiff.eye(f().shape(arg()), f().siz)
        throw new UnsupportedOperationException();
    }

}
