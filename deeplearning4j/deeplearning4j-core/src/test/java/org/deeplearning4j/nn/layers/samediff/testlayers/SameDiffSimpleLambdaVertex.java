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

package org.deeplearning4j.nn.layers.samediff.testlayers;

import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLambdaVertex;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;

public class SameDiffSimpleLambdaVertex extends SameDiffLambdaVertex {

    @Override
    public SDVariable defineVertex(SameDiff sameDiff, VertexInputs inputs) {
        SDVariable in1 = inputs.getInput(0);
        SDVariable in2 = inputs.getInput(1);
        SDVariable ret = in1.mul(in2);
        return ret;
    }
}
