/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.nd4j.autodiff.samediff.transform;

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.ArrayList;
import java.util.List;

public class GraphTransformUtil {

    private GraphTransformUtil(){ }

    public List<SubGraph> getSubgraphsMatching(SameDiff sd, SubGraphPredicate p){

        List<SubGraph> out = new ArrayList<>();
        for(DifferentialFunction df : sd.functions()){
            if(p.matches(sd, df)){
                SubGraph sg = p.getSubGraph(sd, df);
                out.add(sg);
            }
        }

        return out;
    }

}
