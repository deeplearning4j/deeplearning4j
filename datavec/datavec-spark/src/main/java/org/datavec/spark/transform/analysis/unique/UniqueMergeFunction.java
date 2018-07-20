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

package org.datavec.spark.transform.analysis.unique;

import org.apache.spark.api.java.function.Function2;
import org.datavec.api.writable.Writable;

import java.util.Map;
import java.util.Set;

/**
 * Simple function used in AnalyzeSpark.getUnique
 *
 * @author Alex Black
 */
public class UniqueMergeFunction implements Function2<Map<String, Set<Writable>>, Map<String, Set<Writable>>, Map<String, Set<Writable>>> {
    @Override
    public Map<String, Set<Writable>> call(Map<String, Set<Writable>> v1, Map<String, Set<Writable>> v2) throws Exception {
        if(v1 == null){
            return v2;
        }
        if(v2 == null){
            return v1;
        }

        for(String s : v1.keySet()){
            v1.get(s).addAll(v2.get(s));
        }
        return v1;
    }
}
