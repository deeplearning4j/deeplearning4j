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
package org.eclipse.deeplearning4j.nd4j.autodiff.optimization.util;

import lombok.Data;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.optimize.Optimizer;
import org.nd4j.autodiff.samediff.optimize.OptimizerSet;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Data
public class OptTestConfig {

    private SameDiff original;
    private Map<String, INDArray> placeholders;
    private List<String> outputs;
    private File tempFolder;
    private Map<String,Class<? extends Optimizer>> mustApply;
    private List<OptimizerSet> optimizerSets;

    public static Builder builder(){
        return new Builder();
    }

    public static class Builder {

        private SameDiff original;
        private Map<String, INDArray> placeholders;
        private List<String> outputs;
        private File tempFolder;
        private Map<String,Class<? extends Optimizer>> mustApply;
        private List<OptimizerSet> optimizerSets;

        public Builder tempFolder(File tempFolder) {
            this.tempFolder = tempFolder;
            return this;
        }

        public Builder original(SameDiff sd){
            original = sd;
            return this;
        }

        public Builder placeholder(String ph, INDArray arr){
            if(placeholders == null)
                placeholders = new HashMap<>();
            placeholders.put(ph, arr);
            return this;
        }

        public Builder placeholders(Map<String,INDArray> map){
            placeholders = map;
            return this;
        }

        public Builder outputs(String... outputs){
            this.outputs = Arrays.asList(outputs);
            return this;
        }

        public Builder outputs(List<String> outputs){
            this.outputs = outputs;
            return this;
        }

        public Builder mustApply(String opName, Class<? extends Optimizer> optimizerClass){
            if(mustApply == null)
                mustApply = new HashMap<>();
            mustApply.put(opName, optimizerClass);
            return this;
        }

        public Builder optimizerSets(List<OptimizerSet> list){
            this.optimizerSets = list;
            return this;
        }

        public OptTestConfig build(){
            OptTestConfig c = new OptTestConfig();
            c.original = original;
            c.placeholders = placeholders;
            c.outputs = outputs;
            c.tempFolder = tempFolder;
            c.mustApply = mustApply;
            c.optimizerSets = optimizerSets;
            return c;
        }

    }

}