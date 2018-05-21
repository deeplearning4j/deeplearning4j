/*-
 *  * Copyright 2016 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

package org.deeplearning4j.arbiter;


import lombok.AllArgsConstructor;
import lombok.Data;
import org.deeplearning4j.arbiter.optimize.serde.jackson.JsonMapper;
import org.deeplearning4j.arbiter.optimize.serde.jackson.YamlMapper;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.shade.jackson.core.JsonProcessingException;

import java.io.Serializable;

/**
 * Analogous to {@link DL4JConfiguration}, GraphConfiguration includes a configuration for ComputationGraphs, as well
 * as early stopping (or, optionally numEpochs) fields.
 */
@AllArgsConstructor
@Data
public class GraphConfiguration implements Serializable {
    private ComputationGraphConfiguration configuration;
    private EarlyStoppingConfiguration<ComputationGraph> earlyStoppingConfiguration;
    private Integer numEpochs;



    /**
     * Yaml mapping
     * @return
     */
    public String toYaml() {
        try {
            return YamlMapper.getMapper().writeValueAsString(this);
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Json mapping
     * @return
     */
    public String toJson() {
        try {
            return JsonMapper.getMapper().writeValueAsString(this);
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }
}
