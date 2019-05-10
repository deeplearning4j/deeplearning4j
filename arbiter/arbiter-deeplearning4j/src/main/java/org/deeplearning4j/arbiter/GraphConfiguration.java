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
