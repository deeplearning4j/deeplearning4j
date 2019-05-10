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

package org.deeplearning4j.aws.emr;

import com.amazonaws.services.elasticmapreduce.model.Configuration;

import lombok.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;


@Data
@AllArgsConstructor(access = AccessLevel.PRIVATE)
@NoArgsConstructor
@Builder
public class EmrConfig {

    protected String classification;
    protected Map<String, String> properties;
    protected List<EmrConfig> configs;

    Configuration toAwsConfig() {
        Configuration config = new Configuration().withClassification(classification).withProperties(properties);
        List<Configuration> subConfigs = new ArrayList<>();
        for (EmrConfig conf : configs){
            subConfigs.add(conf.toAwsConfig());
        }
        return config.withConfigurations(subConfigs);
    }

}
