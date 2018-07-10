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

package org.deeplearning4j.arbiter.optimize;

import org.apache.commons.math3.distribution.LogNormalDistribution;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.distribution.UniformIntegerDistribution;
import org.deeplearning4j.arbiter.optimize.api.CandidateGenerator;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.api.data.DataSetIteratorFactoryProvider;
import org.deeplearning4j.arbiter.optimize.generator.GridSearchCandidateGenerator;
import org.deeplearning4j.arbiter.optimize.generator.RandomSearchGenerator;
import org.deeplearning4j.arbiter.optimize.parameter.FixedValue;
import org.deeplearning4j.arbiter.optimize.parameter.continuous.ContinuousParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.discrete.DiscreteParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.integer.IntegerParameterSpace;
import org.junit.Test;
import org.nd4j.shade.jackson.annotation.JsonAutoDetect;
import org.nd4j.shade.jackson.annotation.PropertyAccessor;
import org.nd4j.shade.jackson.core.JsonFactory;
import org.nd4j.shade.jackson.databind.DeserializationFeature;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.SerializationFeature;
import org.nd4j.shade.jackson.dataformat.yaml.YAMLFactory;
import org.nd4j.shade.jackson.datatype.joda.JodaModule;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 02/02/2017.
 */
public class TestJson {

    protected static ObjectMapper getObjectMapper(JsonFactory factory) {
        ObjectMapper om = new ObjectMapper(factory);
        om.registerModule(new JodaModule());
        om.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        om.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
        om.enable(SerializationFeature.INDENT_OUTPUT);
        om.setVisibility(PropertyAccessor.ALL, JsonAutoDetect.Visibility.NONE);
        om.setVisibility(PropertyAccessor.FIELD, JsonAutoDetect.Visibility.ANY);
        return om;
    }

    private static ObjectMapper jsonMapper = getObjectMapper(new JsonFactory());
    private static ObjectMapper yamlMapper = getObjectMapper(new YAMLFactory());


    @Test
    public void testParameterSpaceJson() throws Exception {

        List<ParameterSpace<?>> l = new ArrayList<>();
        l.add(new FixedValue<>(1.0));
        l.add(new FixedValue<>(1));
        l.add(new FixedValue<>("string"));
        l.add(new ContinuousParameterSpace(-1, 1));
        l.add(new ContinuousParameterSpace(new LogNormalDistribution(1, 1)));
        l.add(new ContinuousParameterSpace(new NormalDistribution(2, 0.01)));
        l.add(new DiscreteParameterSpace<>(1, 5, 7));
        l.add(new DiscreteParameterSpace<>("first", "second", "third"));
        l.add(new IntegerParameterSpace(0, 10));
        l.add(new IntegerParameterSpace(new UniformIntegerDistribution(0, 50)));

        for (ParameterSpace<?> ps : l) {
            String strJson = jsonMapper.writeValueAsString(ps);
            String strYaml = yamlMapper.writeValueAsString(ps);

            ParameterSpace<?> fromJson = jsonMapper.readValue(strJson, ParameterSpace.class);
            ParameterSpace<?> fromYaml = yamlMapper.readValue(strYaml, ParameterSpace.class);

            assertEquals(ps, fromJson);
            assertEquals(ps, fromYaml);
        }
    }

    @Test
    public void testCandidateGeneratorJson() throws Exception {
        Map<String, Object> commands = new HashMap<>();
        commands.put(DataSetIteratorFactoryProvider.FACTORY_KEY, new HashMap<>());

        List<CandidateGenerator> l = new ArrayList<>();
        l.add(new GridSearchCandidateGenerator(new DiscreteParameterSpace<>(0, 1, 2, 3, 4, 5), 10,
                        GridSearchCandidateGenerator.Mode.Sequential, commands));
        l.add(new GridSearchCandidateGenerator(new DiscreteParameterSpace<>(0, 1, 2, 3, 4, 5), 10,
                        GridSearchCandidateGenerator.Mode.RandomOrder, commands));
        l.add(new RandomSearchGenerator(new DiscreteParameterSpace<>(0, 1, 2, 3, 4, 5), commands));

        for (CandidateGenerator cg : l) {
            String strJson = jsonMapper.writeValueAsString(cg);
            String strYaml = yamlMapper.writeValueAsString(cg);

            CandidateGenerator fromJson = jsonMapper.readValue(strJson, CandidateGenerator.class);
            CandidateGenerator fromYaml = yamlMapper.readValue(strYaml, CandidateGenerator.class);

            assertEquals(cg, fromJson);
            assertEquals(cg, fromYaml);
        }
    }
}
