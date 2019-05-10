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

package org.deeplearning4j.rl4j.learning.sync.qlearning;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

public class QLConfigurationTest {
    @Rule
    public ExpectedException thrown = ExpectedException.none();

    @Test
    public void serialize() throws Exception {
        ObjectMapper mapper = new ObjectMapper();
        QLearning.QLConfiguration qlConfiguration =
                new QLearning.QLConfiguration(
                        123,    //Random seed
                        200,    //Max step By epoch
                        8000, //Max step
                        150000, //Max size of experience replay
                        32,     //size of batches
                        500,    //target update (hard)
                        10,     //num step noop warmup
                        0.01,   //reward scaling
                        0.99,   //gamma
                        1.0,    //td error clipping
                        0.1f,   //min epsilon
                        10000,   //num step for eps greedy anneal
                        true    //double DQN
                );

        // Should not throw..
        String json = mapper.writeValueAsString(qlConfiguration);
        QLearning.QLConfiguration cnf = mapper.readValue(json, QLearning.QLConfiguration.class);
    }
}