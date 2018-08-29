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

package org.nd4j.graph;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.Test;
import org.nd4j.autodiff.execution.conf.ExecutorConfiguration;
import org.nd4j.autodiff.execution.conf.OutputMode;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.io.ClassPathResource;

import static org.junit.Assert.*;

@Slf4j
public class GraphInferenceGrpcClientTest {
    @Test
    public void testSimpleGraph_1() throws Exception {
        val client = new GraphInferenceGrpcClient("127.0.0.1", 40123);

        val tg = TFGraphMapper.getInstance().importGraph(new ClassPathResource("tf_graphs/examples/expand_dim/frozen_model.pb").getInputStream());
        assertNotNull(tg);

        client.registerGraph(119, tg, ExecutorConfiguration.builder().outputMode(OutputMode.IMPLICIT).build());
    }
}