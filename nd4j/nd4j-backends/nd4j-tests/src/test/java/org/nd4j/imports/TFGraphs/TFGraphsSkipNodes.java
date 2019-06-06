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

package org.nd4j.imports.TFGraphs;

import java.util.*;

/**
 * Created by susaneraly on 2/20/18.
 */
public class TFGraphsSkipNodes {

    private static final Map<String, List<String>> SKIP_NODE_MAP = Collections.unmodifiableMap(
            new HashMap<String, List<String>>() {{
                //Note that we are testing equality with keep_prob of 1.0
                //The following are all dependent on rng seed and will fail. All other nodes pass.
                put("deep_mnist",
                        new ArrayList<>(Arrays.asList("dropout/dropout/random_uniform/RandomUniform",
                                "dropout/dropout/random_uniform/mul",
                                "dropout/dropout/random_uniform",
                                "dropout/dropout/add")));
            }});

    public static boolean skipNode(String modelName, String varName) {

        if (!SKIP_NODE_MAP.keySet().contains(modelName)) {
            return false;
        } else {
            for (String some_node : SKIP_NODE_MAP.get(modelName)) {
                if (some_node.equals(varName)) {
                    return true;
                }
            }
            return false;
        }

    }
}
