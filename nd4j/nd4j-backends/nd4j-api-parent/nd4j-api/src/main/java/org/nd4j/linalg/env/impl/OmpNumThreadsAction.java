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

package org.nd4j.linalg.env.impl;

import lombok.val;
import org.nd4j.config.ND4JEnvironmentVars;
import org.nd4j.linalg.api.memory.enums.DebugMode;
import org.nd4j.linalg.env.EnvironmentalAction;
import org.nd4j.linalg.factory.Nd4j;

public class OmpNumThreadsAction implements EnvironmentalAction {
    @Override
    public String targetVariable() {
        return ND4JEnvironmentVars.OMP_NUM_THREADS;
    }

    @Override
    public void process(String value) {
        val v = Integer.valueOf(value).intValue();

        val skipper = System.getenv(ND4JEnvironmentVars.ND4J_SKIP_BLAS_THREADS);
        if (skipper == null) {
            // we infer num threads only if skipper undefined
            Nd4j.setNumThreads(v);
        }
    }
}
