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

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.linalg.env.EnvironmentalAction;
import org.nd4j.linalg.factory.Nd4j;

@Slf4j
public class ConcurrencyAction implements EnvironmentalAction {
    @Override
    public String targetVariable() {
        return "ND4J_SKIP_THREAD_SAFETY_CHECKS";
    }

    @Override
    public void process(String value) {
        val v = Boolean.valueOf(value);

        Nd4j.skipThreadSafetyChecks(v);
    }
}
