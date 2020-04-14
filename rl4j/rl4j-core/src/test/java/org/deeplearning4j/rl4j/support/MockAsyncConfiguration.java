/*******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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

package org.deeplearning4j.rl4j.support;

import lombok.AllArgsConstructor;
import lombok.Value;
import org.deeplearning4j.rl4j.learning.configuration.IAsyncLearningConfiguration;

@Value
@AllArgsConstructor
public class MockAsyncConfiguration implements IAsyncLearningConfiguration {

    private Long seed;
    private int maxEpochStep;
    private int maxStep;
    private int updateStart;
    private double rewardFactor;
    private double gamma;
    private double errorClamp;
    private int numThreads;
    private int nStep;
    private int learnerUpdateFrequency;
}
