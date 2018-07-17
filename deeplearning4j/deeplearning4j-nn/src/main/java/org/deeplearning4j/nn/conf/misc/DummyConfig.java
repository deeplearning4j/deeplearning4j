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

package org.deeplearning4j.nn.conf.misc;

import lombok.AllArgsConstructor;
import org.deeplearning4j.nn.api.TrainingConfig;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.NoOp;

/**
 * A 'dummy' training configuration for use in frozen layers
 *
 * @author Alex Black
 */
@AllArgsConstructor
public class DummyConfig implements TrainingConfig {
    private final String name;

    @Override
    public String getLayerName() {
        return name;
    }

    @Override
    public boolean isPretrain() {
        return false;
    }

    @Override
    public double getL1ByParam(String paramName) {
        return 0;
    }

    @Override
    public double getL2ByParam(String paramName) {
        return 0;
    }

    @Override
    public boolean isPretrainParam(String paramName) {
        return false;
    }

    @Override
    public IUpdater getUpdaterByParam(String paramName) {
        return new NoOp();
    }

    @Override
    public GradientNormalization getGradientNormalization() {
        return GradientNormalization.None;
    }

    @Override
    public double getGradientNormalizationThreshold() {
        return 1.0;
    }

    @Override
    public void setPretrain(boolean pretrain) {
        //No op
    }
}
