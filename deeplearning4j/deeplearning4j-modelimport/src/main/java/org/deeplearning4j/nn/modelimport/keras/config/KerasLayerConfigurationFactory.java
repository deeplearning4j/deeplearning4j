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

package org.deeplearning4j.nn.modelimport.keras.config;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;

@Slf4j
public class KerasLayerConfigurationFactory {

    public KerasLayerConfigurationFactory() {
    }

    public static KerasLayerConfiguration get(Integer kerasMajorVersion) throws UnsupportedKerasConfigurationException {
        if (kerasMajorVersion != 1 && kerasMajorVersion != 2)
            throw new UnsupportedKerasConfigurationException(
                    "Keras major version has to be either 1 or 2 (" + kerasMajorVersion + " provided)");
        else if (kerasMajorVersion == 1)
            return new Keras1LayerConfiguration();
        else
            return new Keras2LayerConfiguration();
    }
}
