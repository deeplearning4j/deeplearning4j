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

package org.deeplearning4j.ui.stats.api;

import org.deeplearning4j.ui.stats.StatsListener;

import java.io.Serializable;

/**
 * Configuration interface for static (unchanging) information, to be reported by {@link StatsListener}.
 * This interface allows for software/hardware/model information to be collected (or, not)
 *
 * @author Alex Black
 */
public interface StatsInitializationConfiguration extends Serializable {

    /**
     * Should software configuration information be collected? For example, OS, JVM, and ND4J backend details
     *
     * @return true if software information should be collected; false if not
     */
    boolean collectSoftwareInfo();

    /**
     * Should hardware configuration information be collected? JVM available processors, number of devices, total memory for each device
     *
     * @return true if hardware information should be collected
     */
    boolean collectHardwareInfo();

    /**
     * Should model information be collected? Model class, configuration (JSON), number of layers, number of parameters, etc.
     *
     * @return true if model information should be collected
     */
    boolean collectModelInfo();

}
