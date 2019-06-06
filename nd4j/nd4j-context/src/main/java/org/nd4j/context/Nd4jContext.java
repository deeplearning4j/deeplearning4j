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

package org.nd4j.context;

import lombok.extern.slf4j.Slf4j;

import java.io.IOException;
import java.io.InputStream;
import java.io.Serializable;
import java.util.Properties;

/**
 * Holds properties for nd4j to be used across different modules
 *
 * @author Adam Gibson
 */
@Slf4j
public class Nd4jContext implements Serializable {

    private Properties conf;
    private static Nd4jContext INSTANCE = new Nd4jContext();

    private Nd4jContext() {
        conf = new Properties();
        conf.putAll(System.getProperties());
    }

    public static Nd4jContext getInstance() {
        return INSTANCE;
    }

    /**
     * Load the additional properties from an input stream and load all system properties
     *
     * @param inputStream
     */
    public void updateProperties(InputStream inputStream) {
        try {
            conf.load(inputStream);
            conf.putAll(System.getProperties());
        } catch (IOException e) {
            log.warn("Error loading system properties from input stream", e);
        }
    }

    /**
     * Get the configuration for nd4j
     *
     * @return
     */
    public Properties getConf() {
        return conf;
    }
}
