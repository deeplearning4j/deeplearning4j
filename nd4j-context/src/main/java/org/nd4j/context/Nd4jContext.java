/*
 *
 * * Copyright 2015 Skymind,Inc. * * Licensed under the Apache License, Version 2.0 (the "License"); * you may not use
 * this file except in compliance with the License. * You may obtain a copy of the License at * *
 * http://www.apache.org/licenses/LICENSE-2.0 * * Unless required by applicable law or agreed to in writing, software *
 * distributed under the License is distributed on an "AS IS" BASIS, * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
 * either express or implied. * See the License for the specific language governing permissions and * limitations under
 * the License.
 *
 *
 */

package org.nd4j.context;

import java.io.IOException;
import java.io.InputStream;
import java.io.Serializable;
import java.util.Properties;

/**
 * Holds properties for nd4j to be used across different modules
 *
 * @author Adam Gibson
 */
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
     * Load the additional properties from an input stream
     *
     * @param inputStream
     */
    public void updateProperties(InputStream inputStream) {
        try {
            conf.load(inputStream);
        } catch (IOException e) {
            e.printStackTrace();
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
