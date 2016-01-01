/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.cli.api.flags;

import org.canova.api.conf.Configuration;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;

/**
 * @author sonali
 */
public class Properties implements Flag {
    private java.util.Properties props = new java.util.Properties();

    @Override
    public <E> E value(String value) throws Exception {
        File file = new File(value);
        BufferedInputStream bis = new BufferedInputStream(new FileInputStream(file));
        props.load(bis);
        bis.close();
        Configuration configuration = new Configuration();
        for(String key : props.stringPropertyNames()) {
            configuration.set(key,props.getProperty(key));
        }
        return (E) configuration;

    }
}
