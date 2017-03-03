/*-
 *  * Copyright 2016 Skymind, Inc.
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
 */

package org.datavec.hadoop.conf;

import org.apache.hadoop.conf.Configuration;
import org.junit.Test;

public class TestConfigurationUtil {

    @Test
    public void testLoadHadoopConfFiles() {

        // this would come from the properties file
        String confPath = "src/test/resources/conf/example_conf/";

        Configuration conf = ConfigurationUtil.generateConfig(confPath);

        System.out.println(" works? " + conf.get("fs.default.name"));


    }

}
