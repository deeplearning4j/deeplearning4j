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

package org.deeplearning4j.cli.flags;

import org.canova.api.conf.Configuration;
import org.deeplearning4j.cli.api.flags.Properties;
import org.deeplearning4j.cli.api.flags.test.BaseFlagTest;
import org.junit.Test;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;

import static org.junit.Assert.assertEquals;

/**
 * @author sonali
 */
public class PropertiesFlagTest extends BaseFlagTest {

    @Test
    public void test() throws Exception {
        File file = new ClassPathResource("testConfig.txt").getFile();
        Properties propertiesFlag = new Properties();
        Configuration testConfig = propertiesFlag.value(file.getAbsolutePath());

        assertEquals("1", testConfig.get("one"));
        assertEquals("2", testConfig.get("two"));

    }
}
