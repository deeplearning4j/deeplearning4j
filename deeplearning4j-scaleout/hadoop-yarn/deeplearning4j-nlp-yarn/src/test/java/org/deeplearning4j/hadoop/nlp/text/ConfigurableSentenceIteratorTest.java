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

package org.deeplearning4j.hadoop.nlp.text;

import static org.junit.Assert.*;

import org.apache.commons.io.FileUtils;
import org.apache.hadoop.conf.Configuration;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.File;
import java.io.IOException;

/**
 * Created by agibsonccc on 1/29/15.
 */
public class ConfigurableSentenceIteratorTest {
    private Configuration conf;

    @Before
    public void before() throws IOException {
        conf = new Configuration();
        conf.set("fs.defaultFS", "file:///");
        File parentDir = new File("parent");
        parentDir.mkdir();
        FileUtils.writeStringToFile(new File(parentDir,"touch"),"hello");
        conf.set(ConfigurableSentenceIterator.ROOT_PATH,parentDir.toURI().toString());


    }

    @After
    public void after() throws IOException {
        FileUtils.deleteDirectory(new File("parent"));
    }

    @Test
    public void testSentenceIterator() throws IOException {
        TestConfigurableSentenceIterator iter = new TestConfigurableSentenceIterator(conf);
        assertEquals(true,iter.hasNext());
        String next = iter.nextSentence();
        assertEquals("hello",next);
        assertFalse(iter.hasNext());

    }

}
