/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.common.resources;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.io.LineIterator;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.common.resources.Resources;
import org.nd4j.common.resources.strumpf.StrumpfResolver;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.Reader;
import java.nio.charset.StandardCharsets;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class TestStrumpf {

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @Test
    public void testResolvingReference() throws Exception {

        File f = Resources.asFile("big/raw_sentences.txt");
        assertTrue(f.exists());

        System.out.println(f.getAbsolutePath());
        try(Reader r = new BufferedReader(new FileReader(f))){
            LineIterator iter = IOUtils.lineIterator(r);
            for( int i=0; i<5 && iter.hasNext(); i++ ){
                System.out.println("LINE " + i + ": " + iter.next());
            }
        }
    }

    @Test
    public void testResolvingActual() throws Exception {
        File f = Resources.asFile("data/irisSvmLight.txt");
        assertTrue(f.exists());

        //System.out.println(f.getAbsolutePath());
        int count = 0;
        try(Reader r = new BufferedReader(new FileReader(f))){
            LineIterator iter = IOUtils.lineIterator(r);
            while(iter.hasNext()){
                String line = iter.next();
                //System.out.println("LINE " + i + ": " + line);
                count++;
            }
        }

        assertEquals(12, count);        //Iris normally has 150 examples; this is subset with 12
    }

    @Test
    public void testResolveLocal() throws Exception {

        File dir = testDir.newFolder();

        String content = "test file content";
        String path = "myDir/myTestFile.txt";
        File testFile = new File(dir, path);
        testFile.getParentFile().mkdir();
        FileUtils.writeStringToFile(testFile, content, StandardCharsets.UTF_8);

        System.setProperty(ND4JSystemProperties.RESOURCES_LOCAL_DIRS, dir.getAbsolutePath());

        try{
            StrumpfResolver r = new StrumpfResolver();
            assertTrue(r.exists(path));
            File f = r.asFile(path);
            assertTrue(f.exists());
            assertEquals(testFile.getAbsolutePath(), f.getAbsolutePath());
            String s = FileUtils.readFileToString(f, StandardCharsets.UTF_8);
            assertEquals(content, s);
        } finally {
            System.setProperty(ND4JSystemProperties.RESOURCES_LOCAL_DIRS, "");
        }
    }

}
