/*
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

package org.datavec.api.util;

import org.junit.Before;
import org.junit.Test;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStream;
import java.io.InputStreamReader;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
public class ClassPathResourceTest {

    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void testGetFile1() throws Exception {
        File intFile = new ClassPathResource("iris.dat").getFile();

        assertTrue(intFile.exists());
        assertEquals(2700, intFile.length());
    }

    @Test
    public void testGetFileSlash1() throws Exception {
        File intFile = new ClassPathResource("/iris.dat").getFile();

        assertTrue(intFile.exists());
        assertEquals(2700, intFile.length());
    }

    @Test
    public void testGetFileWithSpace1() throws Exception {
        File intFile = new ClassPathResource("csvsequence test.txt").getFile();

        assertTrue(intFile.exists());
        assertEquals(60, intFile.length());
    }

    @Test
    public void testInputStream() throws Exception {
        ClassPathResource resource = new ClassPathResource("csvsequence_1.txt");
        File intFile = resource.getFile();

        assertEquals(60, intFile.length());

        InputStream stream = resource.getInputStream();
        BufferedReader reader = new BufferedReader(new InputStreamReader(stream));
        String line = "";
        int cnt = 0;
        while ((line = reader.readLine()) != null) {
            cnt++;
        }

        assertEquals(5, cnt);
    }

    @Test
    public void testInputStreamSlash() throws Exception {
        ClassPathResource resource = new ClassPathResource("/csvsequence_1.txt");
        File intFile = resource.getFile();

        assertEquals(60, intFile.length());

        InputStream stream = resource.getInputStream();
        BufferedReader reader = new BufferedReader(new InputStreamReader(stream));
        String line = "";
        int cnt = 0;
        while ((line = reader.readLine()) != null) {
            cnt++;
        }

        assertEquals(5, cnt);
    }
}