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

package org.deeplearning4j.text.documentiterator;


import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.junit.Rule;
import org.junit.rules.TemporaryFolder;
import org.nd4j.linalg.io.ClassPathResource;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.HashSet;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * Created by fartovii on 09.11.15.
 */
@Slf4j
@Ignore
public class FileDocumentIteratorTest {

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @Before
    public void setUp() throws Exception {

    }

    /**
     * Checks actual number of documents retrieved by DocumentIterator
     * @throws Exception
     */
    @Test
    public void testNextDocument() throws Exception {
        ClassPathResource reuters5250 = new ClassPathResource("/reuters/5250");
        File f = reuters5250.getFile();

        DocumentIterator iter = new FileDocumentIterator(f.getAbsolutePath());

        log.info(f.getAbsolutePath());

        int cnt = 0;
        while (iter.hasNext()) {
            InputStream stream = iter.nextDocument();
            stream.close();
            cnt++;
        }

        assertEquals(24, cnt);
    }


    /**
     * Checks actual number of documents retrieved by DocumentIterator after being RESET
     * @throws Exception
     */
    @Test
    public void testDocumentReset() throws Exception {
        ClassPathResource reuters5250 = new ClassPathResource("/reuters/5250");
        File f = reuters5250.getFile();

        DocumentIterator iter = new FileDocumentIterator(f.getAbsolutePath());

        int cnt = 0;
        while (iter.hasNext()) {
            InputStream stream = iter.nextDocument();
            stream.close();
            cnt++;
        }

        iter.reset();

        while (iter.hasNext()) {
            InputStream stream = iter.nextDocument();
            stream.close();
            cnt++;
        }

        assertEquals(48, cnt);
    }

    @Test(timeout = 5000L)
    public void testEmptyDocument() throws Exception {
        File f = testDir.newFile();
        assertTrue(f.exists());
        assertEquals(0, f.length());

        try {
            DocumentIterator iter = new FileDocumentIterator(f.getAbsolutePath());
        } catch (Throwable t){
            String msg = t.getMessage();
            assertTrue(msg.contains("empty"));
        }
    }

    @Test(timeout = 5000L)
    public void testEmptyDocument2() throws Exception {
        File dir = testDir.newFolder();
        File f1 = new File(dir, "1.txt");
        FileUtils.writeStringToFile(f1, "line 1\nline2", StandardCharsets.UTF_8);
        File f2 = new File(dir, "2.txt");
        f2.createNewFile();
        File f3 = new File(dir, "3.txt");
        FileUtils.writeStringToFile(f3, "line 3\nline4", StandardCharsets.UTF_8);

        DocumentIterator iter = new FileDocumentIterator(dir);
        int count = 0;
        Set<String> lines = new HashSet<>();
        while(iter.hasNext()){
            String next = IOUtils.readLines(iter.nextDocument(), StandardCharsets.UTF_8).get(0);
            lines.add(next);
        }

        assertEquals(4, lines.size());
    }

}
