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

package org.nd4j.common.loader;

import org.apache.commons.io.FileUtils;

import org.junit.jupiter.api.Test;

import org.junit.jupiter.api.io.TempDir;
import org.nd4j.common.loader.FileBatch;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TestFileBatch {



    @Test
    public void testFileBatch(@TempDir Path testDir) throws Exception {
        File baseDir = testDir.toFile();

        List<File> fileList = new ArrayList<>();
        for( int i = 0; i < 10; i++) {
            String s = "File contents - file " + i;
            File f = new File(baseDir, "origFile" + i + ".txt");
            FileUtils.writeStringToFile(f, s, StandardCharsets.UTF_8);
            fileList.add(f);
        }

        FileBatch fb = FileBatch.forFiles(fileList);

        assertEquals(10, fb.getFileBytes().size());
        assertEquals(10, fb.getOriginalUris().size());
        for( int i = 0; i < 10; i++) {
            byte[] expBytes = ("File contents - file " + i).getBytes(StandardCharsets.UTF_8);
            byte[] actBytes = fb.getFileBytes().get(i);
            assertArrayEquals(expBytes, actBytes);

            String expPath = fileList.get(i).toURI().toString();
            String actPath = fb.getOriginalUris().get(i);
            assertEquals(expPath, actPath);
        }

        //Save and load:
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        fb.writeAsZip(baos);
        byte[] asBytes = baos.toByteArray();

        FileBatch fb2;
        try(ByteArrayInputStream bais = new ByteArrayInputStream(asBytes)){
            fb2 = FileBatch.readFromZip(bais);
        }

        assertEquals(fb.getOriginalUris(), fb2.getOriginalUris());
        assertEquals(10, fb2.getFileBytes().size());
        for( int i = 0; i < 10; i++) {
            assertArrayEquals(fb.getFileBytes().get(i), fb2.getFileBytes().get(i));
        }

        //Check that it is indeed a valid zip file:
        File f = Files.createTempFile(testDir,"testfile","zip").toFile();
        fb.writeAsZip(f);

        ZipFile zf = new ZipFile(f);
        Enumeration<? extends ZipEntry> e = zf.entries();
        int count = 0;
        Set<String> names = new HashSet<>();
        while(e.hasMoreElements()){
            ZipEntry entry = e.nextElement();
            names.add(entry.getName());
        }

        zf.close();
        assertEquals(11, names.size()); //10 files, 1 "original file names" file
        assertTrue(names.contains(FileBatch.ORIGINAL_PATHS_FILENAME));
        for( int i = 0; i < 10; i++) {
            String n = "file_" + i + ".txt";
            assertTrue(names.contains(n),n);
        }
    }

}
