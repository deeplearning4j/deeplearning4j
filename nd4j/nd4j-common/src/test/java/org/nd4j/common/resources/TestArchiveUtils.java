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

import org.junit.jupiter.api.Test;

import org.junit.jupiter.api.io.TempDir;
import org.nd4j.common.util.ArchiveUtils;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

public class TestArchiveUtils {

    @Test
    public void testUnzipFileTo(@TempDir Path testDir) throws IOException {
        //random txt file
        File dir = testDir.toFile();
        String content = "test file content";
        String path = "myDir/myTestFile.txt";
        File testFile = new File(dir, path);
        testFile.getParentFile().mkdir();
        FileUtils.writeStringToFile(testFile, content, StandardCharsets.UTF_8);

        //zip it as test.zip
        File zipFile = new File(testFile.getParentFile(),"test.zip");
        FileOutputStream fos = new FileOutputStream(zipFile);
        ZipOutputStream zipOut = new ZipOutputStream(fos);
        FileInputStream fis = new FileInputStream(testFile);
        ZipEntry zipEntry = new ZipEntry(testFile.getName());
        zipOut.putNextEntry(zipEntry);
        byte[] bytes = new byte[1024];
        int length;
        while((length = fis.read(bytes)) >= 0) {
            zipOut.write(bytes, 0, length);
        }
        zipOut.close();
        fis.close();
        fos.close();

        //now unzip to a directory that doesn't previously exist
        File unzipDir = new File(testFile.getParentFile(),"unzipTo");
        ArchiveUtils.unzipFileTo(zipFile.getAbsolutePath(),unzipDir.getAbsolutePath());
    }
}
