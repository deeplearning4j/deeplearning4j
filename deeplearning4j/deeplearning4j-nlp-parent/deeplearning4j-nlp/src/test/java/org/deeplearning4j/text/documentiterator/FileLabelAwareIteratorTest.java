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

package org.deeplearning4j.text.documentiterator;

import lombok.val;
import org.deeplearning4j.BaseDL4JTest;


import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.common.io.ClassPathResource;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;

import java.io.File;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

@Tag(TagNames.FILE_IO)
@NativeTag
public class FileLabelAwareIteratorTest extends BaseDL4JTest {


    @BeforeEach
    public void setUp() throws Exception {

    }

    @Test
    public void testExtractLabelFromPath1(@TempDir Path testDir) throws Exception {
        val dir = testDir.resolve("new-folder").toFile();
        dir.mkdirs();
        val resource = new ClassPathResource("/labeled/");
        resource.copyDirectory(dir);

        val iterator = new FileLabelAwareIterator.Builder().addSourceFolder(dir).build();

        int cnt = 0;
        while (iterator.hasNextDocument()) {
            val document = iterator.nextDocument();
            assertNotEquals(null, document);
            assertNotEquals(null, document.getContent());
            assertNotEquals(null, document.getLabel());
            cnt++;
        }

        assertEquals(3, cnt);


        assertEquals(3, iterator.getLabelsSource().getNumberOfLabelsUsed());

        assertTrue(iterator.getLabelsSource().getLabels().contains("positive"));
        assertTrue(iterator.getLabelsSource().getLabels().contains("negative"));
        assertTrue(iterator.getLabelsSource().getLabels().contains("neutral"));
    }


    @Test
    public void testExtractLabelFromPath2(@TempDir Path testDir) throws Exception {
        testDir = testDir.resolve("new-folder");
        testDir.toFile().mkdirs();
        val dir0 = new File(testDir.toFile(),"dir-0");
        val dir1 = new File(testDir.toFile(),"dir-1");
        dir0.mkdirs();
        dir1.mkdirs();
        val resource = new ClassPathResource("/labeled/");
        val resource2 = new ClassPathResource("/rootdir/");
        resource.copyDirectory(dir0);
        resource2.copyDirectory(dir1);

        FileLabelAwareIterator iterator = new FileLabelAwareIterator.Builder().addSourceFolder(dir0)
                        .addSourceFolder(dir1).build();

        int cnt = 0;
        while (iterator.hasNextDocument()) {
            LabelledDocument document = iterator.nextDocument();
            assertNotEquals(null, document);
            assertNotEquals(null, document.getContent());
            assertNotEquals(null, document.getLabel());
            cnt++;
        }

        assertEquals(5, cnt);


        assertEquals(5, iterator.getLabelsSource().getNumberOfLabelsUsed());

        assertTrue(iterator.getLabelsSource().getLabels().contains("positive"));
        assertTrue(iterator.getLabelsSource().getLabels().contains("negative"));
        assertTrue(iterator.getLabelsSource().getLabels().contains("neutral"));
        assertTrue(iterator.getLabelsSource().getLabels().contains("label1"));
        assertTrue(iterator.getLabelsSource().getLabels().contains("label2"));
    }
}
