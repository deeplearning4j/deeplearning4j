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


import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.common.resources.Resources;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

@Tag(TagNames.FILE_IO)
@NativeTag
public class FilenamesLabelAwareIteratorTest extends BaseDL4JTest {


    @BeforeEach
    public void setUp() throws Exception {

    }

    @Test
    public void testNextDocument(@TempDir Path testDir) throws Exception {
        val tempDir = testDir.toFile();
        Resources.copyDirectory("/big/", tempDir);

        FilenamesLabelAwareIterator iterator = new FilenamesLabelAwareIterator.Builder()
                        .addSourceFolder(tempDir).useAbsolutePathAsLabel(false).build();

        List<String> labels = new ArrayList<>();

        LabelledDocument doc1 = iterator.nextDocument();
        labels.add(doc1.getLabel());

        LabelledDocument doc2 = iterator.nextDocument();
        labels.add(doc2.getLabel());

        LabelledDocument doc3 = iterator.nextDocument();
        labels.add(doc3.getLabel());

        LabelledDocument doc4 = iterator.nextDocument();
        labels.add(doc4.getLabel());

        LabelledDocument doc5 = iterator.nextDocument();
        labels.add(doc5.getLabel());

        LabelledDocument doc6 = iterator.nextDocument();
        labels.add(doc6.getLabel());

        assertFalse(iterator.hasNextDocument());

        System.out.println("Labels: " + labels);

        assertTrue(labels.contains("coc.txt"));
        assertTrue(labels.contains("occurrences.txt"));
        assertTrue(labels.contains("raw_sentences.txt"));
        assertTrue(labels.contains("tokens.txt"));
        assertTrue(labels.contains("raw_sentences_2.txt"));
        assertTrue(labels.contains("rnj.txt"));

    }
}
