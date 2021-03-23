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

import org.deeplearning4j.BaseDL4JTest;


import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.resources.Resources;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;

import java.io.File;

import static org.junit.jupiter.api.Assertions.assertEquals;
@Tag(TagNames.FILE_IO)
@NativeTag
public class BasicLabelAwareIteratorTest extends BaseDL4JTest {



    @BeforeEach
    public void setUp() throws Exception {}

    @Test
    public void testHasNextDocument1() throws Exception {
        File inputFile = Resources.asFile("big/raw_sentences.txt");
        SentenceIterator iter = new BasicLineIterator(inputFile.getAbsolutePath());

        BasicLabelAwareIterator iterator = new BasicLabelAwareIterator.Builder(iter).setLabelTemplate("DOCZ_").build();

        int cnt = 0;
        while (iterator.hasNextDocument()) {
            iterator.nextDocument();
            cnt++;
        }

        assertEquals(97162, cnt);

        LabelsSource generator = iterator.getLabelsSource();

        assertEquals(97162, generator.getLabels().size());
        assertEquals("DOCZ_0", generator.getLabels().get(0));
    }

    @Test
    public void testHasNextDocument2() throws Exception {

        File inputFile = Resources.asFile("big/raw_sentences.txt");
        SentenceIterator iter = new BasicLineIterator(inputFile.getAbsolutePath());

        BasicLabelAwareIterator iterator = new BasicLabelAwareIterator.Builder(iter).setLabelTemplate("DOCZ_").build();

        int cnt = 0;
        while (iterator.hasNextDocument()) {
            iterator.nextDocument();
            cnt++;
        }

        assertEquals(97162, cnt);

        iterator.reset();

        cnt = 0;
        while (iterator.hasNextDocument()) {
            iterator.nextDocument();
            cnt++;
        }

        assertEquals(97162, cnt);

        LabelsSource generator = iterator.getLabelsSource();

        // this is important moment. Iterator after reset should not increase number of labels attained
        assertEquals(97162, generator.getLabels().size());
        assertEquals("DOCZ_0", generator.getLabels().get(0));
    }
}
