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

import org.nd4j.linalg.io.ClassPathResource;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * @author raver119@gmail.com
 */
public class FilenamesLabelAwareIteratorTest {

    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void testNextDocument() throws Exception {
        FilenamesLabelAwareIterator iterator = new FilenamesLabelAwareIterator.Builder()
                        .addSourceFolder(new ClassPathResource("/big").getFile()).useAbsolutePathAsLabel(false).build();

        List<String> labels = new ArrayList<>();

        LabelledDocument doc1 = iterator.nextDocument();
        labels.add(doc1.getLabel());

        LabelledDocument doc2 = iterator.nextDocument();
        labels.add(doc2.getLabel());

        LabelledDocument doc3 = iterator.nextDocument();
        labels.add(doc3.getLabel());

        LabelledDocument doc4 = iterator.nextDocument();
        labels.add(doc4.getLabel());

        assertFalse(iterator.hasNextDocument());

        System.out.println("Labels: " + labels);

        assertTrue(labels.contains("coc.txt"));
        assertTrue(labels.contains("occurrences.txt"));
        assertTrue(labels.contains("raw_sentences.txt"));
        assertTrue(labels.contains("tokens.txt"));
    }
}
