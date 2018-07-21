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

import static org.junit.Assert.*;

/**
 * Created by raver119 on 03.01.16.
 */
public class FileLabelAwareIteratorTest {


    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void testExtractLabelFromPath1() throws Exception {
        ClassPathResource resource = new ClassPathResource("/labeled");

        FileLabelAwareIterator iterator =
                        new FileLabelAwareIterator.Builder().addSourceFolder(resource.getFile()).build();

        int cnt = 0;
        while (iterator.hasNextDocument()) {
            LabelledDocument document = iterator.nextDocument();
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
    public void testExtractLabelFromPath2() throws Exception {
        ClassPathResource resource = new ClassPathResource("/labeled");
        ClassPathResource resource2 = new ClassPathResource("/rootdir");

        FileLabelAwareIterator iterator = new FileLabelAwareIterator.Builder().addSourceFolder(resource.getFile())
                        .addSourceFolder(resource2.getFile()).build();

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
