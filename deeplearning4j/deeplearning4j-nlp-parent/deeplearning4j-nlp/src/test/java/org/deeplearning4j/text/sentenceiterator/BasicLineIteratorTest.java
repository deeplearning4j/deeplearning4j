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

package org.deeplearning4j.text.sentenceiterator;

import org.nd4j.linalg.io.ClassPathResource;
import org.junit.Before;
import org.junit.Test;

import java.io.File;
import java.io.FileInputStream;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
public class BasicLineIteratorTest {

    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void testHasMoreLinesFile() throws Exception {
        ClassPathResource resource = new ClassPathResource("/big/raw_sentences.txt");
        File file = resource.getFile();
        BasicLineIterator iterator = new BasicLineIterator(file);

        int cnt = 0;
        while (iterator.hasNext()) {
            String line = iterator.nextSentence();
            cnt++;
        }

        assertEquals(97162, cnt);

        iterator.reset();

        cnt = 0;
        while (iterator.hasNext()) {
            String line = iterator.nextSentence();
            cnt++;
        }

        assertEquals(97162, cnt);
    }

    @Test
    public void testHasMoreLinesStream() throws Exception {
        ClassPathResource resource = new ClassPathResource("/big/raw_sentences.txt");
        File file = resource.getFile(); //.getParentFile();
        BasicLineIterator iterator = new BasicLineIterator(new FileInputStream(file));

        int cnt = 0;
        while (iterator.hasNext()) {
            String line = iterator.nextSentence();
            cnt++;
        }

        assertEquals(97162, cnt);

        iterator.reset();

        cnt = 0;
        while (iterator.hasNext()) {
            String line = iterator.nextSentence();
            cnt++;
        }

        assertEquals(97162, cnt);
    }
}
