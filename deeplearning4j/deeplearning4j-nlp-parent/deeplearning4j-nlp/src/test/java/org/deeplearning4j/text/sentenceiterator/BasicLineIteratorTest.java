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

package org.deeplearning4j.text.sentenceiterator;

import org.deeplearning4j.BaseDL4JTest;
import org.junit.Rule;
import org.junit.rules.Timeout;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.common.resources.Resources;

import java.io.File;
import java.io.FileInputStream;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
public class BasicLineIteratorTest extends BaseDL4JTest {

    @Rule
    public Timeout timeout = Timeout.seconds(300);

    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void testHasMoreLinesFile() throws Exception {
        File file = Resources.asFile("/big/raw_sentences.txt");
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
        File file = Resources.asFile("/big/raw_sentences.txt");
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
