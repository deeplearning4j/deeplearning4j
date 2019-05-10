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
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileInputStream;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

/**
 * Created by fartovii on 09.11.15.
 */
public class StreamLineIteratorTest {

    protected Logger logger = LoggerFactory.getLogger(StreamLineIteratorTest.class);

    @Test
    public void testHasNext() throws Exception {

        ClassPathResource reuters5250 = new ClassPathResource("/reuters/5250");
        File f = reuters5250.getFile();

        StreamLineIterator iterator = new StreamLineIterator.Builder(new FileInputStream(f)).setFetchSize(100).build();

        int cnt = 0;
        while (iterator.hasNext()) {
            String line = iterator.nextSentence();

            assertNotEquals(null, line);
            logger.info("Line: " + line);
            cnt++;
        }

        assertEquals(24, cnt);
    }
}
