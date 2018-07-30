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
import org.junit.Ignore;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.InputStream;

import static org.junit.Assert.assertEquals;

/**
 * Created by fartovii on 09.11.15.
 */

@Ignore
public class FileDocumentIteratorTest {

    private static final Logger log = LoggerFactory.getLogger(FileDocumentIteratorTest.class);

    @Before
    public void setUp() throws Exception {

    }

    /**
     * Checks actual number of documents retrieved by DocumentIterator
     * @throws Exception
     */
    @Test
    public void testNextDocument() throws Exception {
        ClassPathResource reuters5250 = new ClassPathResource("/reuters/5250");
        File f = reuters5250.getFile();

        DocumentIterator iter = new FileDocumentIterator(f.getAbsolutePath());

        log.info(f.getAbsolutePath());

        int cnt = 0;
        while (iter.hasNext()) {
            InputStream stream = iter.nextDocument();
            stream.close();
            cnt++;
        }

        assertEquals(24, cnt);
    }


    /**
     * Checks actual number of documents retrieved by DocumentIterator after being RESET
     * @throws Exception
     */
    @Test
    public void testDocumentReset() throws Exception {
        ClassPathResource reuters5250 = new ClassPathResource("/reuters/5250");
        File f = reuters5250.getFile();

        DocumentIterator iter = new FileDocumentIterator(f.getAbsolutePath());

        int cnt = 0;
        while (iter.hasNext()) {
            InputStream stream = iter.nextDocument();
            stream.close();
            cnt++;
        }

        iter.reset();

        while (iter.hasNext()) {
            InputStream stream = iter.nextDocument();
            stream.close();
            cnt++;
        }

        assertEquals(48, cnt);
    }
}
