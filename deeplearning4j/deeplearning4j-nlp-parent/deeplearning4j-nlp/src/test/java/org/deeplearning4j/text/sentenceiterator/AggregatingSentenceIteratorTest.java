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

import java.io.File;

import static org.junit.Assert.assertEquals;

/**
 * Created by fartovii on 01.12.15.
 */
public class AggregatingSentenceIteratorTest {

    @Test
    public void testHasNext() throws Exception {
        ClassPathResource resource = new ClassPathResource("/big/raw_sentences.txt");
        File file = resource.getFile();
        BasicLineIterator iterator = new BasicLineIterator(file);
        BasicLineIterator iterator2 = new BasicLineIterator(file);

        AggregatingSentenceIterator aggr = new AggregatingSentenceIterator.Builder().addSentenceIterator(iterator)
                        .addSentenceIterator(iterator2).build();

        int cnt = 0;
        while (aggr.hasNext()) {
            String line = aggr.nextSentence();
            cnt++;
        }

        assertEquals((97162 * 2), cnt);

        aggr.reset();

        while (aggr.hasNext()) {
            String line = aggr.nextSentence();
            cnt++;
        }

        assertEquals((97162 * 4), cnt);
    }
}
