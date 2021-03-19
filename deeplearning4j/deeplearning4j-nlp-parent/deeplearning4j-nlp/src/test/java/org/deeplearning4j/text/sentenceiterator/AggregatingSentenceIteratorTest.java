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
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.nd4j.common.resources.Resources;

import java.io.File;

import static org.junit.jupiter.api.Assertions.assertEquals;

@Disabled("Permissions issues on CI")
public class AggregatingSentenceIteratorTest extends BaseDL4JTest {

    @Test()
    @Timeout(30000)
    public void testHasNext() throws Exception {
        File file = Resources.asFile("/big/raw_sentences.txt");
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
