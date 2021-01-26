/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package com.atilika.kuromoji.ipadic;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.Repeat;
import org.junit.Test;

import static com.atilika.kuromoji.TestUtils.assertCanTokenizeString;

public class RandomizedInputTest extends RandomizedTest {

    private static final int LENGTH = 1024;

    private Tokenizer tokenizer = new Tokenizer();

    @Test
    @Repeat(iterations = 10)
    public void testRandomizedUnicodeInput() {
        assertCanTokenizeString(randomUnicodeOfLength(LENGTH), tokenizer);
    }

    @Test
    @Repeat(iterations = 10)
    public void testRandomizedRealisticUnicodeInput() {
        assertCanTokenizeString(randomRealisticUnicodeOfLength(LENGTH), tokenizer);
    }

    @Test
    @Repeat(iterations = 10)
    public void testRandomizedAsciiInput() {
        assertCanTokenizeString(randomAsciiOfLength(LENGTH), tokenizer);
    }
}
