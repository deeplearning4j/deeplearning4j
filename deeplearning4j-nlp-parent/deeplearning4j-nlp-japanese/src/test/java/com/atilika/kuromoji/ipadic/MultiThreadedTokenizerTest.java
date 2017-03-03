/*-*
 * Copyright © 2010-2015 Atilika Inc. and contributors (see CONTRIBUTORS.md)
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License.  A copy of the
 * License is distributed with this work in the LICENSE.md file.  You may
 * also obtain a copy of the License from
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.atilika.kuromoji.ipadic;

import org.junit.Test;

import java.io.IOException;

import static com.atilika.kuromoji.TestUtils.assertMultiThreadedTokenizedStreamEquals;

public class MultiThreadedTokenizerTest {

    @Test
    public void testMultiThreadedBocchan() throws IOException, InterruptedException {
        assertMultiThreadedTokenizedStreamEquals(5, 25, "/bocchan-ipadic-features.txt", "/bocchan.txt",
                        new Tokenizer());
    }

    @Test
    public void testMultiThreadedUserDictionary() throws IOException, InterruptedException {
        assertMultiThreadedTokenizedStreamEquals(5, 250, "/jawikisentences-ipadic-features.txt", "/jawikisentences.txt",
                        new Tokenizer.Builder().userDictionary(getClass().getResourceAsStream("/userdict.txt"))
                                        .build());
    }
}
