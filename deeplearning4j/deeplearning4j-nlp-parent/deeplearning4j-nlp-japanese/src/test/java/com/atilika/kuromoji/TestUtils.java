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
package com.atilika.kuromoji;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.Assert.*;

public class TestUtils {

    public static void assertTokenSurfacesEquals(List<String> expectedSurfaces,
                    List<? extends TokenBase> actualTokens) {
        List<String> actualSurfaces = new ArrayList<>();

        for (TokenBase token : actualTokens) {
            actualSurfaces.add(token.getSurface());
        }

        assertEquals(expectedSurfaces, actualSurfaces);
    }

    public static void assertCanTokenizeStream(InputStream untokenizedInput, TokenizerBase tokenizer)
                    throws IOException {
        BufferedReader untokenizedInputReader =
                        new BufferedReader(new InputStreamReader(untokenizedInput, StandardCharsets.UTF_8));

        String untokenizedLine;

        while ((untokenizedLine = untokenizedInputReader.readLine()) != null) {
            assertCanTokenizeString(untokenizedLine, tokenizer);
        }

        assertTrue(true);
    }

    public static void assertCanTokenizeString(String input, TokenizerBase tokenizer) {
        List<? extends TokenBase> tokens = tokenizer.tokenize(input);

        if (input.length() > 0) {
            assertFalse(tokens.isEmpty());
        } else {
            assertTrue(tokens.isEmpty());
        }
    }

    public static void assertTokenizedStreamEquals(InputStream tokenizedInput, InputStream untokenizedInput,
                    TokenizerBase tokenizer) throws IOException {
        BufferedReader untokenizedInputReader =
                        new BufferedReader(new InputStreamReader(untokenizedInput, StandardCharsets.UTF_8));
        BufferedReader tokenizedInputReader =
                        new BufferedReader(new InputStreamReader(tokenizedInput, StandardCharsets.UTF_8));

        String untokenizedLine;

        while ((untokenizedLine = untokenizedInputReader.readLine()) != null) {
            List<? extends TokenBase> tokens = tokenizer.tokenize(untokenizedLine);

            for (TokenBase token : tokens) {
                String tokenLine = tokenizedInputReader.readLine();

                assertNotNull(tokenLine);

                // TODO: Verify if this tab handling is correct...
                String[] parts = tokenLine.split("\\t", 2);
                String surface = parts[0];
                String features = parts[1];

                assertEquals(surface, token.getSurface());
                assertEquals(features, token.getAllFeatures());
            }
        }
    }

    public static void assertMultiThreadedTokenizedStreamEquals(int numThreads, final int perThreadRuns,
                    final String tokenizedInputResource, final String untokenizedInputResource,
                    final TokenizerBase tokenizer) throws IOException, InterruptedException {
        List<Thread> threads = new ArrayList<>();

        for (int i = 0; i < numThreads; i++) {
            Thread thread = new Thread(new Runnable() {
                @Override
                public void run() {
                    for (int run = 0; run < perThreadRuns; run++) {

                        //                            System.out.println(Thread.currentThread().getName() + ": tokenizer run " + run);

                        try {
                            InputStream tokenizedInput = getClass().getResourceAsStream(tokenizedInputResource);
                            InputStream untokenizedInput = getClass().getResourceAsStream(untokenizedInputResource);

                            assertTokenizedStreamEquals(tokenizedInput, untokenizedInput, tokenizer);

                            untokenizedInput.close();
                            tokenizedInput.close();
                        } catch (IOException e) {
                            fail(e.getMessage());
                        }
                    }
                }
            });
            threads.add(thread);
            thread.start();
        }

        for (Thread thread : threads) {
            thread.join();
        }

        assertTrue(true);
    }

    public static void assertEqualTokenFeatureLengths(String text, TokenizerBase tokenizer) {
        List<? extends TokenBase> tokens = tokenizer.tokenize(text);
        Set<Integer> lengths = new HashSet<>();

        for (TokenBase token : tokens) {
            lengths.add(token.getAllFeaturesArray().length);
        }

        assertEquals(1, lengths.size());
    }
}
