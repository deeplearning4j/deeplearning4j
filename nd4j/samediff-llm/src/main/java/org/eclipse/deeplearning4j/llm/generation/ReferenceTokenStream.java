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

package org.eclipse.deeplearning4j.llm.generation;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Utility for loading a reference token stream from a flat text file.
 *
 * <p>The file format is whitespace-separated integer token IDs, one or more
 * per line. Lines (or portions of lines) beginning with {@code #} are treated
 * as comments and ignored. Blank lines are also ignored.</p>
 *
 * <p>Example file:</p>
 * <pre>
 * # SmolDocling reference output — 20 tokens
 * 1 2 3 4 5
 * 6 7 8   # trailing comment
 * </pre>
 */
public final class ReferenceTokenStream {

    private ReferenceTokenStream() {
        // utility class — no instantiation
    }

    /**
     * Loads a reference token stream from the file at {@code path}.
     *
     * @param path absolute or relative path to the token file
     * @return array of token IDs in file order
     * @throws IllegalArgumentException if the file does not exist or contains a non-integer token
     * @throws RuntimeException         if the file cannot be read
     */
    public static int[] load(String path) {
        Path p = Paths.get(path);
        if (!Files.exists(p)) {
            throw new IllegalArgumentException("Reference token stream file not found: " + path);
        }
        try {
            List<String> lines = Files.readAllLines(p);
            List<Integer> tokens = new ArrayList<>();
            for (String raw : lines) {
                int hash = raw.indexOf('#');
                String line = (hash >= 0 ? raw.substring(0, hash) : raw).trim();
                if (line.isEmpty()) continue;
                for (String tok : line.split("\\s+")) {
                    if (tok.isEmpty()) continue;
                    try {
                        tokens.add(Integer.parseInt(tok));
                    } catch (NumberFormatException e) {
                        throw new IllegalArgumentException(
                                "Reference token stream contains non-integer token '"
                                        + tok + "' in " + path, e);
                    }
                }
            }
            return tokens.stream().mapToInt(Integer::intValue).toArray();
        } catch (IOException e) {
            throw new RuntimeException("Failed to read reference token stream: " + path, e);
        }
    }

    /**
     * Reads the file path from the named system property and delegates to {@link #load(String)}.
     *
     * @param propertyName system property whose value is the path to the token file
     * @return array of token IDs, or {@code null} if the property is unset or empty
     * @throws IllegalArgumentException if the property is set but the file does not exist or
     *                                  contains a non-integer token
     * @throws RuntimeException         if the file cannot be read
     */
    public static int[] loadFromSystemProperty(String propertyName) {
        String path = System.getProperty(propertyName);
        if (path == null || path.isEmpty()) return null;
        return load(path);
    }
}
