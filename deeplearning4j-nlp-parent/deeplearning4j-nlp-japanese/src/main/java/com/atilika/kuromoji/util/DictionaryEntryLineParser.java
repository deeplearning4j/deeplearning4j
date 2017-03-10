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
package com.atilika.kuromoji.util;

import java.util.ArrayList;
import java.util.List;

public class DictionaryEntryLineParser {

    private static final char QUOTE = '"';
    private static final char COMMA = ',';
    private static final String QUOTE_ESCAPED = "\"\"";

    /**
     * Parse CSV line
     *
     * @param line  line to parse
     * @return String array of parsed valued, null
     * @throws RuntimeException on malformed input
     */
    public static String[] parseLine(String line) {
        boolean insideQuote = false;
        List<String> result = new ArrayList<>();
        StringBuilder builder = new StringBuilder();
        int quoteCount = 0;

        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);

            if (c == QUOTE) {
                insideQuote = !insideQuote;
                quoteCount++;
            }

            if (c == COMMA && !insideQuote) {
                String value = builder.toString();
                value = unescape(value);

                result.add(value);
                builder = new StringBuilder();
                continue;
            }

            builder.append(c);
        }

        result.add(builder.toString());

        if (quoteCount % 2 != 0) {
            throw new RuntimeException("Unmatched quote in entry: " + line);
        }

        return result.toArray(new String[result.size()]);
    }

    /**
     * Unescape input for CSV
     *
     * @param text  text to be unescaped
     * @return unescaped value, not null
     */
    public static String unescape(String text) {
        StringBuilder builder = new StringBuilder();
        boolean foundQuote = false;

        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);

            if (i == 0 && c == QUOTE || i == text.length() - 1 && c == QUOTE) {
                continue;
            }

            if (c == QUOTE) {
                if (foundQuote) {
                    builder.append(QUOTE);
                    foundQuote = false;
                } else {
                    foundQuote = true;
                }
            } else {
                foundQuote = false;
                builder.append(c);
            }
        }

        return builder.toString();
    }

    /**
     * Escape input for CSV
     *
     * @param text  text to be escaped
     * @return escaped value, not null
     */
    public static String escape(String text) {
        boolean hasQuote = text.indexOf(QUOTE) >= 0;
        boolean hasComma = text.indexOf(COMMA) >= 0;

        if (!(hasQuote || hasComma)) {
            return text;
        }

        StringBuilder builder = new StringBuilder();

        if (hasQuote) {
            for (int i = 0; i < text.length(); i++) {
                char c = text.charAt(i);

                if (c == QUOTE) {
                    builder.append(QUOTE_ESCAPED);
                } else {
                    builder.append(c);
                }
            }
        } else {
            builder.append(text);
        }

        if (hasComma) {
            builder.insert(0, QUOTE);
            builder.append(QUOTE);
        }
        return builder.toString();
    }
}
