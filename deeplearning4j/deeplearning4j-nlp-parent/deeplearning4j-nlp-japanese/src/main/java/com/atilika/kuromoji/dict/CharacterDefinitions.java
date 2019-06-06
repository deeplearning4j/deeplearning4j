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
package com.atilika.kuromoji.dict;

import com.atilika.kuromoji.io.IntegerArrayIO;
import com.atilika.kuromoji.io.StringArrayIO;
import com.atilika.kuromoji.util.KuromojiBinFilesFetcher;
import com.atilika.kuromoji.util.ResourceResolver;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;

public final class CharacterDefinitions {

    //    public static final String CHARACTER_DEFINITIONS_FILENAME = "characterDefinitions.bin";
    public static final String CHARACTER_DEFINITIONS_FILENAME = new File(KuromojiBinFilesFetcher.getKuromojiRoot(), "characterDefinitions.bin").getAbsolutePath();

    public static final int INVOKE = 0;

    public static final int GROUP = 1;

    private static final String DEFAULT_CATEGORY = "DEFAULT";

    private static final int LENGTH = 2; // Not used as of now

    private final int[][] categoryDefinitions;

    private final int[][] codepointMappings;

    private final String[] categorySymbols;

    private final int[] defaultCategory;

    public CharacterDefinitions(int[][] categoryDefinitions, int[][] codepointMappings, String[] categorySymbols) {
        this.categoryDefinitions = categoryDefinitions;
        this.codepointMappings = codepointMappings;
        this.categorySymbols = categorySymbols;
        this.defaultCategory = lookupCategories(new String[] {DEFAULT_CATEGORY});
    }

    public int[] lookupCategories(char c) {
        int[] mappings = codepointMappings[c];

        if (mappings == null) {
            return defaultCategory;
        }

        return mappings;
    }

    public int[] lookupDefinition(int category) {
        return categoryDefinitions[category];
    }

    public static CharacterDefinitions newInstance(ResourceResolver resolver) throws IOException {
        InputStream charDefInput = resolver.resolve(CHARACTER_DEFINITIONS_FILENAME);

        int[][] definitions = IntegerArrayIO.readSparseArray2D(charDefInput);
        int[][] mappings = IntegerArrayIO.readSparseArray2D(charDefInput);
        String[] symbols = StringArrayIO.readArray(charDefInput);

        CharacterDefinitions characterDefinition = new CharacterDefinitions(definitions, mappings, symbols);

        return characterDefinition;
    }

    public void setCategories(char c, String[] categoryNames) {
        codepointMappings[c] = lookupCategories(categoryNames);
    }

    private int[] lookupCategories(String[] categoryNames) {
        int[] categories = new int[categoryNames.length];

        for (int i = 0; i < categoryNames.length; i++) {
            String category = categoryNames[i];
            int categoryIndex = -1;

            for (int j = 0; j < categorySymbols.length; j++) {
                if (category.equals(categorySymbols[j])) {
                    categoryIndex = j;
                }
            }

            if (categoryIndex < 0) {
                throw new RuntimeException("No category '" + category + "' found");
            }

            categories[i] = categoryIndex;
        }

        return categories;
    }
}
