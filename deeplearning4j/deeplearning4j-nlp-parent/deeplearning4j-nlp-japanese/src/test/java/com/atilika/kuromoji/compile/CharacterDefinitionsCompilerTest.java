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
package com.atilika.kuromoji.compile;

import com.atilika.kuromoji.dict.CharacterDefinitions;
import com.atilika.kuromoji.io.IntegerArrayIO;
import com.atilika.kuromoji.io.StringArrayIO;
import org.junit.Before;
import org.junit.Test;

import java.io.*;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.*;

public class CharacterDefinitionsCompilerTest {

    private File charDef;

    private Map<Integer, String> categoryIdMap;

    private CharacterDefinitions characterDefinition;

    @Before
    public void setUp() throws IOException {
        charDef = File.createTempFile("kuromoji-chardef-", ".bin");
        charDef.deleteOnExit();

        CharacterDefinitionsCompiler compiler =
                        new CharacterDefinitionsCompiler(new BufferedOutputStream(new FileOutputStream(charDef)));
        compiler.readCharacterDefinition(new BufferedInputStream(
                        CharacterDefinitionsCompilerTest.class.getClassLoader().getResourceAsStream("deeplearning4j-nlp-japanese/char.def")),
                        "euc-jp");
        categoryIdMap = invert(compiler.makeCharacterCategoryMap());
        compiler.compile();

        InputStream input = new BufferedInputStream(new FileInputStream(charDef));

        int[][] definitions = IntegerArrayIO.readSparseArray2D(input);
        int[][] mappings = IntegerArrayIO.readSparseArray2D(input);
        String[] symbols = StringArrayIO.readArray(input);

        characterDefinition = new CharacterDefinitions(definitions, mappings, symbols);
    }

    @Test
    public void testCharacterCategories() throws IOException {
        // Non-defined characters get the default definition
        assertCharacterCategories(characterDefinition, '\u0000', "DEFAULT");
        assertCharacterCategories(characterDefinition, '〇', "SYMBOL", "KANJI", "KANJINUMERIC");
        assertCharacterCategories(characterDefinition, ' ', "SPACE");
        assertCharacterCategories(characterDefinition, '。', "SYMBOL");
        assertCharacterCategories(characterDefinition, 'A', "ALPHA");
        assertCharacterCategories(characterDefinition, 'Ａ', "ALPHA");
    }

    @Test
    public void testAddCategoryDefinitions() {
        assertCharacterCategories(characterDefinition, '・', "KATAKANA");

        characterDefinition.setCategories('・', new String[] {"SYMBOL", "KATAKANA"});

        assertCharacterCategories(characterDefinition, '・', "KATAKANA", "SYMBOL");
        assertCharacterCategories(characterDefinition, '・', "SYMBOL", "KATAKANA");
    }

    public void assertCharacterCategories(CharacterDefinitions characterDefinition, char c, String... categories) {
        int[] categoryIds = characterDefinition.lookupCategories(c);

        if (categoryIds == null) {
            assertNull(categories);
            return;
        }

        assertEquals(categories.length, categoryIds.length);

        List<String> categoryList = Arrays.asList(categories);

        for (int categoryId : categoryIds) {
            String category = categoryIdMap.get(categoryId);
            assertTrue(categoryList.contains(category));
        }
    }

    private static Map<Integer, String> invert(Map<String, Integer> map) {
        Map<Integer, String> inverted = new HashMap<>();

        for (String key : map.keySet()) {
            inverted.put(map.get(key), key);
        }

        return inverted;
    }
}
