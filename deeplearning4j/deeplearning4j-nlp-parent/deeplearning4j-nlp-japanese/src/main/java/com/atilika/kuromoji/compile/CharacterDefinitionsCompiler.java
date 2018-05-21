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

import com.atilika.kuromoji.io.IntegerArrayIO;
import com.atilika.kuromoji.io.StringArrayIO;

import java.io.*;
import java.util.*;

public class CharacterDefinitionsCompiler implements Compiler {

    private Map<String, int[]> categoryDefinitions = new TreeMap<>();

    @SuppressWarnings("unchecked")
    private List<Set<String>> codepointCategories = new ArrayList<>(new TreeSet());

    private OutputStream output;

    public CharacterDefinitionsCompiler(OutputStream output) {
        this.output = output;

        for (int i = 0; i < 65536; i++) {
            codepointCategories.add(null);
        }
    }

    public void readCharacterDefinition(InputStream stream, String encoding) throws IOException {
        LineNumberReader reader = new LineNumberReader(new InputStreamReader(stream, encoding));

        String line;

        while ((line = reader.readLine()) != null) {
            // Strip comments
            line = line.replaceAll("\\s*#.*", "");

            // Skip empty line or comment line
            if (line.isEmpty()) {
                continue;
            }

            if (isCategoryEntry(line)) {
                parseCategory(line);
            } else {
                parseMapping(line);
            }
        }
    }

    private void parseCategory(String line) {
        String[] values = line.split("\\s+");

        String classname = values[0];
        int invoke = Integer.parseInt(values[1]);
        int group = Integer.parseInt(values[2]);
        int length = Integer.parseInt(values[3]);

        assert !categoryDefinitions.containsKey(classname);

        categoryDefinitions.put(classname, new int[] {invoke, group, length});
    }

    private void parseMapping(String line) {
        String[] values = line.split("\\s+");

        assert values.length >= 2;

        String codepointString = values[0];
        List<String> categories = getCategories(values);

        if (codepointString.contains("..")) {
            String[] codepoints = codepointString.split("\\.\\.");

            int lowerCodepoint = Integer.decode(codepoints[0]);
            int upperCodepoint = Integer.decode(codepoints[1]);

            for (int i = lowerCodepoint; i <= upperCodepoint; i++) {
                addMapping(i, categories);
            }

        } else {
            int codepoint = Integer.decode(codepointString);

            addMapping(codepoint, categories);
        }
    }

    private List<String> getCategories(String[] values) {
        return Arrays.asList(values).subList(1, values.length);
    }

    private void addMapping(int codepoint, List<String> categories) {
        for (String category : categories) {
            addMapping(codepoint, category);
        }
    }

    private void addMapping(int codepoint, String category) {
        Set<String> categories = codepointCategories.get(codepoint);

        if (categories == null) {
            categories = new TreeSet<>();
            codepointCategories.set(codepoint, categories);
        }

        categories.add(category);
    }

    private boolean isCategoryEntry(String line) {
        return !line.startsWith("0x");
    }

    public Map<String, Integer> makeCharacterCategoryMap() {
        Map<String, Integer> classMapping = new TreeMap<>();
        int i = 0;

        for (String category : categoryDefinitions.keySet()) {
            classMapping.put(category, i++);
        }
        return classMapping;
    }

    private int[][] makeCharacterDefinitions() {
        Map<String, Integer> categoryMap = makeCharacterCategoryMap();
        int size = categoryMap.size();
        int[][] array = new int[size][];

        for (String category : categoryDefinitions.keySet()) {
            int[] values = categoryDefinitions.get(category);

            assert values.length == 3;

            int index = categoryMap.get(category);
            array[index] = values;
        }

        return array;
    }

    private int[][] makeCharacterMappings() {
        Map<String, Integer> categoryMap = makeCharacterCategoryMap();

        int size = codepointCategories.size();
        int[][] array = new int[size][];

        for (int i = 0; i < size; i++) {
            Set<String> categories = codepointCategories.get(i);

            if (categories != null) {
                int innerSize = categories.size();
                int[] inner = new int[innerSize];

                int j = 0;

                for (String value : categories) {
                    inner[j++] = categoryMap.get(value);
                }
                array[i] = inner;
            }
        }

        return array;
    }

    private String[] makeCharacterCategorySymbols() {
        Map<String, Integer> categoryMap = makeCharacterCategoryMap();
        Map<Integer, String> inverted = new TreeMap<>();

        for (String key : categoryMap.keySet()) {
            inverted.put(categoryMap.get(key), key);
        }

        String[] categories = new String[inverted.size()];

        for (Integer index : inverted.keySet()) {
            categories[index] = inverted.get(index);
        }

        return categories;
    }

    public Map<String, int[]> getCategoryDefinitions() {
        return categoryDefinitions;
    }

    public List<Set<String>> getCodepointCategories() {
        return codepointCategories;
    }

    @Override
    public void compile() throws IOException {
        IntegerArrayIO.writeSparseArray2D(output, makeCharacterDefinitions());
        IntegerArrayIO.writeSparseArray2D(output, makeCharacterMappings());
        StringArrayIO.writeArray(output, makeCharacterCategorySymbols());
        output.close();
    }
}
