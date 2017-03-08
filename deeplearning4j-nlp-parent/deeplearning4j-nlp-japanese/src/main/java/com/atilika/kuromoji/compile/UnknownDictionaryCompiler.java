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

import com.atilika.kuromoji.dict.GenericDictionaryEntry;
import com.atilika.kuromoji.io.IntegerArrayIO;
import com.atilika.kuromoji.io.StringArrayIO;
import com.atilika.kuromoji.util.UnknownDictionaryEntryParser;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class UnknownDictionaryCompiler implements Compiler {

    private OutputStream output;

    protected Map<String, Integer> categoryMap;

    protected List<GenericDictionaryEntry> dictionaryEntries = new ArrayList<>();

    public UnknownDictionaryCompiler(Map<String, Integer> categoryMap, OutputStream output) {
        this.categoryMap = categoryMap;
        this.output = output;
    }

    public void readUnknownDefinition(InputStream input, String encoding) throws IOException {
        LineNumberReader reader = new LineNumberReader(new InputStreamReader(input, encoding));

        UnknownDictionaryEntryParser parser = new UnknownDictionaryEntryParser();
        String line;

        while ((line = reader.readLine()) != null) {
            GenericDictionaryEntry entry = parser.parse(line);

            dictionaryEntries.add(entry);
        }
    }

    public int[][] makeCosts() {
        int[][] costs = new int[dictionaryEntries.size()][];

        for (int i = 0; i < dictionaryEntries.size(); i++) {
            GenericDictionaryEntry entry = dictionaryEntries.get(i);

            costs[i] = new int[] {entry.getLeftId(), entry.getRightId(), entry.getWordCost()};
        }

        return costs;
    }

    public String[][] makeFeatures() {
        String[][] features = new String[dictionaryEntries.size()][];

        for (int i = 0; i < dictionaryEntries.size(); i++) {
            GenericDictionaryEntry entry = dictionaryEntries.get(i);

            List<String> tmp = new ArrayList<>();
            tmp.addAll(entry.getPosFeatures());
            tmp.addAll(entry.getFeatures());

            String[] f = new String[tmp.size()];
            features[i] = tmp.toArray(f);
        }

        return features;
    }

    public int[][] makeCategoryReferences() {
        int[][] entries = new int[categoryMap.size()][];

        for (String category : categoryMap.keySet()) {
            int categoryId = categoryMap.get(category);

            entries[categoryId] = getEntryIndices(category);
        }

        return entries;
    }

    public void printFeatures(String[][] features) {
        for (int i = 0; i < features.length; i++) {
            System.out.println(i);

            String[] array = features[i];

            for (int j = 0; j < array.length; j++) {
                System.out.println("\t" + array[j]);
            }

        }
    }

    public int[] getEntryIndices(String surface) {
        List<Integer> indices = new ArrayList<>();

        for (int i = 0; i < dictionaryEntries.size(); i++) {
            GenericDictionaryEntry entry = dictionaryEntries.get(i);

            if (entry.getSurface().equals(surface)) {
                indices.add(i);
            }
        }

        return toArray(indices);
    }

    private int[] toArray(List<Integer> list) {
        int[] array = new int[list.size()];

        for (int i = 0; i < list.size(); i++) {
            array[i] = list.get(i);
        }

        return array;
    }

    public List<GenericDictionaryEntry> getDictionaryEntries() {
        return dictionaryEntries;
    }

    @Override
    public void compile() throws IOException {
        IntegerArrayIO.writeArray2D(output, makeCosts());
        IntegerArrayIO.writeArray2D(output, makeCategoryReferences());
        StringArrayIO.writeArray2D(output, makeFeatures());
        output.close();
    }
}
