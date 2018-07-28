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
import com.atilika.kuromoji.util.StringUtils;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;

public class UnknownDictionary implements Dictionary {

    //    public static final String UNKNOWN_DICTIONARY_FILENAME = "unknownDictionary.bin";
    public static final String UNKNOWN_DICTIONARY_FILENAME = new File(KuromojiBinFilesFetcher.getKuromojiRoot(),"unknownDictionary.bin").getAbsolutePath();

    private static final String DEFAULT_FEATURE = "*";

    private static final String FEATURE_SEPARATOR = ",";

    private final int[][] entries;

    private final int[][] costs;

    private final String[][] features;

    private final int totalFeatures;

    private final CharacterDefinitions characterDefinition;

    public UnknownDictionary(CharacterDefinitions characterDefinition, int[][] entries, int[][] costs,
                    String[][] features, int totalFeatures) {
        this.characterDefinition = characterDefinition;
        this.entries = entries;
        this.costs = costs;
        this.features = features;
        this.totalFeatures = totalFeatures;
    }

    public UnknownDictionary(CharacterDefinitions characterDefinition, int[][] entries, int[][] costs,
                    String[][] features) {
        this(characterDefinition, entries, costs, features, features.length);
    }


    public int[] lookupWordIds(int categoryId) {
        // Returns an array of word ids
        return entries[categoryId];
    }

    @Override
    public int getLeftId(int wordId) {
        return costs[wordId][0];
    }

    @Override
    public int getRightId(int wordId) {
        return costs[wordId][1];
    }

    @Override
    public int getWordCost(int wordId) {
        return costs[wordId][2];
    }

    @Override
    public String getAllFeatures(int wordId) {
        return StringUtils.join(getAllFeaturesArray(wordId), FEATURE_SEPARATOR);
    }

    @Override
    public String[] getAllFeaturesArray(int wordId) {
        if (totalFeatures == features.length) {
            return features[wordId];
        }

        String[] allFeatures = new String[totalFeatures];
        String[] basicFeatures = features[wordId];

        for (int i = 0; i < basicFeatures.length; i++) {
            allFeatures[i] = basicFeatures[i];
        }

        for (int i = basicFeatures.length; i < totalFeatures; i++) {
            allFeatures[i] = DEFAULT_FEATURE;
        }

        return allFeatures;
    }

    @Override
    public String getFeature(int wordId, int... fields) {
        String[] allFeatures = getAllFeaturesArray(wordId);
        String[] features = new String[fields.length];

        for (int i = 0; i < fields.length; i++) {
            int featureNumber = fields[i];
            features[i] = allFeatures[featureNumber];
        }

        return StringUtils.join(features, FEATURE_SEPARATOR);
    }

    public CharacterDefinitions getCharacterDefinition() {
        return characterDefinition;
    }

    public static UnknownDictionary newInstance(ResourceResolver resolver, CharacterDefinitions characterDefinitions,
                    int totalFeatures) throws IOException {
        InputStream unkDefInput = resolver.resolve(UnknownDictionary.UNKNOWN_DICTIONARY_FILENAME);

        int[][] costs = IntegerArrayIO.readArray2D(unkDefInput);
        int[][] references = IntegerArrayIO.readArray2D(unkDefInput);
        String[][] features = StringArrayIO.readArray2D(unkDefInput);

        UnknownDictionary unknownDictionary =
                        new UnknownDictionary(characterDefinitions, references, costs, features, totalFeatures);

        return unknownDictionary;
    }
}
