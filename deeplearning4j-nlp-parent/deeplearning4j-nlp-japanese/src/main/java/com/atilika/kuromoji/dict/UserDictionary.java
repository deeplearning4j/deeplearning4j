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

import com.atilika.kuromoji.trie.PatriciaTrie;
import com.atilika.kuromoji.util.DictionaryEntryLineParser;
import com.atilika.kuromoji.util.StringUtils;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class UserDictionary implements Dictionary {

    private static final String DEFAULT_FEATURE = "*";

    private static final String FEATURE_SEPARATOR = ",";

    private static final int CUSTOM_DICTIONARY_WORD_ID_OFFSET = 100000000;

    private static final int WORD_COST = -100000;

    private static final int LEFT_ID = 5;

    private static final int RIGHT_ID = 5;

    private int wordId = CUSTOM_DICTIONARY_WORD_ID_OFFSET;

    // The word id below is the word id for the source string
    // surface string => [ word id, 1st token length, 2nd token length, ... , nth token length
    private PatriciaTrie<int[]> entries = new PatriciaTrie<>();

    // Maps wordId to reading
    private Map<Integer, String> readings = new HashMap<>();

    // Maps wordId to part-of-speech
    private Map<Integer, String> partOfSpeech = new HashMap<>();

    private final int readingFeature;

    private final int partOfSpeechFeature;

    private final int totalFeatures;

    public UserDictionary(InputStream inputStream, int totalFeatures, int readingFeature, int partOfSpeechFeature)
                    throws IOException {
        this.totalFeatures = totalFeatures;
        this.readingFeature = readingFeature;
        this.partOfSpeechFeature = partOfSpeechFeature;
        read(inputStream);
    }

    /**
     * Lookup words in text
     *
     * @param text  text to look up user dictionary matches for
     * @return list of UserDictionaryMatch, not null
     */
    public List<UserDictionaryMatch> findUserDictionaryMatches(String text) {
        List<UserDictionaryMatch> matchInfos = new ArrayList<>();
        int startIndex = 0;

        while (startIndex < text.length()) {
            int matchLength = 0;

            while (startIndex + matchLength < text.length()
                            && entries.containsKeyPrefix(text.substring(startIndex, startIndex + matchLength + 1))) {
                matchLength++;
            }

            if (matchLength > 0) {
                String match = text.substring(startIndex, startIndex + matchLength);
                int[] details = entries.get(match);

                if (details != null) {
                    matchInfos.addAll(makeMatchDetails(startIndex, details));
                }
            }

            startIndex++;
        }

        return matchInfos;
    }

    private List<UserDictionaryMatch> makeMatchDetails(int matchStartIndex, int[] details) {
        List<UserDictionaryMatch> matchDetails = new ArrayList<>(details.length - 1);

        int wordId = details[0];
        int startIndex = 0;

        for (int i = 1; i < details.length; i++) {
            int matchLength = details[i];

            matchDetails.add(new UserDictionaryMatch(wordId, matchStartIndex + startIndex, matchLength));

            startIndex += matchLength;
            wordId++;
        }
        return matchDetails;
    }

    public static class UserDictionaryMatch {

        private final int wordId;

        private final int matchStartIndex;

        private final int matchLength;

        public UserDictionaryMatch(int wordId, int matchStartIndex, int matchLength) {
            this.wordId = wordId;
            this.matchStartIndex = matchStartIndex;
            this.matchLength = matchLength;
        }

        public int getWordId() {
            return wordId;
        }

        public int getMatchStartIndex() {
            return matchStartIndex;
        }

        public int getMatchLength() {
            return matchLength;
        }
    }

    @Override
    public int getLeftId(int wordId) {
        return LEFT_ID;
    }

    @Override
    public int getRightId(int wordId) {
        return RIGHT_ID;
    }

    @Override
    public int getWordCost(int wordId) {
        return WORD_COST;
    }

    @Override
    public String[] getAllFeaturesArray(int wordId) {
        String[] features = new String[totalFeatures];

        for (int i = 0; i < totalFeatures; i++) {
            features[i] = getFeature(wordId, i);
        }

        return features;
    }

    @Override
    public String getAllFeatures(int wordId) {
        return StringUtils.join(getAllFeaturesArray(wordId), FEATURE_SEPARATOR);
    }

    @Override
    public String getFeature(int wordId, int... fields) {

        // Is this latter test correct?  There can be duplicate features... -Christian
        if (fields.length == 0 || fields.length == totalFeatures) {
            return getAllFeatures(wordId);
        }

        String[] features = new String[fields.length];

        for (int i = 0; i < fields.length; i++) {

            int featureNumber = fields[i];

            if (featureNumber == readingFeature) {
                features[i] = readings.get(wordId);
            } else if (featureNumber == partOfSpeechFeature) {
                features[i] = partOfSpeech.get(wordId);
            } else {
                features[i] = DEFAULT_FEATURE;
            }
        }

        return StringUtils.join(features, FEATURE_SEPARATOR);
    }

    public void read(InputStream input) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(input, StandardCharsets.UTF_8));
        String line;

        while ((line = reader.readLine()) != null) {
            // Remove comments and trim leading and trailing whitespace
            line = line.replaceAll("#.*$", "");
            line = line.trim();

            // Skip empty lines or comment lines
            if (line.isEmpty()) {
                continue;
            }

            addEntry(line);
        }
    }

    public void addEntry(String entry) {
        String[] values = DictionaryEntryLineParser.parseLine(entry);

        String surface = values[0];
        String segmentationValue = values[1];
        String readingsValue = values[2];
        String partOfSpeech = values[3];

        String[] segmentation;
        String[] readings;

        if (isCustomSegmentation(surface, segmentationValue)) {
            segmentation = split(segmentationValue);
            readings = split(readingsValue);
        } else {
            segmentation = new String[] {segmentationValue};
            readings = new String[] {readingsValue};
        }

        if (segmentation.length != readings.length) {
            throw new RuntimeException("User dictionary entry not properly formatted: " + entry);
        }

        // { wordId, 1st token length, 2nd token length, ... , nth token length
        int[] wordIdAndLengths = new int[segmentation.length + 1];

        wordIdAndLengths[0] = wordId;

        for (int i = 0; i < segmentation.length; i++) {
            wordIdAndLengths[i + 1] = segmentation[i].length();

            this.readings.put(wordId, readings[i]);
            this.partOfSpeech.put(wordId, partOfSpeech);

            wordId++;
        }

        entries.put(surface, wordIdAndLengths);
    }

    private boolean isCustomSegmentation(String surface, String segmentation) {
        return !surface.equals(segmentation);
    }

    private String[] split(String input) {
        return input.split("\\s+");
    }
}
