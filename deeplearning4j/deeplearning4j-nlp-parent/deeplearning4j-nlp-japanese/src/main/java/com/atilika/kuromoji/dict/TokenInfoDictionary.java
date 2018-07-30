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

import com.atilika.kuromoji.buffer.BufferEntry;
import com.atilika.kuromoji.buffer.StringValueMapBuffer;
import com.atilika.kuromoji.buffer.TokenInfoBuffer;
import com.atilika.kuromoji.buffer.WordIdMap;
import com.atilika.kuromoji.util.DictionaryEntryLineParser;
import com.atilika.kuromoji.util.KuromojiBinFilesFetcher;
import com.atilika.kuromoji.util.ResourceResolver;
import com.atilika.kuromoji.util.StringUtils;

import java.io.File;
import java.io.IOException;

public class TokenInfoDictionary implements Dictionary {

    public static final String TOKEN_INFO_DICTIONARY_FILENAME = new File(KuromojiBinFilesFetcher.getKuromojiRoot(), "tokenInfoDictionary.bin").getAbsolutePath();
    public static final String FEATURE_MAP_FILENAME = new File(KuromojiBinFilesFetcher.getKuromojiRoot(),"tokenInfoFeaturesMap.bin").getAbsolutePath();
    public static final String POS_MAP_FILENAME = new File(KuromojiBinFilesFetcher.getKuromojiRoot(),"tokenInfoPartOfSpeechMap.bin").getAbsolutePath();
    public static final String TARGETMAP_FILENAME = new File(KuromojiBinFilesFetcher.getKuromojiRoot(),"tokenInfoTargetMap.bin").getAbsolutePath();

    private static final int LEFT_ID = 0;
    private static final int RIGHT_ID = 1;
    private static final int WORD_COST = 2;
    private static final int TOKEN_INFO_OFFSET = 3;

    private static final String FEATURE_SEPARATOR = ",";

    protected TokenInfoBuffer tokenInfoBuffer;
    protected StringValueMapBuffer posValues;
    protected StringValueMapBuffer stringValues;
    protected WordIdMap wordIdMap;

    public int[] lookupWordIds(int sourceId) {
        return wordIdMap.lookUp(sourceId);
    }

    @Override
    public int getLeftId(int wordId) {
        return tokenInfoBuffer.lookupTokenInfo(wordId, LEFT_ID);
    }

    @Override
    public int getRightId(int wordId) {
        return tokenInfoBuffer.lookupTokenInfo(wordId, RIGHT_ID);
    }

    @Override
    public int getWordCost(int wordId) {
        return tokenInfoBuffer.lookupTokenInfo(wordId, WORD_COST);
    }

    @Override
    public String[] getAllFeaturesArray(int wordId) {
        BufferEntry bufferEntry = tokenInfoBuffer.lookupEntry(wordId);

        int posLength = bufferEntry.posInfos.length;
        int featureLength = bufferEntry.featureInfos.length;

        boolean partOfSpeechAsShorts = false;

        if (posLength == 0) {
            posLength = bufferEntry.tokenInfos.length - TOKEN_INFO_OFFSET;
            partOfSpeechAsShorts = true;
        }

        String[] result = new String[posLength + featureLength];

        if (partOfSpeechAsShorts) {
            for (int i = 0; i < posLength; i++) {
                int feature = bufferEntry.tokenInfos[i + TOKEN_INFO_OFFSET];
                result[i] = posValues.get(feature);
            }
        } else {
            for (int i = 0; i < posLength; i++) {
                int feature = bufferEntry.posInfos[i] & 0xff;
                result[i] = posValues.get(feature);
            }
        }

        for (int i = 0; i < featureLength; i++) {
            int feature = bufferEntry.featureInfos[i];
            String s = stringValues.get(feature);
            result[i + posLength] = s;
        }

        return result;
    }

    @Override
    public String getAllFeatures(int wordId) {
        String[] features = getAllFeaturesArray(wordId);

        for (int i = 0; i < features.length; i++) {
            String feature = features[i];
            features[i] = DictionaryEntryLineParser.escape(feature);
        }

        return StringUtils.join(features, FEATURE_SEPARATOR);
    }

    @Override
    public String getFeature(int wordId, int... fields) {
        if (fields.length == 1) {
            return extractSingleFeature(wordId, fields[0]);
        }

        return extractMultipleFeatures(wordId, fields);
    }

    private String extractSingleFeature(int wordId, int field) {
        int featureId;

        if (tokenInfoBuffer.isPartOfSpeechFeature(field)) {
            featureId = tokenInfoBuffer.lookupPartOfSpeechFeature(wordId, field);
            return posValues.get(featureId);
        }

        featureId = tokenInfoBuffer.lookupFeature(wordId, field);
        return stringValues.get(featureId);
    }

    private String extractMultipleFeatures(int wordId, int[] fields) {
        if (fields.length == 0) {
            return getAllFeatures(wordId);
        }

        if (fields.length == 1) {
            return extractSingleFeature(wordId, fields[0]);
        }

        String[] allFeatures = getAllFeaturesArray(wordId);
        String[] features = new String[fields.length];

        for (int i = 0; i < fields.length; i++) {
            int featureNumber = fields[i];
            features[i] = DictionaryEntryLineParser.escape(allFeatures[featureNumber]);
        }
        return StringUtils.join(features, FEATURE_SEPARATOR);
    }

    public static TokenInfoDictionary newInstance(ResourceResolver resolver) throws IOException {
        TokenInfoDictionary dictionary = new TokenInfoDictionary();
        dictionary.setup(resolver);
        return dictionary;
    }

    private void setup(ResourceResolver resolver) throws IOException {
        tokenInfoBuffer = new TokenInfoBuffer(resolver.resolve(TOKEN_INFO_DICTIONARY_FILENAME));
        stringValues = new StringValueMapBuffer(resolver.resolve(FEATURE_MAP_FILENAME));
        posValues = new StringValueMapBuffer(resolver.resolve(POS_MAP_FILENAME));
        wordIdMap = new WordIdMap(resolver.resolve(TARGETMAP_FILENAME));
    }
}
