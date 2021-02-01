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
package com.atilika.kuromoji.ipadic.compile;

import com.atilika.kuromoji.compile.TokenInfoDictionaryCompilerBase;
import com.atilika.kuromoji.dict.GenericDictionaryEntry;
import com.atilika.kuromoji.util.DictionaryEntryLineParser;

import java.util.ArrayList;
import java.util.List;

public class TokenInfoDictionaryCompiler extends TokenInfoDictionaryCompilerBase<DictionaryEntry> {

    public TokenInfoDictionaryCompiler(String encoding) {
        super(encoding);
    }

    @Override
    protected DictionaryEntry parse(String line) {
        String[] fields = DictionaryEntryLineParser.parseLine(line);
        DictionaryEntry entry = new DictionaryEntry(fields);
        return entry;
    }

    @Override
    protected GenericDictionaryEntry generateGenericDictionaryEntry(DictionaryEntry entry) {
        List<String> pos = extractPosFeatures(entry);
        List<String> features = extractOtherFeatures(entry);

        return new GenericDictionaryEntry.Builder().surface(entry.getSurface()).leftId(entry.getLeftId())
                        .rightId(entry.getRightId()).wordCost(entry.getWordCost()).pos(pos).features(features).build();
    }

    public List<String> extractPosFeatures(DictionaryEntry entry) {
        List<String> posFeatures = new ArrayList<>();

        posFeatures.add(entry.getPartOfSpeechLevel1());
        posFeatures.add(entry.getPartOfSpeechLevel2());
        posFeatures.add(entry.getPartOfSpeechLevel3());
        posFeatures.add(entry.getPartOfSpeechLevel4());

        posFeatures.add(entry.getConjugationType());
        posFeatures.add(entry.getConjugatedForm());

        return posFeatures;
    }

    public List<String> extractOtherFeatures(DictionaryEntry entry) {
        List<String> otherFeatures = new ArrayList<>();

        otherFeatures.add(entry.getBaseForm());
        otherFeatures.add(entry.getReading());
        otherFeatures.add(entry.getPronunciation());

        return otherFeatures;
    }
}
