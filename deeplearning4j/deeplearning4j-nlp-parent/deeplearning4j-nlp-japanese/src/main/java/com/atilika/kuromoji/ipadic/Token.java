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
package com.atilika.kuromoji.ipadic;

import com.atilika.kuromoji.TokenBase;
import com.atilika.kuromoji.dict.Dictionary;
import com.atilika.kuromoji.ipadic.compile.DictionaryEntry;
import com.atilika.kuromoji.viterbi.ViterbiNode;

/**
 * IPADIC token produced by the IPADIC tokenizer with various morphological features
 */
public class Token extends TokenBase {

    public Token(int wordId, String surface, ViterbiNode.Type type, int position, Dictionary dictionary) {
        super(wordId, surface, type, position, dictionary);
    }

    /**
     * Gets the 1st level part-of-speech tag for this token (品詞細分類1)
     *
     * @return 1st level part-of-speech tag, not null
     */
    public String getPartOfSpeechLevel1() {
        return this.getFeature(DictionaryEntry.PART_OF_SPEECH_LEVEL_1);
    }

    /**
     * Gets the 2nd level part-of-speech tag for this token (品詞細分類2)
     *
     * @return 2nd level part-of-speech tag, not null
     */
    public String getPartOfSpeechLevel2() {
        return this.getFeature(DictionaryEntry.PART_OF_SPEECH_LEVEL_2);
    }

    /**
     * Gets the 3rd level part-of-speech tag for this token (品詞細分類3)
     *
     * @return 3rd level part-of-speech tag, not null
     */
    public String getPartOfSpeechLevel3() {
        return this.getFeature(DictionaryEntry.PART_OF_SPEECH_LEVEL_3);
    }

    /**
     * Gets the 4th level part-of-speech tag for this token (品詞細分類4)
     *
     * @return 4th level part-of-speech tag, not null
     */
    public String getPartOfSpeechLevel4() {
        return this.getFeature(DictionaryEntry.PART_OF_SPEECH_LEVEL_4);
    }

    /**
     * Gets the conjugation type for this token (活用型), if applicable
     * <p>
     * If this token does not have a conjugation type, return *
     *
     * @return conjugation type, not null
     */
    public String getConjugationType() {
        return this.getFeature(DictionaryEntry.CONJUGATION_TYPE);
    }

    /**
     * Gets the conjugation form for this token (活用形), if applicable
     * <p>
     * If this token does not have a conjugation form, return *
     *
     * @return conjugation form, not null
     */
    public String getConjugationForm() {
        return this.getFeature(DictionaryEntry.CONJUGATION_FORM);
    }

    /**
     * Gets the base form (also called dictionary form) for this token (基本形)
     *
     * @return base form, not null
     */
    public String getBaseForm() {
        return this.getFeature(DictionaryEntry.BASE_FORM);
    }

    /**
     * Gets the reading for this token (読み) in katakana script
     *
     * @return reading, not null
     */
    public String getReading() {
        return this.getFeature(DictionaryEntry.READING);
    }

    /**
     * Gets the pronunciation for this token (発音)
     *
     * @return pronunciation, not null
     */
    public String getPronunciation() {
        return this.getFeature(DictionaryEntry.PRONUNCIATION);
    }
}
