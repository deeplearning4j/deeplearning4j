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
package com.atilika.kuromoji.ipadic.compile;

import com.atilika.kuromoji.dict.DictionaryEntryBase;

import static com.atilika.kuromoji.dict.DictionaryField.*;

public class DictionaryEntry extends DictionaryEntryBase {
    public static final int PART_OF_SPEECH_LEVEL_1 = 4;
    public static final int PART_OF_SPEECH_LEVEL_2 = 5;
    public static final int PART_OF_SPEECH_LEVEL_3 = 6;
    public static final int PART_OF_SPEECH_LEVEL_4 = 7;
    public static final int CONJUGATION_TYPE = 8;
    public static final int CONJUGATION_FORM = 9;
    public static final int BASE_FORM = 10;
    public static final int READING = 11;
    public static final int PRONUNCIATION = 12;

    public static final int TOTAL_FEATURES = 9;
    public static final int READING_FEATURE = 7;
    public static final int PART_OF_SPEECH_FEATURE = 0;

    private final String posLevel1;
    private final String posLevel2;
    private final String posLevel3;
    private final String posLevel4;

    private final String conjugatedForm;
    private final String conjugationType;

    private final String baseForm;
    private final String reading;
    private final String pronunciation;

    public DictionaryEntry(String[] fields) {
        super(fields[SURFACE], Short.parseShort(fields[LEFT_ID]), Short.parseShort(fields[RIGHT_ID]),
                        Short.parseShort(fields[WORD_COST]));

        posLevel1 = fields[PART_OF_SPEECH_LEVEL_1];
        posLevel2 = fields[PART_OF_SPEECH_LEVEL_2];
        posLevel3 = fields[PART_OF_SPEECH_LEVEL_3];
        posLevel4 = fields[PART_OF_SPEECH_LEVEL_4];

        conjugationType = fields[CONJUGATION_TYPE];
        conjugatedForm = fields[CONJUGATION_FORM];

        baseForm = fields[BASE_FORM];
        reading = fields[READING];
        pronunciation = fields[PRONUNCIATION];
    }

    public String getPartOfSpeechLevel1() {
        return posLevel1;
    }

    public String getPartOfSpeechLevel2() {
        return posLevel2;
    }

    public String getPartOfSpeechLevel3() {
        return posLevel3;
    }

    public String getPartOfSpeechLevel4() {
        return posLevel4;
    }

    public String getConjugatedForm() {
        return conjugatedForm;
    }

    public String getConjugationType() {
        return conjugationType;
    }

    public String getBaseForm() {
        return baseForm;
    }

    public String getReading() {
        return reading;
    }

    public String getPronunciation() {
        return pronunciation;
    }
}
