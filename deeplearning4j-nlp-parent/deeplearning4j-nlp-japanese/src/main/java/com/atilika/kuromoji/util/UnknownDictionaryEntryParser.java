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
package com.atilika.kuromoji.util;

import com.atilika.kuromoji.dict.GenericDictionaryEntry;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class UnknownDictionaryEntryParser extends DictionaryEntryLineParser {

    // NOTE: Currently this code is the same as the IPADIC dictionary entry parser,
    // which is okay for all the dictionaries supported so far...
    public GenericDictionaryEntry parse(String entry) {
        String[] fields = parseLine(entry);

        String surface = fields[0];
        short leftId = Short.parseShort(fields[1]);
        short rightId = Short.parseShort(fields[2]);
        short wordCost = Short.parseShort(fields[3]);

        List<String> pos = new ArrayList<>();
        pos.addAll(Arrays.asList(fields).subList(4, 10));

        List<String> features = new ArrayList<>();
        features.addAll(Arrays.asList(fields).subList(10, fields.length));

        GenericDictionaryEntry dictionaryEntry = new GenericDictionaryEntry.Builder().surface(surface).leftId(leftId)
                        .rightId(rightId).wordCost(wordCost).pos(pos).features(features).build();

        return dictionaryEntry;
    }
}
