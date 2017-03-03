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
package com.atilika.kuromoji.buffer;

import com.atilika.kuromoji.io.IntegerArrayIO;

import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;

public class WordIdMap {

    private final int[] indices;

    private final int[] wordIds;

    private final int[] empty = new int[] {};

    public WordIdMap(InputStream input) throws IOException {
        indices = IntegerArrayIO.readArray(input);
        wordIds = IntegerArrayIO.readArray(input);
    }

    public int[] lookUp(int sourceId) {
        int index = indices[sourceId];

        if (index == -1) {
            return empty;
        }

        return Arrays.copyOfRange(wordIds, index + 1, index + 1 + wordIds[index]);
    }
}
