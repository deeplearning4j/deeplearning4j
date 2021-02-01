/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
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
