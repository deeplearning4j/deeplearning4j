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

import com.atilika.kuromoji.io.IntegerArrayIO;

import java.io.IOException;
import java.io.OutputStream;

public class WordIdMapCompiler implements Compiler {

    private int[][] wordIds = new int[1][];

    private int[] indices;

    private GrowableIntArray wordIdArray = new GrowableIntArray();

    public void addMapping(int sourceId, int wordId) {
        if (wordIds.length <= sourceId) {
            int[][] newArray = new int[sourceId + 1][];
            System.arraycopy(wordIds, 0, newArray, 0, wordIds.length);
            wordIds = newArray;
        }

        // Prepare array -- extend the length of array by one
        int[] current = wordIds[sourceId];
        if (current == null) {
            current = new int[1];
        } else {
            int[] newArray = new int[current.length + 1];
            System.arraycopy(current, 0, newArray, 0, current.length);
            current = newArray;
        }
        wordIds[sourceId] = current;

        int[] targets = wordIds[sourceId];
        targets[targets.length - 1] = wordId;
    }

    public void write(OutputStream output) throws IOException {
        compile();
        IntegerArrayIO.writeArray(output, indices);
        IntegerArrayIO.writeArray(output, wordIdArray.getArray());
    }

    public void compile() {
        this.indices = new int[wordIds.length];
        int wordIdIndex = 0;

        for (int i = 0; i < wordIds.length; i++) {
            int[] inner = wordIds[i];

            if (inner == null) {
                indices[i] = -1;
            } else {
                indices[i] = wordIdIndex;
                wordIdArray.set(wordIdIndex++, inner.length);

                for (int j = 0; j < inner.length; j++) {
                    wordIdArray.set(wordIdIndex++, inner[j]);
                }
            }
        }
    }

    public static class GrowableIntArray {

        private static final float ARRAY_GROWTH_RATE = 1.25f;

        private static final int ARRAY_INITIAL_SIZE = 1024;

        private int maxIndex;

        private int[] array;

        public GrowableIntArray(int size) {
            this.array = new int[size];
            this.maxIndex = 0;
        }

        public GrowableIntArray() {
            this(ARRAY_INITIAL_SIZE);
        }

        public int[] getArray() {
            int length = maxIndex + 1;
            int[] a = new int[length];
            System.arraycopy(array, 0, a, 0, length);
            return a;
        }

        public void set(int index, int value) {
            if (index >= array.length) {
                grow(getNewLength(index));
            }

            if (index > maxIndex) {
                maxIndex = index;
            }

            array[index] = value;
        }

        private void grow(int newLength) {
            int[] tmp = new int[newLength];
            System.arraycopy(array, 0, tmp, 0, maxIndex + 1);
            array = tmp;
        }

        private int getNewLength(int index) {
            return (int) Math.max(index + 1, array.length * ARRAY_GROWTH_RATE);
        }
    }
}
