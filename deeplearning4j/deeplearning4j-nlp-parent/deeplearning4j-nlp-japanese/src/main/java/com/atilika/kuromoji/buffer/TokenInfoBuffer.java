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

import com.atilika.kuromoji.io.ByteBufferIO;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;

public class TokenInfoBuffer {

    private static final int INTEGER_BYTES = Integer.SIZE / Byte.SIZE;
    private static final int SHORT_BYTES = Short.SIZE / Byte.SIZE;

    private ByteBuffer buffer;

    private final int tokenInfoCount;
    private final int posInfoCount;
    private final int featureCount;

    private final int entrySize;

    public TokenInfoBuffer(InputStream is) throws IOException {
        buffer = ByteBufferIO.read(is);
        tokenInfoCount = getTokenInfoCount();
        posInfoCount = getPosInfoCount();
        featureCount = getFeatureCount();
        entrySize = getEntrySize(tokenInfoCount, posInfoCount, featureCount);
    }

    public BufferEntry lookupEntry(int offset) {
        BufferEntry entry = new BufferEntry();

        entry.tokenInfos = new short[tokenInfoCount];
        entry.posInfos = new byte[posInfoCount];
        entry.featureInfos = new int[featureCount];

        int entrySize = getEntrySize(tokenInfoCount, posInfoCount, featureCount);
        int position = getPosition(offset, entrySize);

        // Get left id, right id and word cost
        for (int i = 0; i < tokenInfoCount; i++) {
            entry.tokenInfos[i] = buffer.getShort(position + i * SHORT_BYTES);
        }

        // Get part of speech tags values (not strings yet)
        for (int i = 0; i < posInfoCount; i++) {
            entry.posInfos[i] = buffer.get(position + tokenInfoCount * SHORT_BYTES + i);
        }

        // Get field value references (string references)
        for (int i = 0; i < featureCount; i++) {
            entry.featureInfos[i] =
                            buffer.getInt(position + tokenInfoCount * SHORT_BYTES + posInfoCount + i * INTEGER_BYTES);
        }

        return entry;
    }

    public int lookupTokenInfo(int offset, int i) {
        int position = getPosition(offset, entrySize);
        return buffer.getShort(position + i * SHORT_BYTES);
    }

    public int lookupPartOfSpeechFeature(int offset, int i) {
        int position = getPosition(offset, entrySize);

        return 0xff & buffer.get(position + tokenInfoCount * SHORT_BYTES + i);
    }

    public int lookupFeature(int offset, int i) {
        int position = getPosition(offset, entrySize);

        return buffer.getInt(
                        position + tokenInfoCount * SHORT_BYTES + posInfoCount + (i - posInfoCount) * INTEGER_BYTES);
    }

    public boolean isPartOfSpeechFeature(int i) {
        int posInfoCount = getPosInfoCount();
        return (i < posInfoCount);
    }

    private int getTokenInfoCount() {
        return buffer.getInt(INTEGER_BYTES * 2);
    }

    private int getPosInfoCount() {
        return buffer.getInt(INTEGER_BYTES * 3);
    }

    private int getFeatureCount() {
        return buffer.getInt(INTEGER_BYTES * 4);
    }

    private int getEntrySize(int tokenInfoCount, int posInfoCount, int featureCount) {
        return tokenInfoCount * SHORT_BYTES + posInfoCount + featureCount * INTEGER_BYTES;
    }

    private int getPosition(int offset, int entrySize) {
        return offset * entrySize + INTEGER_BYTES * 5;
    }
}
