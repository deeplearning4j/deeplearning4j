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

import com.atilika.kuromoji.buffer.BufferEntry;
import com.atilika.kuromoji.io.ByteBufferIO;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.util.List;

public class TokenInfoBufferCompiler implements Compiler {

    private static final int INTEGER_BYTES = Integer.SIZE / Byte.SIZE;
    private static final int SHORT_BYTES = Short.SIZE / Byte.SIZE;

    private ByteBuffer buffer;

    private OutputStream output;

    public TokenInfoBufferCompiler(OutputStream output, List<BufferEntry> entries) {
        this.output = output;
        putEntries(entries);
    }

    public void putEntries(List<BufferEntry> entries) {
        int size = calculateEntriesSize(entries) * 2;

        this.buffer = ByteBuffer.allocate(size + INTEGER_BYTES * 4);

        buffer.putInt(size);
        buffer.putInt(entries.size());
        BufferEntry firstEntry = entries.get(0);

        buffer.putInt(firstEntry.tokenInfo.size());
        buffer.putInt(firstEntry.posInfo.size());
        buffer.putInt(firstEntry.features.size());

        for (BufferEntry entry : entries) {
            for (Short s : entry.tokenInfo) {
                buffer.putShort(s);
            }

            for (Byte b : entry.posInfo) {
                buffer.put(b);
            }

            for (Integer feature : entry.features) {
                buffer.putInt(feature);
            }
        }
    }

    private int calculateEntriesSize(List<BufferEntry> entries) {
        if (entries.isEmpty()) {
            return 0;
        } else {
            int size = 0;
            BufferEntry entry = entries.get(0);
            size += entry.tokenInfo.size() * SHORT_BYTES + SHORT_BYTES;
            size += entry.posInfo.size();
            size += entry.features.size() * INTEGER_BYTES;
            size *= entries.size();
            return size;
        }
    }

    @Override
    public void compile() throws IOException {
        ByteBufferIO.write(output, buffer);
        output.close();
    }
}
