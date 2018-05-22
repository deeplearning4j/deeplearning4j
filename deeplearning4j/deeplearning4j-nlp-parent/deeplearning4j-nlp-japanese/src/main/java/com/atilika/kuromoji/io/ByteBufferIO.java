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
package com.atilika.kuromoji.io;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.nio.channels.WritableByteChannel;

public class ByteBufferIO {

    public static ByteBuffer read(InputStream input) throws IOException {
        DataInputStream dataInput = new DataInputStream(new BufferedInputStream(input));

        int size = dataInput.readInt();
        ByteBuffer buffer = ByteBuffer.allocate(size);

        ReadableByteChannel channel = Channels.newChannel(dataInput);
        channel.read(buffer);

        buffer.rewind();
        return buffer;
    }

    public static void write(OutputStream output, ByteBuffer buffer) throws IOException {
        DataOutputStream dataOutput = new DataOutputStream(new BufferedOutputStream(output));

        dataOutput.writeInt(buffer.position());
        buffer.flip();

        WritableByteChannel channel = Channels.newChannel(dataOutput);
        channel.write(buffer);
        dataOutput.flush(); // TODO: Do we need this?
    }
}
