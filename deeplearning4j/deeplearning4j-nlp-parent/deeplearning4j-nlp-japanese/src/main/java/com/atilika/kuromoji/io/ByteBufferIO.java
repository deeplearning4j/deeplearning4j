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
