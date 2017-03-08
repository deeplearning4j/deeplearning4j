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
import java.nio.IntBuffer;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.nio.channels.WritableByteChannel;

public class IntegerArrayIO {

    private static final int INT_BYTES = Integer.SIZE / Byte.SIZE;

    public static int[] readArray(InputStream input) throws IOException {
        DataInputStream dataInput = new DataInputStream(input);
        int length = dataInput.readInt();

        ByteBuffer tmpBuffer = ByteBuffer.allocate(length * INT_BYTES);
        ReadableByteChannel channel = Channels.newChannel(dataInput);
        channel.read(tmpBuffer);

        tmpBuffer.rewind();
        IntBuffer intBuffer = tmpBuffer.asIntBuffer();

        int[] array = new int[length];
        intBuffer.get(array);

        return array;
    }

    public static void writeArray(OutputStream output, int[] array) throws IOException {
        DataOutputStream dataOutput = new DataOutputStream(output);
        int length = array.length;

        dataOutput.writeInt(length);

        ByteBuffer tmpBuffer = ByteBuffer.allocate(length * INT_BYTES);
        IntBuffer intBuffer = tmpBuffer.asIntBuffer();

        tmpBuffer.rewind();
        intBuffer.put(array);

        WritableByteChannel channel = Channels.newChannel(dataOutput);
        channel.write(tmpBuffer);
    }

    public static int[][] readArray2D(InputStream input) throws IOException {
        DataInputStream dataInput = new DataInputStream(input);
        int length = dataInput.readInt();

        int[][] array = new int[length][];

        for (int i = 0; i < length; i++) {
            array[i] = readArray(dataInput);
        }

        return array;
    }

    public static void writeArray2D(OutputStream output, int[][] array) throws IOException {
        DataOutputStream dataOutput = new DataOutputStream(output);
        int length = array.length;

        dataOutput.writeInt(length);

        for (int i = 0; i < length; i++) {
            writeArray(dataOutput, array[i]);
        }
    }

    public static int[][] readSparseArray2D(InputStream input) throws IOException {
        DataInputStream dataInput = new DataInputStream(input);
        int length = dataInput.readInt();

        int[][] array = new int[length][];

        int index;

        while ((index = dataInput.readInt()) >= 0) {
            array[index] = readArray(dataInput);
        }

        return array;
    }

    public static void writeSparseArray2D(OutputStream output, int[][] array) throws IOException {
        DataOutputStream dataOutput = new DataOutputStream(output);
        int length = array.length;

        dataOutput.writeInt(length);

        for (int i = 0; i < length; i++) {
            int[] inner = array[i];

            if (inner != null) {
                dataOutput.writeInt(i);
                writeArray(dataOutput, inner);
            }
        }
        // This negative index serves as an end-of-array marker
        dataOutput.writeInt(-1);
    }
}
