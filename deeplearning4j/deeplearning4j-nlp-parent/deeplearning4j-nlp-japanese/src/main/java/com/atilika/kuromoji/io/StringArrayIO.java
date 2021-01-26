/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
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

public class StringArrayIO {

    public static String[] readArray(InputStream input) throws IOException {
        DataInputStream dataInput = new DataInputStream(input);
        int length = dataInput.readInt();

        String[] array = new String[length];

        for (int i = 0; i < length; i++) {
            array[i] = dataInput.readUTF();
        }

        return array;
    }

    public static void writeArray(OutputStream output, String[] array) throws IOException {
        DataOutputStream dataOutput = new DataOutputStream(output);
        int length = array.length;

        dataOutput.writeInt(length);

        for (int i = 0; i < array.length; i++) {
            dataOutput.writeUTF(array[i]);
        }
    }

    public static String[][] readArray2D(InputStream input) throws IOException {
        DataInputStream dataInput = new DataInputStream(input);
        int length = dataInput.readInt();

        String[][] array = new String[length][];

        for (int i = 0; i < length; i++) {
            array[i] = readArray(dataInput);
        }

        return array;
    }

    public static void writeArray2D(OutputStream output, String[][] array) throws IOException {
        DataOutputStream dataOutput = new DataOutputStream(output);
        int length = array.length;

        dataOutput.writeInt(length);

        for (int i = 0; i < length; i++) {
            writeArray(dataOutput, array[i]);
        }
    }

    public static String[][] readSparseArray2D(InputStream input) throws IOException {
        DataInputStream dataInput = new DataInputStream(input);
        int length = dataInput.readInt();

        String[][] array = new String[length][];

        int index;

        while ((index = dataInput.readInt()) >= 0) {
            array[index] = readArray(dataInput);
        }

        return array;
    }

    public static void writeSparseArray2D(OutputStream output, String[][] array) throws IOException {
        DataOutputStream dataOutput = new DataOutputStream(output);
        int length = array.length;

        dataOutput.writeInt(length);

        for (int i = 0; i < length; i++) {
            String[] inner = array[i];

            if (inner != null) {
                dataOutput.writeInt(i);
                writeArray(dataOutput, inner);
            }
        }
        // This negative index serves as an end-of-array marker
        dataOutput.writeInt(-1);
    }
}
