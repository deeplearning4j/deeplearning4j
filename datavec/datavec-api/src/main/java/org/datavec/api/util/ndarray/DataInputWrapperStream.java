/*-
 *  * Copyright 2017 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

package org.datavec.api.util.ndarray;

import lombok.AllArgsConstructor;

import java.io.DataInput;
import java.io.IOException;
import java.io.InputStream;

/**
 * A simple class to wrap a {@link DataInput} instance in an {@link InputStream}
 *
 * @author Alex Black
 */
@AllArgsConstructor
public class DataInputWrapperStream extends InputStream {

    private final DataInput underlying;


    @Override
    public int read() throws IOException {
        /*From InputStream.read() javadoc:
        "Reads the next byte of data from the input stream. The value byte is
         returned as an <code>int</code> in the range <code>0</code> to
         <code>255</code>."
         Therefore: we need to use readUnsignedByte(), with returns 0 to 255. readByte() returns -128 to 127
         */
        return underlying.readUnsignedByte();
    }
}
