/*-
 *  * Copyright 2016 Skymind, Inc.
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

package org.datavec.image.mnist;


import java.io.IOException;


/**
 * 
 * MNIST database label file.
 * 
 */
public class MnistLabelFile extends MnistDbFile {

    /**
     * Creates new MNIST database label file ready for reading.
     * 
     * @param name
     *            the system-dependent filename
     * @param mode
     *            the access mode
     * @throws java.io.IOException
     * @throws java.io.FileNotFoundException
     */
    public MnistLabelFile(String name, String mode) throws IOException {
        super(name, mode);
    }

    /**
     * Reads the integer at the current position.
     *
     * @return integer representing the label
     * @throws java.io.IOException
     */
    public int readLabel() throws IOException {
        return readUnsignedByte();
    }

    @Override
    protected int getMagicNumber() {
        return 2049;
    }
}
