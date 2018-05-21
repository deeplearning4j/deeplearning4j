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
 * MNIST database image file. Contains additional header information for the
 * number of rows and columns per each entry.
 * 
 */
public class MnistImageFile extends MnistDbFile {
    private int rows;
    private int cols;

    /**
     * Creates new MNIST database image file ready for reading.
     * 
     * @param name
     *            the system-dependent filename
     * @param mode
     *            the access mode
     * @throws java.io.IOException
     * @throws java.io.FileNotFoundException
     */
    public MnistImageFile(String name, String mode) throws IOException {
        super(name, mode);

        // read header information
        rows = readInt();
        cols = readInt();
    }

    /**
     * Reads the image at the current position.
     *
     * @return matrix representing the image
     * @throws java.io.IOException
     */
    public int[][] readImage() throws IOException {
        int[][] dat = new int[getRows()][getCols()];
        for (int i = 0; i < getCols(); i++) {
            for (int j = 0; j < getRows(); j++) {
                dat[i][j] = readUnsignedByte();
            }
        }
        return dat;
    }

    /**
     * Move the cursor to the next image.
     *
     * @throws java.io.IOException
     */
    public void nextImage() throws IOException {
        super.next();
    }

    /**
     * Move the cursor to the previous image.
     *
     * @throws java.io.IOException
     */
    public void prevImage() throws IOException {
        super.prev();
    }

    @Override
    protected int getMagicNumber() {
        return 2051;
    }

    /**
     * Number of rows per image.
     * 
     * @return int
     */
    public int getRows() {
        return rows;
    }

    /**
     * Number of columns per image.
     * 
     * @return int
     */
    public int getCols() {
        return cols;
    }

    @Override
    public int getEntryLength() {
        return cols * rows;
    }

    @Override
    public int getHeaderSize() {
        return super.getHeaderSize() + 8; // to more integers - rows and columns
    }
}
