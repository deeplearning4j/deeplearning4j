/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.datasets.mnist;


import java.io.FileNotFoundException;
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
     * @throws IOException
     * @throws FileNotFoundException
     */
    public MnistLabelFile(String name, String mode) throws IOException {
        super(name, mode);
    }

    /**
     * Reads the integer at the current position.
     * 
     * @return integer representing the label
     * @throws IOException
     */
    public int readLabel() throws IOException {
        return readUnsignedByte();
    }

    /** Read the specified number of labels from the current position*/
    public int[] readLabels(int num) throws IOException {
        int[] out = new int[num];
        for (int i = 0; i < num; i++)
            out[i] = readLabel();
        return out;
    }

    @Override
    protected int getMagicNumber() {
        return 2049;
    }
}
