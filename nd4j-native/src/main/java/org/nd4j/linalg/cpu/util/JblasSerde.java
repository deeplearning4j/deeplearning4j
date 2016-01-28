/*
 *
 *  * Copyright 2015 Skymind,Inc.
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
 *
 *
 */

package org.nd4j.linalg.cpu.util;

import org.jblas.DoubleMatrix;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.NDArray;

import java.io.DataInputStream;
import java.io.IOException;

/**
 * Handles loading and saving of jblas matrices
 *
 * @author Adam Gibson
 */
public class JblasSerde {

    public static INDArray fromAscii(String path) throws IOException {
        return new NDArray(DoubleMatrix.loadAsciiFile(path));
    }

    /**
     * Read in the jblas binary format
     *
     * @param dataInputStream the data inputstream to use
     * @return an ndarray of the same shape
     * @throws IOException
     */
    public static INDArray readJblasBinary(DataInputStream dataInputStream) throws IOException {
        DoubleMatrix d = new DoubleMatrix();
        d.in(dataInputStream);
        return new NDArray(d);

    }


}
