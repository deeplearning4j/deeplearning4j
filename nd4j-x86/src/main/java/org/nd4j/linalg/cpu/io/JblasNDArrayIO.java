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

package org.nd4j.linalg.cpu.io;

import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.io.BaseNDArrayIO;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.cpu.util.JblasSerde;

import java.io.*;

/**
 * Jblas NDArray IO
 *
 * @author Adam Gibson
 */
public class JblasNDArrayIO extends BaseNDArrayIO {
    /**
     * Read in an ndrray from an input stream
     *
     * @param is the input stream to read in from
     * @return the ndarray read in
     */
    @Override
    public INDArray read(InputStream is) throws IOException {
        return JblasSerde.readJblasBinary(new DataInputStream(is));
    }

    /**
     * Read in a complex ndarray
     *
     * @param is the input stream to read in from
     * @return the complex ndarray that was read in
     */
    @Override
    public IComplexNDArray readComplex(InputStream is) throws IOException {
        return Nd4j.createComplex(read(is));
    }

    /**
     * Write an ndarray to the output stream
     *
     * @param out the ndarray to write
     * @param to  the output stream to write to
     */
    @Override
    public void write(INDArray out, OutputStream to) throws IOException {
        Nd4j.write(out, new DataOutputStream(to));
    }

    /**
     * Write a complex ndarray to an output stream
     *
     * @param out the ndarray to write
     * @param to  the output stream to write to
     */
    @Override
    public void writeComplex(IComplexNDArray out, OutputStream to) throws IOException {
        Nd4j.writeComplex(out, new DataOutputStream(to));
    }
}
