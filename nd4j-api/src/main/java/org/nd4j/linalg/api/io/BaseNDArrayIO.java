/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.nd4j.linalg.api.io;

import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.*;

/**
 * Base class for NDArray IO
 *
 * @author Adam Gibson
 */
public abstract class BaseNDArrayIO implements NDArrayIO {

    /**
     * Write an ndarray to the output stream
     *
     * @param out the ndarray to write
     * @param to  the output stream to write to
     */
    @Override
    public void write(INDArray out, File to) throws IOException {
        FileOutputStream fos = new FileOutputStream(to);
        BufferedOutputStream bos = new BufferedOutputStream(fos);
        write(out, bos);
        bos.flush();
        bos.close();
    }

    /**
     * Write a complex ndarray to an output stream
     *
     * @param out the ndarray to write
     * @param to  the output stream to write to
     */
    @Override
    public void writeComplex(IComplexNDArray out, File to) throws IOException {
        FileOutputStream fos = new FileOutputStream(to);
        BufferedOutputStream bos = new BufferedOutputStream(fos);
        write(out, bos);
        bos.flush();
        bos.close();
    }

    /**
     * Read in an ndarray from a file
     *
     * @param file the file to read in from
     * @return the ndarray that was read in
     */
    @Override
    public INDArray read(File file) throws IOException {
        FileInputStream fis = new FileInputStream(file);
        INDArray ret = read(fis);
        fis.close();
        return ret;
    }

    /**
     * Read in a complex ndarray from a file
     *
     * @param file the ndarray to read from
     * @return the read in ndarray
     */
    @Override
    public IComplexNDArray readComplex(File file) throws IOException {
        FileInputStream fis = new FileInputStream(file);
        IComplexNDArray ret = readComplex(fis);
        fis.close();
        return ret;
    }
}
