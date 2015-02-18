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

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * NDArrayIO (interop)
 *
 * @author Adam Gibson
 */
public interface NDArrayIO {


    /**
     * Read in an ndrray from an input stream
     *
     * @param is the input stream to read in from
     * @return the ndarray read in
     */
    public INDArray read(InputStream is) throws IOException;

    /**
     * Read in a complex ndarray
     *
     * @param is the input stream to read in from
     * @return the complex ndarray that was read in
     */
    public IComplexNDArray readComplex(InputStream is) throws IOException;


    /**
     * Read in an ndarray from a file
     *
     * @param file the file to read in from
     * @return the ndarray that was read in
     */
    public INDArray read(File file) throws IOException;


    /**
     * Read in a complex ndarray from a file
     *
     * @param file the ndarray to read from
     * @return the read in ndarray
     */
    public IComplexNDArray readComplex(File file) throws IOException;


    /**
     * Write an ndarray to the output stream
     *
     * @param out the ndarray to write
     * @param to  the output stream to write to
     */
    public void write(INDArray out, File to) throws IOException;

    /**
     * Write a complex ndarray to an output stream
     *
     * @param out the ndarray to write
     * @param to  the output stream to write to
     */
    public void writeComplex(IComplexNDArray out, File to) throws IOException;

    /**
     * Write an ndarray to the output stream
     *
     * @param out the ndarray to write
     * @param to  the output stream to write to
     */
    public void write(INDArray out, OutputStream to) throws IOException;

    /**
     * Write a complex ndarray to an output stream
     *
     * @param out the ndarray to write
     * @param to  the output stream to write to
     */
    public void writeComplex(IComplexNDArray out, OutputStream to) throws IOException;


}
