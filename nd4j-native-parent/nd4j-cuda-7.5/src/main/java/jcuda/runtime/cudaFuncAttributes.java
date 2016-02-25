/*
 * JCuda - Java bindings for NVIDIA CUDA driver and runtime API
 *
 * Copyright (c) 2009-2015 Marco Hutter - http://www.jcuda.org
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

package jcuda.runtime;

/**
 * CUDA function attributes.<br />
 * <br />
 * Most comments are taken from the CUDA reference manual.<br />
 *
 * @see JCuda#cudaFuncGetAttributes(cudaFuncAttributes, String)
 */
public class cudaFuncAttributes
{

    /**
     * Size of shared memory in bytes
     */
    public long sharedSizeBytes;

    /**
     * Size of constant memory in bytes
     */
    public long constSizeBytes;

    /**
     * Size of local memory in bytes
     */
    public long localSizeBytes;

    /**
     * Maximum number of threads per block
     */
    public int maxThreadsPerBlock;

    /**
     * Number of registers used
     */
    public int numRegs;

    /**
     * PTX virtual architecture version for which the function was
     * compiled. This value is the major PTX version * 10 + the minor PTX
     * version, so a PTX version 1.3 function would return the value 13.
     * For device emulation kernels, this is set to 9999.
     */
    public int ptxVersion;

    /**
     * Binary architecture version for which the function was compiled.
     * This value is the major binary version * 10 + the minor binary version,
     * so a binary version 1.3 function would return the value 13.
     * For device emulation kernels, this is set to 9999.
     */
    public int binaryVersion;

    /**
     * The attribute to indicate whether the function has been compiled with
     * user specified option "-Xptxas --dlcm=ca" set.
     */
    public int cacheModeCA;

    /**
     * Creates new, uninitialized cudaFuncAttributes
     */
    public cudaFuncAttributes()
    {
    }

    /**
     * Returns a String representation of this object.
     *
     * @return A String representation of this object.
     */
    @Override
    public String toString()
    {
        return "cudaFuncAttributes["+
            "sharedSizeBytes="+sharedSizeBytes+","+
            "constSizeBytes="+constSizeBytes+","+
            "localSizeBytes="+localSizeBytes+","+
            "maxThreadsPerBlock="+maxThreadsPerBlock+","+
            "numRegs="+numRegs+","+
            "ptxVersion="+ptxVersion+","+
            "binaryVersion="+binaryVersion+"," +
            "cacheModeCA="+cacheModeCA+"]";
    }


};
