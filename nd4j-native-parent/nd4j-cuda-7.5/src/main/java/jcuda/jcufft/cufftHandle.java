/*
 * JCufft - Java bindings for CUFFT, the NVIDIA CUDA FFT library,
 * to be used with JCuda
 *
 * Copyright (c) 2008-2015 Marco Hutter - http://www.jcuda.org
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

package jcuda.jcufft;

/**
 * A handle type used to store and access CUFFT plans
 */
public class cufftHandle
{
    /**
     * The plan id, written by native methods
     */
    private int plan;

    /**
     * The dimension of this plan
     */
    private int dim = 0;

    /**
     * The cufftType of this plan (cufftType.CUFFT_R2C, cufftType.CUFFT_C2R or cufftType.CUFFT_C2C)
     */
    private int type;

    /**
     * The size of this plan
     */
    private int sizeX = 0;

    /**
     * The size of this plan
     */
    private int sizeY = 0;

    /**
     * The size of this plan
     */
    private int sizeZ = 0;

    /**
     * The batch size of this plan, for 1D transforms
     */
    private int batchSize = 0;

    /**
     * Returns a String representation of this JCufftHandle
     *
     * @return A String representation of this JCufftHandle
     */
    public String toString()
    {
        if (dim == 0)
        {
            return "cufftHandle[uninitialized]";
        }
        String result = "cufftHandle[id="+plan+",dim="+dim+",type="+cufftType.stringFor(type)+", size=";
        switch (dim)
        {
            case 1:
                result += "("+sizeX+"), batch="+batchSize;
                break;

            case 2:
                result += "("+sizeX+","+sizeY+")";
                break;

            case 3:
                result += "("+sizeX+","+sizeY+","+sizeZ+")";
                break;
        }
        result += "]";
        return result;
    }

    /**
     * Set the batch size of this plan
     *
     * @param batchSize The batch size of this plan
     */
    void setBatchSize(int batchSize)
    {
        this.batchSize = batchSize;
    }

    /**
     * Set the type of this plan (JCufft.CUFFT_R2C, JCufft.CUFFT_C2R or JCufft.CUFFT_C2C)
     * @param type
     */
    void setType(int type)
    {
        this.type = type;
    }

    /**
     * Set the dimension of this plan (1,2 or 3)
     *
     * @param dim The dimension of this plan
     */
    void setDimension(int dim)
    {
        this.dim = dim;
    }

    /**
     * Set the size of this plan
     *
     * @param x Size in x
     * @param y Size in y
     * @param z Size in z
     */
    void setSize(int x, int y, int z)
    {
        this.sizeX = x;
        this.sizeY = y;
        this.sizeZ = z;
    }

}
