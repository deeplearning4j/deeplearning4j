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
 * Java port of a cudaExtent.
 *
 * @see JCuda#cudaMalloc3D
 * @see JCuda#cudaMemset3D
 * @see JCuda#cudaMalloc3DArray
 *
 */
public class cudaExtent
{
    /**
     * The width of this cudaExtent, in elements
     */
    public long width;

    /**
     * The height of this cudaExtent, in elements
     */
    public long height;

    /**
     * The depth of this cudaExtent
     */
    public long depth;

    /**
     * Creates a new cudaExtent with all-zero sizes
     */
    public cudaExtent()
    {
    }

    /**
     * Creates a new cudaExtent with the given sizes
     *
     * @param width The width of the cudaExtent
     * @param height The height of the cudaExtent
     * @param depth The depth of the cudaExtent
     */
    public cudaExtent(int width, int height, int depth)
    {
        this.width = width;
        this.height = height;
        this.depth = depth;
    }

    /**
     * Creates a new cudaExtent with the given sizes
     *
     * @param width The width of the cudaExtent
     * @param height The height of the cudaExtent
     * @param depth The depth of the cudaExtent
     */
    public cudaExtent(long width, long height, long depth)
    {
        this.width = width;
        this.height = height;
        this.depth = depth;
    }

    /**
     * Returns a String representation of this object.
     *
     * @return A String representation of this object.
     */
    @Override
    public String toString()
    {
        return "cudaExtent["+
            "width="+width+","+
            "height="+height+","+
            "depth="+depth+"]";
    }

}