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
 * Java port of the cudaChannelFormatDesc.<br />
 * <br />
 * Most comments are taken from the CUDA reference manual.
 *
 * @see JCuda#cudaMallocArray
 * @see JCuda#cudaMalloc3DArray
 * @see jcuda.runtime.cudaChannelFormatKind
 */
public class cudaChannelFormatDesc
{
    /**
     * Number of bits in x component
     */
    public int x;

    /**
     * Number of bits in y component
     */
    public int y;

    /**
     * Number of bits in z component
     */
    public int z;

    /**
     * Number of bits in w component
     */
    public int w;

    /**
     * The channel format kind. Must be one of the cudaChannelFormatKind values.
     *
     * @see jcuda.runtime.cudaChannelFormatKind
     */
    public int f;

    /**
     * Creates an uninitialized cudaChannelFormatDesc
     */
    public cudaChannelFormatDesc()
    {
    }

    /**
     * Creates a cudaChannelFormatDesc with the given bit counts
     * and the given format kind.
     *
     * @param x Number of bits in x component
     * @param y Number of bits in y component
     * @param z Number of bits in z component
     * @param w Number of bits in w component
     * @param f The format kind
     *
     * @see jcuda.runtime.cudaChannelFormatKind
     */
    public cudaChannelFormatDesc(int x, int y, int z, int w, int f)
    {
        this.x = x;
        this.y = y;
        this.z = z;
        this.w = w;
        this.f = f;
    }

    /**
     * Returns a String representation of this object.
     *
     * @return A String representation of this object.
     */
    @Override
    public String toString()
    {
        return "cudaChannelFormatDesc["+
            "x="+x+","+
            "y="+y+","+
            "z="+z+","+
            "w="+w+","+
            "f="+cudaChannelFormatKind.stringFor(f)+"]";
    }

}
