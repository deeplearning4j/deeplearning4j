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
 * CUDA resource view descriptor
 */
public class cudaResourceViewDesc
{
    /**
     * Resource view format
     *
     * @see cudaResourceViewFormat
     */
    public int format;

    /**
     * Width of the resource view
     */
    public long width;

    /**
     * Height of the resource view
     */
    public long height;

    /**
     * Depth of the resource view
     */
    public long depth;

    /**
     * First defined mipmap level
     */
    public int firstMipmapLevel;

    /**
     * Last defined mipmap level
     */
    public int lastMipmapLevel;

    /**
     * First layer index
     */
    public int firstLayer;

    /**
     * Last layer index
     */
    public int lastLayer;

    /**
     * Creates a new, uninitialized cudaResourceViewDesc
     */
    public cudaResourceViewDesc()
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
        return "cudaResourceViewDesc["+createString(",")+"]";
    }

    /**
     * Creates and returns a formatted (aligned, multi-line) String
     * representation of this object
     *
     * @return A formatted String representation of this object
     */
    public String toFormattedString()
    {
        return "CUDA resource view descriptor:\n    "+createString("\n    ");
    }

    /**
     * Creates and returns a string representation of this object,
     * using the given separator for the fields
     *
     * @param f Separator
     * @return A String representation of this object
     */
    private String createString(String f)
    {
        StringBuilder sb = new StringBuilder();
        sb.append("format="+cudaResourceViewFormat.stringFor(format)+f);
        sb.append("width="+width+f);
        sb.append("height="+height+f);
        sb.append("depth="+depth+f);
        sb.append("firstMipmapLevel="+firstMipmapLevel+f);
        sb.append("lastMipmapLevel="+lastMipmapLevel+f);
        sb.append("firstLayer="+firstLayer+f);
        sb.append("lastLayer="+lastLayer+f);
        return sb.toString();
    }
}
