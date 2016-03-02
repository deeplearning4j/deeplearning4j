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
 * Java port of cudaMemcpy3DPeerParms.<br />
 * <br />
 * Most comments are taken from the CUDA reference manual.<br />
 * <br />
 * @see JCuda#cudaMemcpy3DPeer
 */
public class cudaMemcpy3DPeerParms
{
    /**
     * The source array.
     */
    public cudaArray srcArray = new cudaArray();

    /**
     * The srcPos and dstPos fields are optional offsets into the source and destination objects and are defined in
     * units of each object's elements. The element for a host or device pointer is assumed to be unsigned char.
     * For CUDA arrays, positions must be in the range [0, 2048) for any dimension.
     */
    public cudaPos srcPos = new cudaPos();

    /**
     * The source pointer.
     */
    public cudaPitchedPtr srcPtr = new cudaPitchedPtr();

    /**
     * Source device
     */
    public int srcDevice;


    /**
     * The destination array.
     */
    public cudaArray dstArray = new cudaArray();

    /**
     * The srcPos and dstPos fields are optional offsets into the source and destination objects and are defined in
     * units of each object's elements. The element for a host or device pointer is assumed to be unsigned char.
     * For CUDA arrays, positions must be in the range [0, 2048) for any dimension.
     */
    public cudaPos dstPos = new cudaPos();

    /**
     * The destination pointer.
     */
    public cudaPitchedPtr dstPtr = new cudaPitchedPtr();

    /**
     * Destination device
     */
    public int dstDevice;

    /**
     * The extent field defines the dimensions of the transferred area in elements. If a CUDA array is participating
     * in the copy the extent is defined in terms of that array's elements. If no CUDA array is participating in the
     * copy then the extents are defined in elements of unsigned char.
     */
    public cudaExtent extent = new cudaExtent();

    /**
     * Returns a String representation of this object.
     *
     * @return A String representation of this object.
     */
    @Override
    public String toString()
    {
        return "cudaMemcpy3DPeerParms["+createString(",")+"]";
    }

    /**
     * Creates and returns a formatted (aligned, multi-line) String
     * representation of this object
     *
     * @return A formatted String representation of this object
     */
    public String toFormattedString()
    {
        return "Parameters for cudaMemcopy3DPeer:\n    "+createString("\n    ");
    }

    /**
     * Creates and returns a string representation of this object,
     * using the given separator for the fields
     *
     * @return A String representation of this object
     */
    private String createString(String f)
    {
        return
            "srcArray="+srcArray+f+
            "srcPos="+srcPos+f+
            "srcPtr="+srcPtr+f+
            "srcDevice="+srcDevice+f+
            "dstArray="+dstArray+f+
            "dstPos="+dstPos+f+
            "dstPtr="+dstPtr+f+
            "dstDevice="+dstDevice+f+
            "extent="+extent;
    }



}
