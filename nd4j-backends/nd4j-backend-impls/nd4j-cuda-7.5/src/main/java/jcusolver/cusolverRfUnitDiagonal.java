/*
 * JCusolver - Java bindings for CUSOLVER, the NVIDIA CUDA solver
 * library, to be used with JCuda
 *
 * Copyright (c) 2010-2015 Marco Hutter - http://www.jcuda.org
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
package jcusolver;

/** CUSOLVERRF unit diagonal */
public class cusolverRfUnitDiagonal
{
    /**
     * default
     */
    public static final int CUSOLVERRF_UNIT_DIAGONAL_STORED_L = 0;
    public static final int CUSOLVERRF_UNIT_DIAGONAL_STORED_U = 1;
    public static final int CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_L = 2;
    public static final int CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_U = 3;

    /**
     * Private constructor to prevent instantiation
     */
    private cusolverRfUnitDiagonal(){}

    /**
     * Returns a string representation of the given constant
     *
     * @return A string representation of the given constant
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CUSOLVERRF_UNIT_DIAGONAL_STORED_L: return "CUSOLVERRF_UNIT_DIAGONAL_STORED_L";
            case CUSOLVERRF_UNIT_DIAGONAL_STORED_U: return "CUSOLVERRF_UNIT_DIAGONAL_STORED_U";
            case CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_L: return "CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_L";
            case CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_U: return "CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_U";
        }
        return "INVALID cusolverRfUnitDiagonal: "+n;
    }
}

