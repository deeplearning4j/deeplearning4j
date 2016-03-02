/*
 * JCurand - Java bindings for CURAND, the NVIDIA CUDA random
 * number generation library, to be used with JCuda
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
package jcuda.jcurand;

/**
 * CURAND orderings of results in memory
 */
public class curandOrdering
{
    /**
     * Best ordering for pseudorandom results
     */
    public static final int CURAND_ORDERING_PSEUDO_BEST = 100;
    /**
     * Specific default 4096 thread sequence for pseudorandom results
     */
    public static final int CURAND_ORDERING_PSEUDO_DEFAULT = 101;
    /**
     * Specific seeding pattern for fast lower quality pseudorandom results
     */
    public static final int CURAND_ORDERING_PSEUDO_SEEDED = 102;
    /**
     * Specific n-dimensional ordering for quasirandom results
     */
    public static final int CURAND_ORDERING_QUASI_DEFAULT = 201;

    /**
     * Private constructor to prevent instantiation
     */
    private curandOrdering(){}

    /**
     * Returns a string representation of the given constant
     *
     * @return A string representation of the given constant
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CURAND_ORDERING_PSEUDO_BEST: return "CURAND_ORDERING_PSEUDO_BEST";
            case CURAND_ORDERING_PSEUDO_DEFAULT: return "CURAND_ORDERING_PSEUDO_DEFAULT";
            case CURAND_ORDERING_PSEUDO_SEEDED: return "CURAND_ORDERING_PSEUDO_SEEDED";
            case CURAND_ORDERING_QUASI_DEFAULT: return "CURAND_ORDERING_QUASI_DEFAULT";
        }
        return "INVALID curandOrdering";
    }
}

