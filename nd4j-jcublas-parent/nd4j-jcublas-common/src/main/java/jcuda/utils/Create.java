/*
 * JCudaUtils - Utilities for JCuda 
 * http://www.jcuda.org
 *
 * Copyright (c) 2010 Marco Hutter - http://www.jcuda.org
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

package jcuda.utils;

import java.util.*;

/**
 * Utility class for creating arrays containing random data.
 * By default, the methods of this class will always return
 * the same random data for the same sequence of calls, but
 * by invoking the {@link Create#randomize()} method, the
 * results may be turned to be 'really' random.
 */
public class Create
{
    /**
     * Private constructor to prevent instantiation
     */
    private Create()
    {
    }
    
    /**
     * The random object used in this class
     */
    private static Random random = new Random(0);
    
    /**
     * Will randomize the random number generator
     */
    public static void randomize()
    {
        random = new Random();
    }

    /**
     * Will initialize the random number generator
     * with the given seed.
     * 
     * @param seed The random seed
     */
    public static void randomize(long seed)
    {
        random = new Random(seed);
    }
    
    /**
     * Creates a new array with the given size, containing random data
     * 
     * @param size The size of the array
     * @return The array
     */
    public static float[] createRandomFloatData(int size)
    {
        float a[] = new float[size];
        for (int i=0; i<size; i++)
        {
            a[i] = random.nextFloat();
        }
        return a;
    }

    /**
     * Creates a new array with the given size, containing random data
     * 
     * @param size The size of the array
     * @return The array
     */
    public static double[] createRandomDoubleData(int size)
    {
        double a[] = new double[size];
        for (int i=0; i<size; i++)
        {
            a[i] = random.nextDouble();
        }
        return a;
    }
    
    /**
     * Creates a new array with the given size, containing random data
     * 
     * @param size The size of the array
     * @return The array
     */
    public static int[] createRandomIntData(int size)
    {
        int a[] = new int[size];
        for (int i=0; i<size; i++)
        {
            a[i] = random.nextInt(Integer.MAX_VALUE);
        }
        return a;
    }

}
