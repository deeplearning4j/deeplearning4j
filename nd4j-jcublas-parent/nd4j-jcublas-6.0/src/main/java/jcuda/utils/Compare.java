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

import java.util.Arrays;

import jcuda.CudaException;

/**
 * Utility functions for comparing arrays. 
 * <br />
 * Some of the functions are ported from the CUTIL
 * comparison functions.
 */
public class Compare
{
    /**
     * The minimum value that a non-null epsilon should have
     */
    private static final float MIN_EPSILON_ERROR = 1e-3f;
    
    /**
     * Whether the output will be verbose
     */
    private static boolean verbose = false;

    /**
     * Set the flag which indicates whether the output will be 
     * verbose and printing information about the differences
     * between the arrays that are compared. 
     * 
     * @param verbose Whether the output will be verbose
     */
    public static void setVerbose(boolean verbose)
    {
        Compare.verbose = verbose;
    }
    
    /**
     * Private constructor to prevent instantiation
     */
    private Compare()
    {
    }


    /**
     * Returns whether the given arrays are equal
     * 
     * @param reference The reference array
     * @param data the actual data array
     */
    public static boolean compare(float reference[], float data[])
    {
        return Arrays.equals(reference, data);
    }

    /**
     * Returns whether the given arrays are equal
     * 
     * @param reference The reference array
     * @param data the actual data array
     */
    public static boolean compare(int reference[], int data[])
    {
        return Arrays.equals(reference, data);
    }


    /**
     * Returns whether the given arrays are equal
     * 
     * @param reference The reference array
     * @param data the actual data array
     */
    public static boolean compare(byte reference[], byte data[])
    {
        return Arrays.equals(reference, data);
    }

    /**
     * Checks if two arrays are equal within the given limits. 
     * If the arrays have different lengths, only the number of
     * elements of the shorter array will be compared.
     * 
     * @param reference The reference array
     * @param data the actual data array
     * @param epsilon The epsilon for the comparison
     * @param threshold The % of array elements that may be differ by
     * more than the given epsilon
     * @return Whether the arrays are equal within the given limits.
     */
    public static boolean compare(byte reference[], byte data[],
        float epsilon, float threshold)
    {
        if (epsilon < 0)
        {
            throw new CudaException(
                "The epsilon must be >=0, but is " + epsilon);
        }

        float maxError = Math.max((float) epsilon, MIN_EPSILON_ERROR);
        int errorCount = 0;
        int len = Math.min(reference.length, data.length);
        for (int i = 0; i < len; i++)
        {
            float diff = Math.abs((float) reference[i] - (float) data[i]);
            if (diff >= maxError)
            {
                errorCount++;
                if (verbose)
                {
                    if (errorCount < 50)
                    {
                        System.out.printf(
                            "\n    ERROR(epsilon=%4.3f), i=%d, (ref)0x%02x " +
                            "/ (data)0x%02x / (diff)%d\n",
                            maxError, i, reference[i], data[i], (int) diff);
                    }
                }
            }
        }

        if (threshold == 0.0f)
        {
            if (errorCount > 0)
            {
                System.out.printf("total # of errors = %d\n", errorCount);
            }
            return (errorCount == 0);
        }
        else
        {

            if (errorCount > 0)
            {
                System.out.printf("%4.2f(%%) of bytes mismatched (count=%d)\n",
                    (float) errorCount * 100 / (float) len, errorCount);
            }

            return (len * threshold > errorCount);
        }
    }

    /**
     * Checks if two arrays are equal within the given limits. 
     * If the arrays have different lengths, only the number of
     * elements of the shorter array will be compared.
     * 
     * @param reference The reference array
     * @param data the actual data array
     * @param epsilon The epsilon for the comparison
     * @return Whether the arrays are equal within the given limits.
     */
    public static boolean compare(byte reference[], byte data[], float epsilon)
    {
        if (epsilon < 0)
        {
            throw new CudaException(
                "The epsilon must be >=0, but is " + epsilon);
        }

        float maxError = Math.max((float) epsilon, MIN_EPSILON_ERROR);
        int errorCount = 0;
        int len = Math.min(reference.length, data.length);
        for (int i = 0; i < len; ++i)
        {
            float diff = Math.abs((float) reference[i] - (float) data[i]);
            if (diff >= maxError)
            {
                errorCount++;
                if (verbose)
                {
                    if (errorCount < 50)
                    {
                        System.out.printf(
                            "\n    ERROR(epsilon=%4.3f), i=%d, (ref)0x%02x " +
                            "/ (data)0x%02x / (diff)%d\n",
                            maxError, i, reference[i], data[i], (int) diff);
                    }
                }
            }
        }
        if (errorCount > 0)
        {
            System.out.printf("total # of errors = %d\n", errorCount);
        }
        return (errorCount == 0);
    }

    
    /**
     * Checks if two arrays are equal within the given limits. 
     * If the arrays have different lengths, only the number of
     * elements of the shorter array will be compared.
     * 
     * @param reference The reference array
     * @param data the actual data array
     * @param epsilon The epsilon for the comparison, must be >=0
     * @param threshold The % of array elements that may be differ by
     * more than the given epsilon
     * @return Whether the arrays are equal within the given limits.
     */
    public static boolean compare(int reference[], int data[],
        float epsilon, float threshold)
    {
        if (epsilon < 0)
        {
            throw new CudaException(
                "The epsilon must be >=0, but is " + epsilon);
        }

        float maxError = Math.max((float) epsilon, MIN_EPSILON_ERROR);
        int errorCount = 0;
        int len = Math.min(reference.length, data.length);
        for (int i = 0; i < len; i++)
        {
            float diff = Math.abs((float) reference[i] - (float) data[i]);
            if (diff >= maxError)
            {
                errorCount++;
                if (verbose)
                {
                    if (errorCount < 50)
                    {
                        System.out.printf(
                            "\n    ERROR(epsilon=%4.3f), i=%d, (ref)0x%02x " +
                            "/ (data)0x%02x / (diff)%d\n",
                            maxError, i, reference[i], data[i], (int) diff);
                    }
                }
            }
        }

        if (threshold == 0.0f)
        {
            if (errorCount > 0)
            {
                System.out.printf("total # of errors = %d\n", errorCount);
            }
            return (errorCount == 0);
        }
        else
        {

            if (errorCount > 0)
            {
                System.out.printf("%4.2f(%%) of bytes mismatched (count=%d)\n",
                    (float) errorCount * 100 / (float) len, errorCount);
            }

            return (len * threshold > errorCount);
        }
    }

    
    /**
     * Checks if two arrays are equal within the given limits. 
     * If the arrays have different lengths, only the number of
     * elements of the shorter array will be compared.
     * 
     * @param reference The reference array
     * @param data the actual data array
     * @param epsilon The epsilon for the comparison, must be >=0
     * @return Whether the arrays are equal within the given limits.
     */
    public static boolean compare(int reference[], int data[], float epsilon)
    {
        if (epsilon < 0)
        {
            throw new CudaException(
                "The epsilon must be >=0, but is " + epsilon);
        }

        float maxError = Math.max((float) epsilon, MIN_EPSILON_ERROR);
        int errorCount = 0;
        int len = Math.min(reference.length, data.length);
        for (int i = 0; i < len; ++i)
        {
            float diff = Math.abs((float) reference[i] - (float) data[i]);
            if (diff >= maxError)
            {
                errorCount++;
                if (verbose)
                {
                    if (errorCount < 50)
                    {
                        System.out.printf(
                            "\n    ERROR(epsilon=%4.3f), i=%d, (ref)0x%02x " +
                            "/ (data)0x%02x / (diff)%d\n",
                            maxError, i, reference[i], data[i], (int) diff);
                    }
                }
            }
        }
        if (errorCount > 0)
        {
            System.out.printf("total # of errors = %d\n", errorCount);
        }
        return (errorCount == 0);
    }
    
    
    /**
     * Checks if two arrays are equal within the given limits. 
     * If the arrays have different lengths, only the number of
     * elements of the shorter array will be compared.
     * 
     * @param reference The reference array
     * @param data the actual data array
     * @param epsilon The epsilon for the comparison
     * @return Whether the arrays are equal within the given limits.
     */
    public static boolean compare(float reference[], float data[], float epsilon)
    {
        return compare(reference, data, epsilon, 0.0f);
    }

    /**
     * Checks if two arrays are equal within the given limits. 
     * If the arrays have different lengths, only the number of
     * elements of the shorter array will be compared.
     * 
     * @param reference The reference array
     * @param data the actual data array
     * @param epsilon The epsilon for the comparison, must be >=0
     * @param threshold The % of array elements that may be differ by
     * more than the given epsilon
     * @return Whether the arrays are equal within the given limits.
     */
    public static boolean compare(float reference[], float data[],
        float epsilon, float threshold)
    {
        if (epsilon < 0)
        {
            throw new CudaException(
                "The epsilon must be >=0, but is " + epsilon);
        }

        boolean result = true;
        int errorCount = 0;
        int len = Math.min(reference.length, data.length);
        for (int i = 0; i < len; ++i)
        {
            float diff = reference[i] - data[i];
            boolean comp = (diff <= epsilon) && (diff >= -epsilon);
            result &= comp;
            if (!comp)
            {
                errorCount++;
            }
            if (verbose)
            {
                if (!comp)
                {
                    System.out.println("ERROR, i = " + i + ",\t " +
                        reference[i] + " / " + data[i] +
                        " (reference / data)\n");
                }
            }
        }

        if (threshold == 0.0f)
        {
            return result;
        }
        else
        {
            if (errorCount > 0)
            {
                System.out.printf("%4.2f(%%) of bytes mismatched (count=%d)\n",
                    (float) errorCount * 100 / (float) len, errorCount);
            }

            return (len * threshold > errorCount);
        }
    }

    /**
     * Checks if two arrays are equal using the L2 norm.
     * If the arrays have different lengths, only the number of
     * elements of the shorter array will be compared.
     * 
     * @param reference The reference array
     * @param data the actual data array
     * @param epsilon The epsilon for the comparison, must be >=0
     * @return Whether the arrays are equal within the given limits.
     */
    public static boolean compareL2(float reference[], float data[],
        float epsilon)
    {
        if (epsilon < 0)
        {
            throw new CudaException(
                "The epsilon must be >=0, but is " + epsilon);
        }

        float error = 0;
        float ref = 0;
        int len = Math.min(reference.length, data.length);
        for (int i = 0; i < len; ++i)
        {

            float diff = reference[i] - data[i];
            error += diff * diff;
            ref += reference[i] * reference[i];
        }

        float normRef = (float) Math.sqrt(ref);
        if (Math.abs(ref) < 1e-7)
        {
            if (verbose)
            {
                System.out.println("ERROR, reference l2-norm is 0\n");
            }
            return false;
        }
        float normError = (float) Math.sqrt(error);
        error = normError / normRef;
        boolean result = error < epsilon;
        if (verbose)
        {
            if (!result)
            {
                System.out.println("ERROR, l2-norm error " + error +
                    " is greater than epsilon " + epsilon + "\n");
            }
        }

        return result;
    }

}
