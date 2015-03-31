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

import java.util.Locale;

/**
 * Utility methods for creating formatted String representations
 * of 1D- 2D- and 3D arrays
 */
public class Print
{
    /**
     * Private constructor to prevent instantiation
     */
    private Print()
    {
    }
    
    /**
     * The default format that will be used for floating point numbers
     */
    private static final String DEFAULT_FLOAT_FORMAT = "%6.3f  ";
    
    /**
     * The default Locale that will be used for formatting
     */
    private static final Locale DEFAULT_LOCALE = Locale.ENGLISH;
    
    /**
     * Creates a String representation of the given array, using 
     * the default format string for its elements.
     *
     * @param a The array
     * @return The String representation
     */
    public static String toString1D(float a[])
    {
        return toString1D(a, DEFAULT_FLOAT_FORMAT);
    }
    
    /**
     * Creates a String representation of the given array, using 
     * the given format string for its elements.
     *
     * @param a The array
     * @param format The format string
     * @return The String representation
     */
    public static String toString1D(float a[], String format)
    {
        StringBuilder sb = new StringBuilder();
        for (int i=0; i<a.length; i++)
        {
            sb.append(String.format(DEFAULT_LOCALE, format, a[i]));
        }
        return sb.toString();
    }

    /**
     * Creates a String representation of the given array, using 
     * the default format string for its elements. The String 
     * will be formatted as a rectangular matrix with the given
     * number of columns. 
     *
     * @param a The array
     * @param columns The number of columns
     * @return The String representation
     */
    public static String toString2D(float a[], int columns)
    {
        return toString2D(a, columns, DEFAULT_FLOAT_FORMAT);
    }
    
    /**
     * Creates a String representation of the given array, using 
     * the given format string for its elements. The String 
     * will be formatted as a rectangular matrix with the given
     * number of columns. 
     *
     * @param a The array
     * @param columns The number of columns
     * @param format The format string
     * @return The String representation
     */
    public static String toString2D(float[] a, int columns, String format)
    {
        StringBuilder sb = new StringBuilder();
        for (int i=0; i<a.length; i++)
        {
            if (i>0 && i % columns == 0)
            {
                sb.append("\n");
            }
            sb.append(String.format(DEFAULT_LOCALE, format, a[i]));
        }
        return sb.toString();
    }

    /**
     * Creates a String representation of the given array, using 
     * the default format string for its elements. The String 
     * will be formatted as a rectangular matrix.
     *
     * @param a The array
     * @return The String representation
     */
    public static String toString2D(float a[][])
    {
        return toString2D(a, DEFAULT_FLOAT_FORMAT);
    }
    
    /**
     * Creates a String representation of the given array, using 
     * the given format string for its elements. The String 
     * will be formatted as a rectangular matrix.
     *
     * @param a The array
     * @param format The format string
     * @return The String representation
     */
    public static String toString2D(float a[][], String format)
    {
        StringBuilder sb = new StringBuilder();
        for (int i=0; i<a.length; i++)
        {
            sb.append(toString1D(a[i], format));
            sb.append("\n");
        }
        return sb.toString();
    }

    /**
     * Creates a String representation of the given array, using 
     * the default format string for its elements. The String 
     * will be formatted as a sequence of rectangular matrices.
     *
     * @param a The array
     * @return The String representation
     */
    public static String toString3D(float a[][][])
    {
        return toString3D(a, DEFAULT_FLOAT_FORMAT);
    }
    
    /**
     * Creates a String representation of the given array, using 
     * the given format string for its elements. The String 
     * will be formatted as a sequence of rectangular matrices.
     *
     * @param a The array
     * @param format The format string
     * @return The String representation
     */
    public static String toString3D(float a[][][], String format)
    {
        StringBuilder sb = new StringBuilder();
        for (int i=0; i<a.length; i++)
        {
            sb.append(toString2D(a[i], format));
            sb.append("\n");
        }
        return sb.toString();
    }

}
