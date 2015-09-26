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

import java.io.*;
import java.util.*;
import java.util.regex.Pattern;

import jcuda.CudaException;

/**
 * File I/O functions, similar to the CUTIL file functions
 */
public class FileIO
{
    /**
     * Private constructor to prevent instantiation
     */
    private FileIO()
    {
    }
    
    /**
     * Reads the file with the given name containing single precision floating 
     * point data, and returns the contents of this file as an array
     * 
     * @param filename The name of the file
     * @return The contents of the file
     * @throws CudaException if the file could not be read
     */
    public static float[] readFileFloat(String filename)
    {
        if (filename == null)
        {
            throw new CudaException("The filename is null");
        }

        List<Float> values = new ArrayList<Float>();
        try 
        {
            Scanner scanner = new Scanner(new File(filename));
            scanner.useDelimiter(Pattern.compile("\\s+"));
            while (scanner.hasNext()) 
            {
                try
                {
                    values.add(Float.parseFloat(scanner.next()));
                } 
                catch (NumberFormatException e)
                {
                    throw new CudaException(
                        "Could not read value from file '"+filename+"'", e);
                }
            }
            scanner.close();
        }
        catch (FileNotFoundException e)
        {
            throw new CudaException("File '"+filename+"' not found", e);
        }

        float data[] = new float[values.size()];
        for (int i=0; i<values.size(); i++)
        {
            data[i] = values.get(i);
        }
        return data;
    }

    /**
     * Reads the file with the given name containing double precision floating 
     * point data, and returns the contents of this file as an array
     * 
     * @param filename The name of the file
     * @return The contents of the file
     * @throws CudaException if the file could not be read
     */
    public static double[] readFileDouble(String filename)
    {
        if (filename == null)
        {
            throw new CudaException("The filename is null");
        }

        List<Double> values = new ArrayList<Double>();
        try 
        {
            Scanner scanner = new Scanner(new File(filename));
            scanner.useDelimiter(Pattern.compile("\\s+"));
            while (scanner.hasNext()) 
            {
                try
                {
                    values.add(Double.parseDouble(scanner.next()));
                } 
                catch (NumberFormatException e)
                {
                    throw new CudaException(
                        "Could not read value from file '"+filename+"'", e);
                }
            }
            scanner.close();
        }
        catch (FileNotFoundException e)
        {
            throw new CudaException("File '"+filename+"' not found", e);
        }

        double data[] = new double[values.size()];
        for (int i=0; i<values.size(); i++)
        {
            data[i] = values.get(i);
        }
        return data;
    }

    /**
     * Reads the file with the given name containing integer data, and 
     * returns the contents of this file as an array
     * 
     * @param filename The name of the file
     * @return The contents of the file
     * @throws CudaException if the file could not be read
     */
    public static int[] readFileInt(String filename)
    {
        if (filename == null)
        {
            throw new CudaException("The filename is null");
        }

        List<Integer> values = new ArrayList<Integer>();
        try 
        {
            Scanner scanner = new Scanner(new File(filename));
            scanner.useDelimiter(Pattern.compile("\\s+"));
            while (scanner.hasNext()) 
            {
                try
                {
                    values.add(Integer.parseInt(scanner.next()));
                } 
                catch (NumberFormatException e)
                {
                    throw new CudaException(
                        "Could not read value from file '"+filename+"'", e);
                }
            }
            scanner.close();
        }
        catch (FileNotFoundException e)
        {
            throw new CudaException("File '"+filename+"' not found", e);
        }

        int data[] = new int[values.size()];
        for (int i=0; i<values.size(); i++)
        {
            data[i] = values.get(i);
        }
        return data;
    }

    /**
     * Reads the file with the given name containing byte data, and 
     * returns the contents of this file as an array
     * 
     * @param filename The name of the file
     * @return The contents of the file
     * @throws CudaException if the file could not be read
     */
    public static byte[] readFileByte(String filename)
    {
        if (filename == null)
        {
            throw new CudaException("The filename is null");
        }

        List<Byte> values = new ArrayList<Byte>();
        try 
        {
            Scanner scanner = new Scanner(new File(filename));
            scanner.useDelimiter(Pattern.compile("\\s+"));
            while (scanner.hasNext()) 
            {
                try
                {
                    values.add(Byte.parseByte(scanner.next()));
                } 
                catch (InputMismatchException e)
                {
                    throw new CudaException(
                        "Could not read value from file '"+filename+"'", e);
                }
            }
            scanner.close();
        }
        catch (FileNotFoundException e)
        {
            throw new CudaException("File '"+filename+"' not found", e);
        }

        byte data[] = new byte[values.size()];
        for (int i=0; i<values.size(); i++)
        {
            data[i] = values.get(i);
        }
        return data;
    }
    

    /**
     * Returns a String containing the values of the given
     * float, double, byte or int array separated by single
     * spaces
     * 
     * @param array The array
     * @return The String for the array
     */
    private static String arrayString(Object array)
    {
        StringBuffer sb = new StringBuffer();
        if (array instanceof float[])
        {
            float a[] = (float[])array;
            for (int i=0; i<a.length; i++)
            {
                sb.append(String.valueOf(a[i])+" ");
            }
        }
        else if (array instanceof double[])
        {
            double a[] = (double[])array;
            for (int i=0; i<a.length; i++)
            {
                sb.append(String.valueOf(a[i])+" ");
            }
        }
        else if (array instanceof int[])
        {
            int a[] = (int[])array;
            for (int i=0; i<a.length; i++)
            {
                sb.append(String.valueOf(a[i])+" ");
            }
        }
        else if (array instanceof byte[])
        {
            byte a[] = (byte[])array;
            for (int i=0; i<a.length; i++)
            {
                sb.append(String.valueOf(a[i])+" ");
            }
        }
        return sb.toString();
    }

    /**
     * Writes the given array into a file with the given name.
     * 
     * @param filename The file name
     * @param data The data to write
     * @param epsilonString The epsilon
     * @throws CudaException If either the filename or the data
     * are null, or the file could not be written.
     */
    private static void writeFile(String filename, Object data, String epsilonString)
    {
        if (filename == null)
        {
            throw new CudaException("The filename is null");
        }
        if (data == null)
        {
            throw new CudaException("The data is null");
        }

        BufferedWriter bw = null;
        try
        {
            bw = new BufferedWriter(
                new OutputStreamWriter(new FileOutputStream(filename)));
            bw.write("# "+epsilonString+"\n");
            bw.write(arrayString(data));
            bw.write("\n");
        }
        catch (FileNotFoundException e)
        {
            throw new CudaException(
                "File '"+filename+"' not found", e);
        }
        catch (IOException e)
        {
            throw new CudaException(
                "Error while writing file '"+filename+"'", e);
        }
        finally
        {
            if (bw != null)
            {
                try
                {
                    bw.close();
                }
                catch (IOException e)
                {
                    throw new CudaException(
                        "Error while closing file '"+filename+"'", e);
                }
            }
        }
        
    }
    
    
    /**
     * Writes the given array into a file with the given name.
     *  
     * @param filename The name of the file
     * @param data The data to write
     * @param epsilon Epsilon for comparison
     * @throws CudaException if the file could not be written
     */
    public static void writeFile(String filename, float data[], float epsilon)
    {
        writeFile(filename, data, String.valueOf(epsilon));
    }

    /**
     * Writes the given array into a file with the given name.
     *  
     * @param filename The name of the file
     * @param data The data to write
     * @param epsilon Epsilon for comparison
     * @throws CudaException if the file could not be written
     */
    public static void writeFile(String filename, double data[], double epsilon)
    {
        writeFile(filename, data, String.valueOf(epsilon));
    }

    /**
     * Writes the given array into a file with the given name.
     *  
     * @param filename The name of the file
     * @param data The data to write
     * @throws CudaException if the file could not be written
     */
    public static void writeFile(String filename, int data[])
    {
        writeFile(filename, data, "0");
    }


    /**
     * Writes the given array into a file with the given name.
     *  
     * @param filename The name of the file
     * @param data The data to write
     * @throws CudaException if the file could not be written
     */
    public static void writeFile(String filename, byte data[])
    {
        writeFile(filename, data, "0");
    }

    
    /**
     * Read the contents of the file with the given name,
     * and return it as a String
     * 
     * @param filename The name of the file
     * @return The contents of the file
     * @throws CudaException If the file could not be read
     */
    public static String readFileAsString(String filename)
    {
        BufferedReader br = null;
        try
        {
            br = new BufferedReader(
                new InputStreamReader(new FileInputStream(filename)));
            StringBuilder sb = new StringBuilder();
            String line = null;
            while (true)
            {
                line = br.readLine();
                if (line == null)
                {
                    break;
                }
                sb.append(line+"\n");
            }
            return sb.toString();
        }
        catch (FileNotFoundException e)
        {
            throw new CudaException("File not found: "+filename, e);
        }
        catch (IOException e)
        {
            throw new CudaException("Error while reading file "+filename, e);
        }
        finally
        {
            if (br != null)
            {
                try
                {
                    br.close();
                }
                catch (IOException ex)
                {
                }
            }
        }
    }

}
