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
 * Command line argument utility functions, similar to the CUTIL 
 * functions. <br />
 * <br />
 * All command line arguments begin with '--' followed by the token. 
 * The token and value are separated by '='. <br />
 * Example:</br> 
 * <code>--samples=50</code>
 * <br />
 * Arrays have the form <br />
 * <code>--model=[one.obj,two.obj,three.obj]</code>
 * (without whitespaces)
 */
public class CommandLine
{
    /**
     * Private constructor to prevent instantiation
     */
    private CommandLine()
    {
    }
    
    /**
     * Returns whether the given command line arguments contain an
     * argument with the given name (the name must be given without
     * the "--" prefix);
     * 
     * @param argv The arguments
     * @param flagName The argument to look for
     * @return Whether the argument is found
     */
    public static boolean checkCommandLineFlag(String argv[], String flagName)
    {
        for (int i=0; i<argv.length; i++)
        {
            String a = argv[i].trim();
            String n = flagName.trim();
            if (a.startsWith("--"+n))
            {
                return true;
            }
        }
        return false;
    }
    
    
    /**
     * Creates a map that maps the arguments (without the "--" prefix)
     * to their values
     * 
     * @param argv The arguments
     * @return The map
     */
    private static Map<String, String> createArgMap(String argv[])
    {
        Map<String, String> argMap = new HashMap<String, String>();
        String allArgs = "";
        for (String a : argv)
        {
            allArgs += a;
        }
        Scanner scanner = new Scanner(allArgs);
        scanner.useDelimiter("[ =]");
        String key = null;
        while (scanner.hasNext())
        {
            String token = scanner.next();
            if (token.isEmpty() || token.equals(" "))
            {
                continue;
            }
            //System.out.println("Handle token '"+token+"'");
            if (key == null)
            {
                if (!token.startsWith("-"))
                {
                    throw new IllegalArgumentException("Illegal argument: "+token);
                }
                token = token.substring(1);
                if (token.startsWith("-"))
                {
                    token = token.substring(1);
                }
                key = token; 
            }
            else
            {
                if (token.startsWith("-"))
                {
                    argMap.put(key, null);
                    key = null;

                    token = token.substring(1);
                    if (token.startsWith("-"))
                    {
                        token = token.substring(1);
                    }
                    key = token; 
                }
                else
                {
                    argMap.put(key, token);
                }
            }
        }
        return argMap;
        
    }
    

    /**
     * Returns the value of the command line argument with the given
     * name as an int.
     * 
     * @param argv The arguments
     * @param argName The name of the argument
     * @param defaultValue The default value to use of no value was given
     * @return The value of the argument
     * @throws IllegalArgumentException If the given argument has
     * no appropriate value
     */
    public static int getCommandLineArgumentInt(String argv[], String argName, 
        int defaultValue)
    {
        Map<String, String> argMap = createArgMap(argv);
        String value = argMap.get(argName);
        if (value == null)
        {
            return defaultValue;
        }
        try
        {
            return Integer.parseInt(value);
        }
        catch (NumberFormatException e)
        {
            throw new IllegalArgumentException(
                "Argument '"+argName+"' has illegal int value '"+value+"'");
        }
    }

    /**
     * Returns the value of the command line argument with the given
     * name as a float.
     * 
     * @param argv The arguments
     * @param argName The name of the argument
     * @param defaultValue The default value to use of no value was given
     * @return The value of the argument
     * @throws IllegalArgumentException If the given argument has
     * no appropriate value
     */
    public static float getCommandLineArgumentFloat(String argv[], String argName, 
        float defaultValue)
    {
        Map<String, String> argMap = createArgMap(argv);
        String value = argMap.get(argName);
        if (value == null)
        {
            return defaultValue;
        }
        try
        {
            return Float.parseFloat(value);
        }
        catch (NumberFormatException e)
        {
            throw new IllegalArgumentException(
                "Argument '"+argName+"' has illegal float value '"+value+"'");
        }
    }

    /**
     * Returns the value of the command line argument with the given
     * name as a String.
     * 
     * @param argv The arguments
     * @param argName The name of the argument
     * @return The value of the argument
     */
    public static String getCommandLineArgumentString(String argv[], String argName)
    {
        Map<String, String> argMap = createArgMap(argv);
        String value = argMap.get(argName);
        return value;
    }

    /**
     * Returns the value of the command line argument with the given
     * name as a list of strings.
     * 
     * @param argv The arguments
     * @param argName The name of the argument
     * @return The value of the argument
     * @throws IllegalArgumentException If the given argument has
     * no appropriate value
     */
    public static List<String> getCommandLineArgumentListString(String argv[], String argName)
    {
        Map<String, String> argMap = createArgMap(argv);
        String value = argMap.get(argName);
        if (value==null)
        {
            return null;
        }
        if (!value.startsWith("[") || !value.endsWith("]"))
        {
            throw new IllegalArgumentException(
                "Argument '"+argName+"' has illegal array value '"+value+"'");
        }
        value = value.substring(1, value.length()-1);
        String values[] = value.split(",");
        List<String> result = new ArrayList<String>();
        for (String s : values)
        {
            result.add(s);
        }
        return result;
    }

}
