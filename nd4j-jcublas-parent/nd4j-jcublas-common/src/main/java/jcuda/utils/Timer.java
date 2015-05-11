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
 * Simple timer functionality, similar to the CUTIL timer functions
 */
public class Timer
{
    /**
     * The current set of timers
     */
    private static Map<Object, Timer> timers = 
        new LinkedHashMap<Object, Timer>();

    /**
     * Creates a new timer with the given name
     * 
     * @param name The name of the timer
     */
    public static synchronized void createTimer(Object name)
    {
        timers.put(name, new Timer());
    }
    
    /**
     * Deletes the timer with the given name
     * 
     * @param name The name of the timer
     */
    public static synchronized void deleteTimer(Object name)
    {
        timers.remove(name);
    }

    /**
     * Start the timer with the given name
     * 
     * @param name The name of the timer
     */
    public static synchronized void startTimer(Object name)
    {
        getTimer(name).start();
    }
    
    /**
     * Stop the timer with the given name
     * 
     * @param name The name of the timer
     */
    public static synchronized void stopTimer(Object name)
    {
        getTimer(name).stop();
    }
    
    /**
     * Reset the timer with the given name
     * 
     * @param name The name of the timer to reset
     */
    public static synchronized void resetTimer(Object name)
    {
        getTimer(name).reset();
    }
    
    /**
     * Returns the total time in milliseconds of all runs since the
     * creation or the last reset.
     * 
     * @param name The name of the timer
     * @return The time
     */
    public static synchronized int getTimerValue(Object name)
    {
        return getTimer(name).getValue();
    }

    /**
     * Returns the average time in milliseconds for the timer, which is
     * the total time for the timer divided by the number of completed 
     * (stopped) runs the timer has made. This excludes the current 
     * running time if the timer is currently running.
     * 
     * @return The average timer value
     */ 
    public static synchronized int getAverageTimerValue(Object name)
    {
        return getTimer(name).getAverageValue();
    }
    
    
    /**
     * Pretty print a summary of all timers that currently exist
     */
    public static synchronized void prettyPrint()
    {
        System.out.println(createPrettyString());
    }
    
    
    /**
     * Creates a "pretty" String containing a summary of all timers
     * that currently exist.
     * 
     * @return A pretty summary
     */
    public static synchronized String createPrettyString()
    {
        int maxLength = 0;
        for (Object object : timers.keySet())
        {
            maxLength = Math.max(maxLength, String.valueOf(object).length());
        }
        StringBuilder sb = new StringBuilder();
        int headerIndent = Math.max(0, maxLength-3);
        sb.append(String.format("%"+headerIndent+"s  [ms] ", " "));
        sb.append("Duration:    Average:\n");
        for (Map.Entry<Object, Timer> entry : timers.entrySet())
        {
            Object key = entry.getKey();
            String keyString = String.valueOf(key);
            sb.append(String.format("%"+(maxLength+2)+"s: ", keyString));
            Timer timer = entry.getValue();
            sb.append(String.format("%9d", timer.getValue()) + "   ");
            sb.append(String.format("%9d", timer.getAverageValue()));
            if (timer.running)
            {
                sb.append(" (running)");
            }
            sb.append("\n");
        }
        return sb.toString();
    }
    
    
    /**
     * Returns the Timer with the given name. Creates it if necessary.
     * 
     * @param name The name of the timer.
     * @return The timer
     */
    private static Timer getTimer(Object name)
    {
        Timer timer = timers.get(name);
        if (timer == null)
        {
            timer = new Timer();
            timers.put(name, timer);
        }
        return timer;
    }
    
    /**
     * The System.nanoTime of the last start
     */
    private long startTime;
    
    /**
     * The total time of this timer, in nanoseconds
     */
    private long totalTimeNS = 0;
    
    /**
     * The runs of the timer
     */
    private int runs = 0;
    
    /**
     * Whether this timer is currently running
     */
    private boolean running = false;

    /**
     * Private constructor. Instantiation only via the static methods.
     */
    private Timer()
    {
    }
    
    /**
     * Start this timer
     */
    private void start()
    {
        startTime = System.nanoTime();
        running = true;
    }
    
    /**
     * Stop this timer
     */
    private void stop()
    {
        long stopTime = System.nanoTime();
        totalTimeNS += (stopTime-startTime);
        runs++;
        running = false;
    }
    
    /**
     * Reset this timer
     */
    private void reset()
    {
        runs = 0;
        totalTimeNS = 0;
        running = false;
    }

    /**
     * Returns the value of this timer, in milliseconds
     * 
     * @return The value of this timer
     */
    private int getValue()
    {
        if (running)
        {
            long currentTime = System.nanoTime();
            long time = totalTimeNS + (currentTime-startTime);
            return (int)(time / 1000000);
        }
        return (int)(totalTimeNS / 1000000);
    }
    
    /**
     * Returns the average value of this timer, in milliseconds
     * 
     * @return The average value of this timer
     */
    private int getAverageValue()
    {
        if (runs == 0)
        {
            return 0;
        }
        return (int)(totalTimeNS / 1000000 / runs);
    }
    
}
