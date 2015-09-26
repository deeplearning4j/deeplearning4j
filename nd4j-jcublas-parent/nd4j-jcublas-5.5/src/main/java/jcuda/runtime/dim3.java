/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package jcuda.runtime;

/**
 * Java port of a dim3.
 *
 * @see JCuda#cudaConfigureCall
 */
public class dim3
{
    /**
     * The x size
     */
    public int x = 1;

    /**
     * The y size
     */
    public int y = 1;

    /**
     * The z size
     */
    public int z = 1;

    /**
     * Creates a new dim3, with default size (1,1,1)
     */
    public dim3()
    {
        x = 1;
        y = 1;
        z = 1;
    }

    /**
     * Creates a new dim3, with the given size
     *
     * @param x The x size
     * @param y The y size
     * @param z The z size
     */
    public dim3(int x, int y, int z)
    {
        this.x = x;
        this.y = y;
        this.z = z;
    }

    /**
     * Returns a String representation of this object.
     *
     * @return A String representation of this object.
     */
    @Override
    public String toString()
    {
        return "dim3["+
            "x="+x+","+
            "y="+y+","+
            "z="+z+"]";
    }
}
