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

package jcuda.jcufft;

/**
 * The type of a CUFFT operation
 */
public class cufftType
{
    /**
     * CUFFT transform type Real to complex (interleaved)
     */
    public static final int CUFFT_R2C = 0x2A;
    
    
    /**
     * CUFFT transform type Complex (interleaved) to real
     */
    public static final int CUFFT_C2R = 0x2C;
    
    
    /**
     * CUFFT transform type Complex to complex, interleaved
     */
    public static final int CUFFT_C2C = 0x29;

    /**
     * CUFFT transform type Double to Double-Complex
     */
    public static final int CUFFT_D2Z = 0x6a;     
    
    /**
     * CUFFT transform type Double-Complex to Double
     */
    public static final int CUFFT_Z2D = 0x6c;
    
    /**
     * CUFFT transform type Double-Complex to Double-Complex
     */
    public static final int CUFFT_Z2Z = 0x69; 
    

    /**
     * Returns the String identifying the given cufftType
     * 
     * @param m The cufftType
     * @return The String identifying the given cufftType
     */
    public static String stringFor(int m)
    {
        switch (m)
        {
            case CUFFT_R2C : return "CUFFT_R2C";
            case CUFFT_C2R : return "CUFFT_C2R";
            case CUFFT_C2C : return "CUFFT_C2C";
            case CUFFT_D2Z : return "CUFFT_D2Z";
            case CUFFT_Z2D : return "CUFFT_Z2D";
            case CUFFT_Z2Z : return "CUFFT_Z2Z";
        }
        return "INVALID cufftType: " + m;
    }
    
    /**
     * Private constructor to prevent instantiation.
     */
    private cufftType()
    {
    }
    
}
