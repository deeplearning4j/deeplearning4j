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
package jcuda.jcurand;

/**
 * CURAND choice of direction vector set
 */
public class curandDirectionVectorSet
{
    /**
     * Specific set of 32-bit direction vectors generated from polynomials 
     * recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions
     */
    public static final int CURAND_DIRECTION_VECTORS_32_JOEKUO6 = 101;
    /**
     * Specific set of 32-bit direction vectors generated from polynomials 
     * recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions, 
     * and scrambled
     */
    public static final int CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6 = 102;
    /**
     * Specific set of 64-bit direction vectors generated from polynomials 
     * recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions
     */
    public static final int CURAND_DIRECTION_VECTORS_64_JOEKUO6 = 103;
    /**
     * Specific set of 64-bit direction vectors generated from polynomials 
     * recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions, 
     * and scrambled
     */
    public static final int CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6 = 104;

    /**
     * Private constructor to prevent instantiation
     */
    private curandDirectionVectorSet(){}

    /**
     * Returns a string representation of the given constant
     *
     * @return A string representation of the given constant
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CURAND_DIRECTION_VECTORS_32_JOEKUO6: return "CURAND_DIRECTION_VECTORS_32_JOEKUO6";
            case CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6: return "CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6";
            case CURAND_DIRECTION_VECTORS_64_JOEKUO6: return "CURAND_DIRECTION_VECTORS_64_JOEKUO6";
            case CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6: return "CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6";
        }
        return "INVALID curandDirectionVectorSet";
    }
}

