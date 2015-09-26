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

