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
package jcuda.jcusparse;

// used in csrsv2, csric02, and csrilu02
public class cusparseSolvePolicy
{
    /**
     * no level information is generated, only reports structural zero.
     */
    public static final int CUSPARSE_SOLVE_POLICY_NO_LEVEL = 0;
    public static final int CUSPARSE_SOLVE_POLICY_USE_LEVEL = 1;

    /**
     * Private constructor to prevent instantiation
     */
    private cusparseSolvePolicy(){}

    /**
     * Returns a string representation of the given constant
     *
     * @return A string representation of the given constant
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CUSPARSE_SOLVE_POLICY_NO_LEVEL: return "CUSPARSE_SOLVE_POLICY_NO_LEVEL";
            case CUSPARSE_SOLVE_POLICY_USE_LEVEL: return "CUSPARSE_SOLVE_POLICY_USE_LEVEL";
        }
        return "INVALID cusparseSolvePolicy: "+n;
    }
}

