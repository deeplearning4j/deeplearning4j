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
package jcuda.driver;

/**
 * GPU Direct v3 tokens. I don't have the slightest idea what this is.
 */
public class CUDA_POINTER_ATTRIBUTE_P2P_TOKENS 
{
    public long p2pToken;
    
    public int vaSpaceToken;
    
    /**
     * Creates a new, uninitialized CUDA_POINTER_ATTRIBUTE_P2P_TOKENS
     */
    public CUDA_POINTER_ATTRIBUTE_P2P_TOKENS()
    {
        
    }
    
    /**
     * Returns a String representation of this object.
     *
     * @return A String representation of this object.
     */
    @Override
    public String toString()
    {
        return "CUDA_POINTER_ATTRIBUTE_P2P_TOKENS["+createString(",")+"]";
    }

    /**
     * Creates and returns a formatted (aligned, multi-line) String
     * representation of this object
     *
     * @return A formatted String representation of this object
     */
    public String toFormattedString()
    {
        return "CUDA p2p tokensr:\n    "+createString("\n    ");
    }

    /**
     * Creates and returns a string representation of this object,
     * using the given separator for the fields
     *
     * @param f Separator
     * @return A String representation of this object
     */
    private String createString(String f)
    {
        StringBuilder sb = new StringBuilder();
        sb.append("p2pToken="+p2pToken+f);
        sb.append("vaSpaceToken="+vaSpaceToken+f);
        return sb.toString();
    }

    
}
