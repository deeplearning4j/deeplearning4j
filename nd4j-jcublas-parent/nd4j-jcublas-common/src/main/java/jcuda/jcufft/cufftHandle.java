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
 * A handle type used to store and access CUFFT plans
 */
public class cufftHandle
{
    /** 
     * The plan id, written by native methods 
     */
    private int plan;

    /**
     * The dimension of this plan
     */
    private int dim = 0;
    
    /**
     * The cufftType of this plan (cufftType.CUFFT_R2C, cufftType.CUFFT_C2R or cufftType.CUFFT_C2C)
     */
    private int type;
    
    /**
     * The size of this plan
     */
    private int sizeX = 0;

    /**
     * The size of this plan
     */
    private int sizeY = 0;
    
    /**
     * The size of this plan
     */
    private int sizeZ = 0;
    
    /**
     * The batch size of this plan, for 1D transforms
     */
    private int batchSize = 0;
    
    /**
     * Returns a String representation of this JCufftHandle
     * 
     * @return A String representation of this JCufftHandle
     */
    public String toString()
    {
        if (dim == 0)
        {
            return "cufftHandle[uninitialized]";
        }
        String result = "cufftHandle[id="+plan+",dim="+dim+",type="+cufftType.stringFor(type)+", size=";
        switch (dim)
        {
            case 1:
                result += "("+sizeX+"), batch="+batchSize;
                break;

            case 2:
                result += "("+sizeX+","+sizeY+")";
                break;

            case 3:
                result += "("+sizeX+","+sizeY+","+sizeZ+")";
                break;
        }
        result += "]";
        return result;
    }

    /**
     * Set the batch size of this plan
     * 
     * @param batchSize The batch size of this plan 
     */
    void setBatchSize(int batchSize)
    {
        this.batchSize = batchSize;
    }
    
    /**
     * Set the type of this plan (JCufft.CUFFT_R2C, JCufft.CUFFT_C2R or JCufft.CUFFT_C2C)
     * @param type
     */
    void setType(int type)
    {
        this.type = type;
    }
    
    /**
     * Set the dimension of this plan (1,2 or 3)
     * 
     * @param dim The dimension of this plan
     */
    void setDimension(int dim)
    {
        this.dim = dim;
    }

    /**
     * Set the size of this plan
     * 
     * @param x Size in x
     * @param y Size in y
     * @param z Size in z
     */
    void setSize(int x, int y, int z)
    {
        this.sizeX = x;
        this.sizeY = y;
        this.sizeZ = z;
    }
    
}
