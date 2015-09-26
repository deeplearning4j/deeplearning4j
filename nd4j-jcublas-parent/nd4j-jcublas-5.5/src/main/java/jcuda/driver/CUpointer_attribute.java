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
 * Pointer information.<br />
 * <br />
 * Most comments are taken from the CUDA reference manual
 */
public class CUpointer_attribute
{
    /**
     * The ::CUcontext on which a pointer was allocated or registered
     */
    public static final int CU_POINTER_ATTRIBUTE_CONTEXT = 1;
    
    /** 
     * The ::CUmemorytype describing the physical location of a pointer 
     */
    public static final int CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2;    
    
    /** 
     * The address at which a pointer's memory may be accessed on the device 
     */
    public static final int CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 3;
    
    /** 
     * The address at which a pointer's memory may be accessed on the host 
     */
    public static final int CU_POINTER_ATTRIBUTE_HOST_POINTER = 4;   
    
    /** 
     * A pair of tokens for use with the nv-p2p.h Linux kernel interface 
     */
    public static final int CU_POINTER_ATTRIBUTE_P2P_TOKENS = 5;
    
    /** 
     * Synchronize every synchronous memory operation initiated on this region 
     */
    public static final int CU_POINTER_ATTRIBUTE_SYNC_MEMOPS = 6;
    
    /** 
     * A process-wide unique ID for an allocated memory region
     */
    public static final int CU_POINTER_ATTRIBUTE_BUFFER_ID = 7;
    
    /** 
     * Indicates if the pointer points to managed memory 
     */
    public static final int CU_POINTER_ATTRIBUTE_IS_MANAGED = 8;
    
    /**
     * Returns the String identifying the given CUpointer_attribute
     *
     * @param n The CUpointer_attribute
     * @return The String identifying the given CUpointer_attribute
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CU_POINTER_ATTRIBUTE_CONTEXT : return "CU_POINTER_ATTRIBUTE_CONTEXT";
            case CU_POINTER_ATTRIBUTE_MEMORY_TYPE : return "CU_POINTER_ATTRIBUTE_MEMORY_TYPE";
            case CU_POINTER_ATTRIBUTE_DEVICE_POINTER : return "CU_POINTER_ATTRIBUTE_DEVICE_POINTER";
            case CU_POINTER_ATTRIBUTE_HOST_POINTER : return "CU_POINTER_ATTRIBUTE_HOST_POINTER";
            case CU_POINTER_ATTRIBUTE_P2P_TOKENS : return "CU_POINTER_ATTRIBUTE_P2P_TOKENS";
            case CU_POINTER_ATTRIBUTE_SYNC_MEMOPS : return "CU_POINTER_ATTRIBUTE_SYNC_MEMOPS";
            case CU_POINTER_ATTRIBUTE_BUFFER_ID : return "CU_POINTER_ATTRIBUTE_BUFFER_ID";
            case CU_POINTER_ATTRIBUTE_IS_MANAGED : return "CU_POINTER_ATTRIBUTE_IS_MANAGED";
        }
        return "INVALID CUpointer_attribute: "+n;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private CUpointer_attribute()
    {
    }
    
}
