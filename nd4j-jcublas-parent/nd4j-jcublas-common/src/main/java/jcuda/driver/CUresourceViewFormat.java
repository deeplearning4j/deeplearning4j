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
 * Resource view format
 */
public class CUresourceViewFormat
{
    /**
     * No resource view format (use underlying resource format) 
     */
    public static final int CU_RES_VIEW_FORMAT_NONE          = 0x00;
    
    /**
     * 1 channel unsigned 8-bit integers 
     */
    public static final int CU_RES_VIEW_FORMAT_UINT_1X8      = 0x01;
    
    /**
     * 2 channel unsigned 8-bit integers 
     */
    public static final int CU_RES_VIEW_FORMAT_UINT_2X8      = 0x02;
    
    /**
     * 4 channel unsigned 8-bit integers 
     */
    public static final int CU_RES_VIEW_FORMAT_UINT_4X8      = 0x03;
    
    /**
     * 1 channel signed 8-bit integers 
     */
    public static final int CU_RES_VIEW_FORMAT_SINT_1X8      = 0x04;
    
    /**
     * 2 channel signed 8-bit integers 
     */
    public static final int CU_RES_VIEW_FORMAT_SINT_2X8      = 0x05;
    
    /**
     * 4 channel signed 8-bit integers 
     */
    public static final int CU_RES_VIEW_FORMAT_SINT_4X8      = 0x06;
    
    /**
     * 1 channel unsigned 16-bit integers 
     */
    public static final int CU_RES_VIEW_FORMAT_UINT_1X16     = 0x07;
    
    /**
     * 2 channel unsigned 16-bit integers 
     */
    public static final int CU_RES_VIEW_FORMAT_UINT_2X16     = 0x08;
    
    /**
     * 4 channel unsigned 16-bit integers 
     */
    public static final int CU_RES_VIEW_FORMAT_UINT_4X16     = 0x09;
    
    /**
     * 1 channel signed 16-bit integers 
     */
    public static final int CU_RES_VIEW_FORMAT_SINT_1X16     = 0x0a;
    
    /**
     * 2 channel signed 16-bit integers 
     */
    public static final int CU_RES_VIEW_FORMAT_SINT_2X16     = 0x0b;
    
    /**
     * 4 channel signed 16-bit integers 
     */
    public static final int CU_RES_VIEW_FORMAT_SINT_4X16     = 0x0c;
    
    /**
     * 1 channel unsigned 32-bit integers 
     */
    public static final int CU_RES_VIEW_FORMAT_UINT_1X32     = 0x0d;
    
    /**
     * 2 channel unsigned 32-bit integers 
     */
    public static final int CU_RES_VIEW_FORMAT_UINT_2X32     = 0x0e;
    
    /**
     * 4 channel unsigned 32-bit integers 
     */
    public static final int CU_RES_VIEW_FORMAT_UINT_4X32     = 0x0f;
    
    /**
     * 1 channel signed 32-bit integers 
     */
    public static final int CU_RES_VIEW_FORMAT_SINT_1X32     = 0x10;
    
    /**
     * 2 channel signed 32-bit integers 
     */
    public static final int CU_RES_VIEW_FORMAT_SINT_2X32     = 0x11;
    
    /**
     * 4 channel signed 32-bit integers 
     */
    public static final int CU_RES_VIEW_FORMAT_SINT_4X32     = 0x12;
    
    /**
     * 1 channel 16-bit floating point 
     */
    public static final int CU_RES_VIEW_FORMAT_FLOAT_1X16    = 0x13;
    
    /**
     * 2 channel 16-bit floating point 
     */
    public static final int CU_RES_VIEW_FORMAT_FLOAT_2X16    = 0x14;
    
    /**
     * 4 channel 16-bit floating point 
     */
    public static final int CU_RES_VIEW_FORMAT_FLOAT_4X16    = 0x15;
    
    /**
     * 1 channel 32-bit floating point 
     */
    public static final int CU_RES_VIEW_FORMAT_FLOAT_1X32    = 0x16;
    
    /**
     * 2 channel 32-bit floating point 
     */
    public static final int CU_RES_VIEW_FORMAT_FLOAT_2X32    = 0x17;
    
    /**
     * 4 channel 32-bit floating point 
     */
    public static final int CU_RES_VIEW_FORMAT_FLOAT_4X32    = 0x18;
    
    /**
     * Block compressed 1 
     */
    public static final int CU_RES_VIEW_FORMAT_UNSIGNED_BC1  = 0x19;
    
    /**
     * Block compressed 2 
     */
    public static final int CU_RES_VIEW_FORMAT_UNSIGNED_BC2  = 0x1a;
    
    /**
     * Block compressed 3 
     */
    public static final int CU_RES_VIEW_FORMAT_UNSIGNED_BC3  = 0x1b;
    
    /**
     * Block compressed 4 unsigned 
     */
    public static final int CU_RES_VIEW_FORMAT_UNSIGNED_BC4  = 0x1c;
    
    /**
     * Block compressed 4 signed 
     */
    public static final int CU_RES_VIEW_FORMAT_SIGNED_BC4    = 0x1d;
    
    /**
     * Block compressed 5 unsigned 
     */
    public static final int CU_RES_VIEW_FORMAT_UNSIGNED_BC5  = 0x1e;
    
    /**
     * Block compressed 5 signed 
     */
    public static final int CU_RES_VIEW_FORMAT_SIGNED_BC5    = 0x1f;
    
    /**
     * Block compressed 6 unsigned half-float 
     */
    public static final int CU_RES_VIEW_FORMAT_UNSIGNED_BC6H = 0x20;
    
    /**
     * Block compressed 6 signed half-float 
     */
    public static final int CU_RES_VIEW_FORMAT_SIGNED_BC6H   = 0x21;
    
    /**
     * Block compressed 7 
     */
    public static final int CU_RES_VIEW_FORMAT_UNSIGNED_BC7  = 0x22;
    
    
    /**
     * Returns the String identifying the given CUresourceViewFormat
     *
     * @param n The CUresourceViewFormat
     * @return The String identifying the given CUresourceViewFormat
     */
    public static String stringFor(int n)
    {
        switch (n) 
        {
            case CU_RES_VIEW_FORMAT_NONE          : return"CU_RES_VIEW_FORMAT_NONE";
            case CU_RES_VIEW_FORMAT_UINT_1X8      : return"CU_RES_VIEW_FORMAT_UINT_1X8";
            case CU_RES_VIEW_FORMAT_UINT_2X8      : return"CU_RES_VIEW_FORMAT_UINT_2X8";
            case CU_RES_VIEW_FORMAT_UINT_4X8      : return"CU_RES_VIEW_FORMAT_UINT_4X8";
            case CU_RES_VIEW_FORMAT_SINT_1X8      : return"CU_RES_VIEW_FORMAT_SINT_1X8";
            case CU_RES_VIEW_FORMAT_SINT_2X8      : return"CU_RES_VIEW_FORMAT_SINT_2X8";
            case CU_RES_VIEW_FORMAT_SINT_4X8      : return"CU_RES_VIEW_FORMAT_SINT_4X8";
            case CU_RES_VIEW_FORMAT_UINT_1X16     : return"CU_RES_VIEW_FORMAT_UINT_1X16";
            case CU_RES_VIEW_FORMAT_UINT_2X16     : return"CU_RES_VIEW_FORMAT_UINT_2X16";
            case CU_RES_VIEW_FORMAT_UINT_4X16     : return"CU_RES_VIEW_FORMAT_UINT_4X16";
            case CU_RES_VIEW_FORMAT_SINT_1X16     : return"CU_RES_VIEW_FORMAT_SINT_1X16";
            case CU_RES_VIEW_FORMAT_SINT_2X16     : return"CU_RES_VIEW_FORMAT_SINT_2X16";
            case CU_RES_VIEW_FORMAT_SINT_4X16     : return"CU_RES_VIEW_FORMAT_SINT_4X16";
            case CU_RES_VIEW_FORMAT_UINT_1X32     : return"CU_RES_VIEW_FORMAT_UINT_1X32";
            case CU_RES_VIEW_FORMAT_UINT_2X32     : return"CU_RES_VIEW_FORMAT_UINT_2X32";
            case CU_RES_VIEW_FORMAT_UINT_4X32     : return"CU_RES_VIEW_FORMAT_UINT_4X32";
            case CU_RES_VIEW_FORMAT_SINT_1X32     : return"CU_RES_VIEW_FORMAT_SINT_1X32";
            case CU_RES_VIEW_FORMAT_SINT_2X32     : return"CU_RES_VIEW_FORMAT_SINT_2X32";
            case CU_RES_VIEW_FORMAT_SINT_4X32     : return"CU_RES_VIEW_FORMAT_SINT_4X32";
            case CU_RES_VIEW_FORMAT_FLOAT_1X16    : return"CU_RES_VIEW_FORMAT_FLOAT_1X16";
            case CU_RES_VIEW_FORMAT_FLOAT_2X16    : return"CU_RES_VIEW_FORMAT_FLOAT_2X16";
            case CU_RES_VIEW_FORMAT_FLOAT_4X16    : return"CU_RES_VIEW_FORMAT_FLOAT_4X16";
            case CU_RES_VIEW_FORMAT_FLOAT_1X32    : return"CU_RES_VIEW_FORMAT_FLOAT_1X32";
            case CU_RES_VIEW_FORMAT_FLOAT_2X32    : return"CU_RES_VIEW_FORMAT_FLOAT_2X32";
            case CU_RES_VIEW_FORMAT_FLOAT_4X32    : return"CU_RES_VIEW_FORMAT_FLOAT_4X32";
            case CU_RES_VIEW_FORMAT_UNSIGNED_BC1  : return"CU_RES_VIEW_FORMAT_UNSIGNED_BC1";
            case CU_RES_VIEW_FORMAT_UNSIGNED_BC2  : return"CU_RES_VIEW_FORMAT_UNSIGNED_BC2";
            case CU_RES_VIEW_FORMAT_UNSIGNED_BC3  : return"CU_RES_VIEW_FORMAT_UNSIGNED_BC3";
            case CU_RES_VIEW_FORMAT_UNSIGNED_BC4  : return"CU_RES_VIEW_FORMAT_UNSIGNED_BC4";
            case CU_RES_VIEW_FORMAT_SIGNED_BC4    : return"CU_RES_VIEW_FORMAT_SIGNED_BC4";
            case CU_RES_VIEW_FORMAT_UNSIGNED_BC5  : return"CU_RES_VIEW_FORMAT_UNSIGNED_BC5";
            case CU_RES_VIEW_FORMAT_SIGNED_BC5    : return"CU_RES_VIEW_FORMAT_SIGNED_BC5";
            case CU_RES_VIEW_FORMAT_UNSIGNED_BC6H : return"CU_RES_VIEW_FORMAT_UNSIGNED_BC6H";
            case CU_RES_VIEW_FORMAT_SIGNED_BC6H   : return"CU_RES_VIEW_FORMAT_SIGNED_BC6H";
            case CU_RES_VIEW_FORMAT_UNSIGNED_BC7  : return"CU_RES_VIEW_FORMAT_UNSIGNED_BC7";
        }
        return "INVALID CUresourceViewFormat: "+n;
    }
    
    /**
     * Private constructor to prevent instantiation.
     */
    private CUresourceViewFormat()
    {
    }

}
