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

import jcuda.NativePointerObject;

/**
 * Java port of a textureReference.<br />
 * <br />
 * Most comments are taken from the CUDA reference manual or CUDA
 * programming guide.
 */
public class textureReference extends NativePointerObject
{
    /**
     * Specifies whether texture coordinates are normalized or not. If it is
     * non-zero, all elements in the texture are addressed with texture coordinates in
     * the range [0,1] rather than in the range [0,width-1], [0,height-1], or
     * [0,depth-1] where width, height, and depth are the texture sizes;
     */
    public int normalized;

    /**
     * Specifies the filtering mode, that is how the value returned when
     * fetching the texture is computed based on the input texture coordinates.
     * filterMode is equal to cudaFilterModePoint or
     * cudaFilterModeLinear; if it is cudaFilterModePoint, the returned
     * value is the texel whose texture coordinates are the closest to the input texture
     * coordinates; if it is cudaFilterModeLinear, the returned value is the linear
     * interpolation of the two (for a one-dimensional texture), four (for a
     * two-dimensional texture), or eight (for a three-dimensional texture) texels whose
     * texture coordinates are the closest to the input texture coordinates;
     * cudaFilterModeLinear is only valid for returned values of floating-point
     * type;
     *
     * @see cudaTextureFilterMode
     */
    public int filterMode;

    /**
     * Specifies the addressing mode, that is how out-of-range texture
     * coordinates are handled. addressMode is an array of size three whose first,
     * second, and third elements specify the addressing mode for the first, second, and
     * third texture coordinates, respectively; the addressing mode is equal to either
     * cudaAddressModeClamp, in which case out-of-range texture coordinates are
     * clamped to the valid range, or cudaAddressModeWrap, in which case out-of range
     * texture coordinates are wrapped to the valid range;
     * cudaAddressModeWrap is only supported for normalized texture coordinates;
     *
     * @see cudaTextureAddressMode
     */
    public int addressMode[] = new int[3];

    /**
     * Describes the format of the value that is returned when fetching
     * the texture.
     *
     * @see cudaChannelFormatDesc
     */
    public cudaChannelFormatDesc channelDesc;
    
    /**
     * Perform sRGB->linear conversion during texture read
     */
    public int sRGB;
    
    /**
     * Limit to the anisotropy ratio
     */
    public int maxAnisotropy;
    
    /**
     * Mipmap filter mode
     * 
     * @see cudaTextureFilterMode
     */
    public int mipmapFilterMode;
    
    /**
     * Offset applied to the supplied mipmap level
     */
    public float mipmapLevelBias;
    
    /**
     * Lower end of the mipmap level range to clamp access to
     */
    public float minMipmapLevelClamp;

    /**
     * Upper end of the mipmap level range to clamp access to
     */
    public float maxMipmapLevelClamp;
    
    /**
     * Creates a new, uninitialized textureReference
     */
    public textureReference()
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
        return "textureReference[" +
            "nativePointer=0x"+Long.toHexString(getNativePointer())+","+
            "normalized="+normalized+","+
            "filterMode="+cudaTextureFilterMode.stringFor(filterMode)+","+
            "addressMode=["+
                cudaTextureAddressMode.stringFor(addressMode[0])+","+
                cudaTextureAddressMode.stringFor(addressMode[1])+","+
                cudaTextureAddressMode.stringFor(addressMode[2])+"]"+","+
            "channelDesc="+channelDesc+","+
            "sRGB="+sRGB+","+
            "maxAnisotropy="+maxAnisotropy+","+
            "mipmapFilterMode="+cudaTextureFilterMode.stringFor(mipmapFilterMode)+","+
            "mipmapLevelBias="+mipmapLevelBias+","+
            "minMipmapLevelClamp="+minMipmapLevelClamp+","+
            "maxMipmapLevelClamp="+maxMipmapLevelClamp+"]";
    }
    
}
