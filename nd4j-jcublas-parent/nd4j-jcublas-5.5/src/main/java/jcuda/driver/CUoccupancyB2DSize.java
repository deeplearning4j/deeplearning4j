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
 * Java port of the function that is passed to 
 * {@link JCudaDriver#cuOccupancyMaxPotentialBlockSize} and maps a
 * certain kernel block size to the size of the per-block dynamic
 * shared memory
 */
public interface CUoccupancyB2DSize 
{
    /**
     * Returns the size of the dynamic shared memory for the given kernel
     * block size
     * 
     * @param blockSize The kernel block size
     * @return The shared memory size
     */
    long call(int blockSize); 
}