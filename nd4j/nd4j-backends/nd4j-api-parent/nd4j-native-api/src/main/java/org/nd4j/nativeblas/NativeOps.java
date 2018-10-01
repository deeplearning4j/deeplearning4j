/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.nativeblas;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.Cast;


/**
 * Native interface for
 * op execution on cpu
 * @author Adam Gibson
 */
public abstract class NativeOps extends Pointer {
    public NativeOps(Pointer p) {
        super(p);
    }

    public static int getCores(int totals) {
        // that's special case for Xeon Phi
        if (totals >= 256)
            return 64;

        int ht_off = totals / 2; // we count off HyperThreading without any excuses
        if (ht_off <= 4)
            return 4; // special case for Intel i5. and nobody likes i3 anyway

        if (ht_off > 24) {
            int rounds = 0;
            while (ht_off > 24) { // we loop until final value gets below 24 cores, since that's reasonable threshold as of 2016
                if (ht_off > 24) {
                    ht_off /= 2; // we dont' have any cpus that has higher number then 24 physical cores
                    rounds++;
                }
            }
            // 20 threads is special case in this branch
            if (ht_off == 20 && rounds < 2)
                ht_off /= 2;
        } else { // low-core models are known, but there's a gap, between consumer cpus and xeons
            if (ht_off <= 6) {
                // that's more likely consumer-grade cpu, so leave this value alone
                return ht_off;
            } else {
                if (isOdd(ht_off)) // if that's odd number, it's final result
                    return ht_off;

                // 20 threads & 16 threads are special case in this branch, where we go min value
                if (ht_off == 20 || ht_off == 16)
                    ht_off /= 2;
            }
        }
        return ht_off;
    }

    private static boolean isOdd(int value) {
        return (value % 2 != 0);
    }

    /**
     * This method allows you to specify minimal number of elements per thread/block during op call
     * PLEASE NOTE: Changing this value might and will affect performance.
     *
     * @param value
     */
    public native void setElementThreshold(int value);

    /**
     * This method allows you to specify minimal number of TADs per thread/block during op call
     * PLEASE NOTE: Changing this value might and will affect performance.
     *
     * @param value
     */
    public abstract void setTADThreshold(int value);

    /**
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     */
    public abstract void execIndexReduceScalar(PointerPointer extraPointers, int opNum, Pointer x,
                                                 @Cast("Nd4jLong *") LongPointer xShapeInfo, Pointer extraParams, Pointer z,
                                                 @Cast("Nd4jLong *") LongPointer zShapeInfo);

    /**
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfoBuffer
     * @param dimension
     * @param dimensionLength
     */
    public abstract void execIndexReduce(PointerPointer extraPointers, int opNum, Pointer x,
                                         @Cast("Nd4jLong *") LongPointer xShapeInfo, Pointer extraParams, Pointer result,
                                         @Cast("Nd4jLong *") LongPointer resultShapeInfoBuffer, IntPointer dimension, int dimensionLength);

    /**
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param y
     * @param yShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param dimension
     * @param dimensionLength
     */
    public abstract void execBroadcast(PointerPointer extraPointers, int opNum, Pointer x,
                                       @Cast("Nd4jLong *") LongPointer xShapeInfo, Pointer y, @Cast("Nd4jLong *") LongPointer yShapeInfo, Pointer result,
                                       @Cast("Nd4jLong *") LongPointer resultShapeInfo, IntPointer dimension, int dimensionLength);


    /**
     * @param opNum
     * @param dx
     * @param xShapeInfo
     * @param y
     * @param yShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param extraParams
     */
    public abstract void execPairwiseTransform(PointerPointer extraPointers, int opNum, Pointer dx,
                                               @Cast("Nd4jLong *") LongPointer xShapeInfo, Pointer y, @Cast("Nd4jLong *") LongPointer yShapeInfo, Pointer result,
                                               @Cast("Nd4jLong *") LongPointer resultShapeInfo, Pointer extraParams);

    /**
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    public abstract void execReduceFloat(PointerPointer extraPointers, int opNum, Pointer x, @Cast("Nd4jLong *") LongPointer xShapeInfo, Pointer extraParams, Pointer result, @Cast("Nd4jLong *") LongPointer resultShapeInfo);

    public abstract void execReduceSame(PointerPointer extraPointers, int opNum, Pointer x, @Cast("Nd4jLong *") LongPointer xShapeInfo, Pointer extraParams, Pointer result, @Cast("Nd4jLong *") LongPointer resultShapeInfo);

    public abstract void execReduceBool(PointerPointer extraPointers, int opNum, Pointer x, @Cast("Nd4jLong *") LongPointer xShapeInfo, Pointer extraParams, Pointer result, @Cast("Nd4jLong *") LongPointer resultShapeInfo);

    public abstract void execReduceLong(PointerPointer extraPointers, int opNum, Pointer x, @Cast("Nd4jLong *") LongPointer xShapeInfo, Pointer extraParams, Pointer result, @Cast("Nd4jLong *") LongPointer resultShapeInfo);

    /**
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    public abstract void execReduceFloat(PointerPointer extraPointers, int opNum, Pointer x, @Cast("Nd4jLong *") LongPointer xShapeInfo, Pointer extraParams, Pointer result, @Cast("Nd4jLong *") LongPointer resultShapeInfo, IntPointer dimension, int dimensionLength);

    public abstract void execReduceSame(PointerPointer extraPointers, int opNum, Pointer x, @Cast("Nd4jLong *") LongPointer xShapeInfo, Pointer extraParams, Pointer result, @Cast("Nd4jLong *") LongPointer resultShapeInfo, IntPointer dimension, int dimensionLength);

    public abstract void execReduceBool(PointerPointer extraPointers, int opNum, Pointer x, @Cast("Nd4jLong *") LongPointer xShapeInfo, Pointer extraParams, Pointer result, @Cast("Nd4jLong *") LongPointer resultShapeInfo, IntPointer dimension, int dimensionLength);

    public abstract void execReduceLong(PointerPointer extraPointers, int opNum, Pointer x, @Cast("Nd4jLong *") LongPointer xShapeInfo, Pointer extraParams, Pointer result, @Cast("Nd4jLong *") LongPointer resultShapeInfo, IntPointer dimension, int dimensionLength);

    /**
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParamsVals
     * @param y
     * @param yShapeInfo
     * @param result
     * @param resultShapeInfo
     */
    public abstract void execReduce3(PointerPointer extraPointers, int opNum, Pointer x,
                                     @Cast("Nd4jLong *") LongPointer xShapeInfo, Pointer extraParamsVals, Pointer y, @Cast("Nd4jLong *") LongPointer yShapeInfo,
                                     Pointer result, @Cast("Nd4jLong *") LongPointer resultShapeInfo);

    /**
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParamsVals
     * @param y
     * @param yShapeInfo
     */
    public abstract void execReduce3Scalar(PointerPointer extraPointers, int opNum, Pointer x,
                                           @Cast("Nd4jLong *") LongPointer xShapeInfo, Pointer extraParamsVals, Pointer y, @Cast("Nd4jLong *") LongPointer yShapeInfo, Pointer z, @Cast("Nd4jLong *") LongPointer zShapeInfo);

    /**
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParamsVals
     * @param y
     * @param yShapeInfo
     * @param result
     * @param resultShapeInfoBuffer
     * @param dimension
     * @param dimensionLength
     */
    public abstract void execReduce3(PointerPointer extraPointers, int opNum, Pointer x,
                                     @Cast("Nd4jLong *") LongPointer xShapeInfo, Pointer extraParamsVals, Pointer y, @Cast("Nd4jLong *") LongPointer yShapeInfo,
                                     Pointer result, @Cast("Nd4jLong *") LongPointer resultShapeInfoBuffer, IntPointer dimension, int dimensionLength);

    public abstract void execReduce3All(PointerPointer extraPointers, int opNum, Pointer x,
                                        @Cast("Nd4jLong *") LongPointer xShapeInfo, Pointer extraParamsVals, Pointer y, @Cast("Nd4jLong *") LongPointer yShapeInfo,
                                        Pointer result, @Cast("Nd4jLong *") LongPointer resultShapeInfoBuffer, IntPointer dimension, int dimensionLength,
                                        @Cast("Nd4jLong *") LongPointer xTadShape, @Cast("Nd4jLong *") LongPointer xOffsets, @Cast("Nd4jLong *") LongPointer yTadShape,
                                        @Cast("Nd4jLong *") LongPointer yOffsets);


    /**
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param scalar
     * @param extraParams
     */
    public abstract void execScalar(PointerPointer extraPointers, int opNum, Pointer x,
                                    @Cast("Nd4jLong *") LongPointer xShapeInfo, Pointer result, @Cast("Nd4jLong *") LongPointer resultShapeInfo, Pointer scalar, LongPointer scalarShapeInfo,Pointer extraParams);

    /**
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param biasCorrected
     */
    public abstract void execSummaryStatsScalar(PointerPointer extraPointers, int opNum, Pointer x,
                                                  @Cast("Nd4jLong *") LongPointer xShapeInfo, Pointer extraParams, Pointer z, @Cast("Nd4jLong *") LongPointer zShapeInfo, boolean biasCorrected);

    /**
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     * @param biasCorrected
     */
    public abstract void execSummaryStats(PointerPointer extraPointers, int opNum, Pointer x,
                                          @Cast("Nd4jLong *") LongPointer xShapeInfo, Pointer extraParams, Pointer result, @Cast("Nd4jLong *") LongPointer resultShapeInfo,
                                          boolean biasCorrected);

    /**
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfoBuffer
     * @param dimension
     * @param dimensionLength
     */
    public abstract void execSummaryStats(PointerPointer extraPointers, int opNum, Pointer x,
                                          @Cast("Nd4jLong *") LongPointer xShapeInfo, Pointer extraParams, Pointer result,
                                          @Cast("Nd4jLong *") LongPointer resultShapeInfoBuffer, IntPointer dimension, int dimensionLength, boolean biasCorrected);


    /**
     * @param extraPointers
     * @param opNum
     * @param dx
     * @param xShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param extraParams
     */
    public abstract void execTransformFloat(PointerPointer extraPointers, int opNum, Pointer dx, @Cast("Nd4jLong *") LongPointer xShapeInfo, Pointer result, @Cast("Nd4jLong *") LongPointer resultShapeInfo, Pointer extraParams);

    public abstract void execTransformSame(PointerPointer extraPointers, int opNum, Pointer dx, @Cast("Nd4jLong *") LongPointer xShapeInfo, Pointer result, @Cast("Nd4jLong *") LongPointer resultShapeInfo, Pointer extraParams);

    public abstract void execTransformStrict(PointerPointer extraPointers, int opNum, Pointer dx, @Cast("Nd4jLong *") LongPointer xShapeInfo, Pointer result, @Cast("Nd4jLong *") LongPointer resultShapeInfo, Pointer extraParams);

    public abstract void execTransformBool(PointerPointer extraPointers, int opNum, Pointer dx, @Cast("Nd4jLong *") LongPointer xShapeInfo, Pointer result, @Cast("Nd4jLong *") LongPointer resultShapeInfo, Pointer extraParams);


    /**
     * ScalarOp along dimension
     *
     * @param extraPointers   pointers to tadShapes and tadoffsets
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param z
     * @param zShapeInfo
     * @param scalars
     * @param extraParams
     * @param dimension
     * @param dimensionLength
     */
    public abstract void execScalar(PointerPointer extraPointers, int opNum, Pointer x,
                                    @Cast("Nd4jLong *") LongPointer xShapeInfo, Pointer z, @Cast("Nd4jLong *") LongPointer zShapeInfo, Pointer scalars, @Cast("Nd4jLong *") LongPointer scalarShapeInfo,
                                    Pointer extraParams, IntPointer dimension, int dimensionLength);

    /**
     * @param extraPointers
     * @param offset
     * @param order
     * @param results
     * @param resultShapeInfo
     * @param input
     * @param inputShapeInfo
     */
    public abstract void flatten(PointerPointer extraPointers, int offset, char order, Pointer results,
                                 @Cast("Nd4jLong *") LongPointer resultShapeInfo, Pointer input, @Cast("Nd4jLong *") LongPointer inputShapeInfo);

    /**
     * @param extraPointers
     * @param dimension
     * @param numArrays
     * @param data
     * @param inputShapeInfo
     * @param results
     * @param resultShapeInfo
     * @param tadPointers
     * @param tadOffsets
     */
    public abstract void concat(PointerPointer extraPointers, int dimension, int numArrays, PointerPointer data,
                                      PointerPointer inputShapeInfo, Pointer results, @Cast("Nd4jLong *") LongPointer resultShapeInfo,
                                      PointerPointer tadPointers, PointerPointer tadOffsets);

    public abstract void specialConcat(PointerPointer extraPointers, int dimension, int numArrays,
                                             PointerPointer data, PointerPointer inputShapeInfo, Pointer results,
                                             @Cast("Nd4jLong *") LongPointer resultShapeInfo, PointerPointer tadPointers, PointerPointer tadOffsets);


    /**
     * Gets the maximum number of open mp threads
     *
     * @return
     */
    public abstract int ompGetMaxThreads();

    /**
     * Gets the number of open mp threads
     *
     * @return
     */
    public abstract int ompGetNumThreads();

    /**
     * Sets the number of openmp threads
     *
     * @param threads
     */
    public abstract void setOmpNumThreads(int threads);

    /**
     * Sets the minimal number of openmp threads for variative methods
     *
     * @param threads
     */
    public abstract void setOmpMinThreads(int threads);

    /**
     * NEVER EVER USE THIS METHOD OUTSIDE OF  CUDA
     */
    public abstract void initializeDevicesAndFunctions();

    public abstract void initializeFunctions(PointerPointer functions);

    public abstract Pointer mallocHost(long memorySize, int flags);

    public abstract Pointer mallocDevice(long memorySize, Pointer ptrToDeviceId, int flags);

    public abstract int freeHost(Pointer pointer);

    public abstract int freeDevice(Pointer pointer, Pointer deviceId);

    public abstract Pointer createContext();

    public abstract Pointer createStream();

    public abstract Pointer createEvent();

    public abstract int registerEvent(Pointer event, Pointer stream);

    public abstract int destroyEvent(Pointer event);

    public abstract int setDevice(Pointer ptrToDeviceId);

    public abstract int getDevice();

    public abstract int streamSynchronize(Pointer stream);

    public abstract int eventSynchronize(Pointer event);

    public abstract long getDeviceFreeMemory(Pointer ptrToDeviceId);

    public abstract long getDeviceTotalMemory(Pointer ptrToDeviceId);

    public abstract int getDeviceMajor(Pointer ptrToDeviceId);

    public abstract int getDeviceMinor(Pointer ptrToDeviceId);

    public abstract String getDeviceName(Pointer ptrToDeviceId);

    public abstract int memcpy(Pointer dst, Pointer src, long size, int flags, Pointer reserved);

    public abstract int memcpyAsync(Pointer dst, Pointer src, long size, int flags, Pointer reserved);

    public abstract int memcpyConstantAsync(long dst, Pointer src, long size, int flags, Pointer reserved);

    public abstract int memset(Pointer dst, int value, long size, int flags, Pointer reserved);

    public abstract int memsetAsync(Pointer dst, int value, long size, int flags, Pointer reserved);

    public abstract Pointer getConstantSpace();

    public abstract int getAvailableDevices();

    public abstract void enableDebugMode(boolean reallyEnable);

    public abstract void enableVerboseMode(boolean reallyEnable);

    public abstract void setGridLimit(int gridSize);

    public abstract void tadOnlyShapeInfo(@Cast("Nd4jLong *") LongPointer shapeInfo, IntPointer dimension, int dimensionLength,
                                          @Cast("Nd4jLong *") LongPointer targetBuffer, @Cast("Nd4jLong *") LongPointer offsetsBuffer);

    ///////////////

    public abstract void pullRows(PointerPointer extraPointers, Pointer x, @Cast("Nd4jLong *") LongPointer xShapeInfo,
                                        Pointer z, @Cast("Nd4jLong *") LongPointer zShapeInfo, long n, @Cast("Nd4jLong *") LongPointer indexes, @Cast("Nd4jLong *") LongPointer tadShapeInfo,
                                        @Cast("Nd4jLong *") LongPointer tadOffsets, @Cast("Nd4jLong *") LongPointer zTadShapeInfo,
                                        @Cast("Nd4jLong *") LongPointer zTadOffsets);


    ///////////////////////

    public abstract void average(PointerPointer extraPointers, PointerPointer x, Pointer z, LongPointer zShapeInfo, int n, long length, boolean propagate);

    ///////////////////////

    public abstract void accumulate(PointerPointer extraPointers, PointerPointer x, Pointer z, LongPointer zShapeInfo, int n, long length);

    ///////////////////////

    public abstract void enableP2P(boolean reallyEnable);

    public abstract void checkP2P();

    public abstract boolean isP2PAvailable();

    //

    public abstract void shuffle(PointerPointer extraPointers, PointerPointer x, PointerPointer xShapeInfo,
                                 PointerPointer z, PointerPointer zShapeInfo, int N, IntPointer shuffleMap,
                                 PointerPointer tadShapeInfo, PointerPointer tadOffsets);


    // opType conversion

    public abstract void convertTypes(PointerPointer extras, int srcType, Pointer x, long N, int dstType, Pointer z);

    public abstract boolean isExperimentalEnabled();

    // GridOps


    // MetaOps
    public abstract void execMetaPredicateShape(PointerPointer extras, int opTypeA, int opNumA, int opTypeB,
                                                int opNumB, long N, Pointer dx, @Cast("Nd4jLong *") LongPointer xShape, Pointer dy, @Cast("Nd4jLong *") LongPointer yShape,
                                                Pointer dz, @Cast("Nd4jLong *") LongPointer zShape, Pointer extraA, Pointer extraB, double scalarA,
                                                double scalarB);


    /////////////////////////

    public abstract void execAggregate(PointerPointer extras, int opNum,
                                       PointerPointer arguments,
                                       int numArguments,
                                       @Cast("Nd4jLong **") PointerPointer shapes,
                                       int numShapes,
                                       IntPointer indexArguments,
                                       int numIndexArguments,
                                       @Cast("int **") PointerPointer intArrays,
                                       int numIntArrays,
                                       Pointer realArguments,
                                       int numRealArguments,
                                       @Cast("nd4j::DataType") int dataType);

    public abstract void execAggregateBatch(PointerPointer extras, int numAggregates, int opNum, int maxArgs,
                                            int maxShapes, int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals,
                                            Pointer ptrToArguments, @Cast("nd4j::DataType") int dataType);


    //////////////
    public abstract void execRandom(PointerPointer extraPointers, int opNum, Pointer state, Pointer z,
                                    @Cast("Nd4jLong *") LongPointer zShapeBuffer, Pointer extraArguments);

    public abstract void execRandom(PointerPointer extraPointers, int opNum, Pointer state, Pointer x,
                                    @Cast("Nd4jLong *") LongPointer xShapeBuffer, Pointer y, @Cast("Nd4jLong *") LongPointer yShapeBuffer, Pointer z,
                                    @Cast("Nd4jLong *") LongPointer zShapeBuffer, Pointer extraArguments);

    public abstract void execRandom(PointerPointer extraPointers, int opNum, Pointer state, Pointer x,
                                    @Cast("Nd4jLong *") LongPointer xShapeBuffer, Pointer z, @Cast("Nd4jLong *") LongPointer zShapeBuffer, Pointer extraArguments);

    ////////////////////


    public abstract Pointer initRandom(PointerPointer extraPointers, long seed, long numberOfElements, Pointer pointerToBuffer);

    public abstract void refreshBuffer(PointerPointer extraPointers, long seed, Pointer pointer);

    public abstract void reSeedBuffer(PointerPointer extraPointers, long seed, Pointer pointer);

    public abstract void destroyRandom(Pointer pointer);


    /**
     * Create a numpy array from an nd4j
     * array
     *
     * @param data        a pointer to the data
     * @param shapeBuffer the shapebuffer for the nd4j array
     * @param wordSize    the word size (4 for float, 8 for doubles)
     * @return a pointer to a numpy array
     */
    public abstract Pointer numpyFromNd4j(Pointer data, Pointer shapeBuffer, long wordSize);


    /**
     * Get the element size for a numpy array
     *
     * @param npyArray the numpy array's address
     *                 to get the length for
     * @return
     */
    public abstract int elementSizeForNpyArrayHeader(Pointer npyArray);


    /**
     * @param npyArrayStruct
     * @return
     */
    public abstract Pointer dataPointForNumpyStruct(Pointer npyArrayStruct);


    /**
     * Creates a numpy header for nd4j
     *
     * @param data        the data to use
     * @param shapeBuffer the shape buffer for the array
     * @param wordSize    the word size
     * @return
     */
    public abstract Pointer numpyHeaderForNd4j(Pointer data, Pointer shapeBuffer, long wordSize, LongPointer length);

    /**
     * Load numpy from a header
     * based on the cnpy parse from header method.
     *
     * @param data the header data to parse
     * @return a pointer to a numpy cnpy:NpyArray struct
     */
    public abstract Pointer loadNpyFromHeader(Pointer data);

    /**
     * @param npyArray
     * @return
     */
    public abstract Pointer dataPointForNumpyHeader(Pointer npyArray);

    /**
     * Get the shape buffer from a
     * numpy array.
     * **Warning** this allocates memory
     *
     * @param npyArray
     * @return
     */
    public abstract Pointer shapeBufferForNumpyHeader(Pointer npyArray);

    /**
     * Used in {@link org.nd4j.linalg.factory.NDArrayFactory#createFromNpyPointer(Pointer)}
     * to allow reuse of an in memory numpy buffer.
     * This is heavily used for python interop
     *
     * @param npyArray the pointer to the numpy array to use
     * @return the pointer for the numpy array
     */
    public abstract Pointer dataPointForNumpy(Pointer npyArray);

    /**
     * Get a shape buffer for a numpy array.
     * Used in conjunction with {@link org.nd4j.linalg.factory.NDArrayFactory#createFromNpyPointer(Pointer)}
     *
     * @param npyArray the numpy array to get the shape buffer for
     * @return a pointer representing the shape buffer for numpy
     */
    public abstract Pointer shapeBufferForNumpy(Pointer npyArray);

    /**
     * Thie method releases numpy pointer
     * <p>
     * PLEASE NOTE: This method should be ONLY used if pointer/numpy array was originated from file
     *
     * @param npyArray
     */
    public abstract void releaseNumpy(Pointer npyArray);


    /**
     * Create a numpy array pointer
     * from a file
     *
     * @param path the path to the file
     * @return
     */
    public abstract Pointer numpyFromFile(BytePointer path);


    /**
     * Return the length of a shape buffer
     * based on the pointer
     *
     * @param buffer the buffer pointer to check
     * @return
     */
    public abstract int lengthForShapeBufferPointer(Pointer buffer);

    /**
     * Calculate the element size
     * for a numpy array
     *
     * @param npyArray the numpy array to get the
     *                 element size for
     * @return the element size for a given array
     */
    public abstract int elementSizeForNpyArray(Pointer npyArray);


    /**
     * The pointer to get the address for
     *
     * @param address the address to get the pointer
     * @return the pointer for the given address
     */
    public abstract Pointer pointerForAddress(long address);


    public abstract void tear(PointerPointer extras, Pointer tensor, @Cast("Nd4jLong *") LongPointer xShapeInfo,
                                    PointerPointer targets, @Cast("Nd4jLong *") LongPointer zShapeInfo, @Cast("Nd4jLong *") LongPointer tadShapeInfo,
                                    @Cast("Nd4jLong *") LongPointer tadOffsets);


    public abstract long encodeBitmap(PointerPointer extraPointers, Pointer dx, LongPointer xShapeInfo, long N, IntPointer dz, float threshold);

    public abstract void decodeBitmap(PointerPointer extraPointers, Pointer dx, long N, Pointer dz, LongPointer zShapeInfo);


    public abstract void encodeThresholdP1(PointerPointer extraPointers, Pointer dx, LongPointer xShapeInfo, long N, IntPointer dz, float threshold);

    public abstract void encodeThresholdP2Int(PointerPointer extraPointers, IntPointer dx, long N, IntPointer dz);

    public abstract void encodeThresholdP3(PointerPointer extraPointers, Pointer dx, LongPointer xShapeInfo, IntPointer offsets, long N, IntPointer dz);

    public abstract void decodeThreshold(PointerPointer extraPointers, Pointer dx, long N, Pointer dz, LongPointer zShapeInfo);

    public abstract void sort(PointerPointer extraPointers, Pointer dx, @Cast("Nd4jLong *") LongPointer xShapeInfo, boolean descending);


    public abstract void sortTad(PointerPointer extraPointers, Pointer dx, @Cast("Nd4jLong *") LongPointer xShapeInfo,
                                 IntPointer dimension, int dimensionLength, @Cast("Nd4jLong *") LongPointer tadShapeInfo,
                                 @Cast("Nd4jLong *") LongPointer tadOffsets, boolean descending);


    public abstract void sortCooIndices(PointerPointer extraPointers, @Cast("Nd4jLong *") LongPointer indices, Pointer values, long length, int rank);


    public abstract LongPointer mmapFile(PointerPointer extraPointers, String fileName, long length);

    public abstract void munmapFile(PointerPointer extraPointers, LongPointer ptrMap, long length);

    public abstract ResultWrapperAbstraction executeFlatGraph(PointerPointer extraPointers, Pointer flatBufferPointer);

    public abstract String getAllCustomOps();

    public abstract String getAllOperations();

    public abstract int execCustomOp(PointerPointer extraPointers, long opHashCode, PointerPointer inputBuffers, PointerPointer inputShapes, int numInput, PointerPointer outputBuffers, PointerPointer outputShapes, int numOutputs, DoublePointer tArgs, int numTArgs, @Cast("Nd4jLong *") LongPointer iArgs, int numIArgs, boolean isInplace);

    public abstract Pointer calculateOutputShapes(PointerPointer extraPointers, long hash, PointerPointer inputShapes, int numInputShapes, DoublePointer tArgs, int numTArgs, @Cast("Nd4jLong *") LongPointer iArgs, int numIArgs);

    public abstract Pointer calculateOutputShapes(PointerPointer extraPointers, long hash, PointerPointer inputBunffers, PointerPointer inputShapes, int numInputShapes, DoublePointer tArgs, int numTArgs, @Cast("Nd4jLong *") LongPointer iArgs, int numIArgs);

    public abstract int registerGraph(PointerPointer extraPointers, long graphId, Pointer flatBufferPointer);

    public abstract Pointer executeStoredGraph(PointerPointer extraPointers, long graphId, PointerPointer inputBuffers, PointerPointer inputShapes, IntPointer inputIndices, int numInputs);

    public abstract void deleteResultWrapper(Pointer ptr);

    public abstract void deleteShapeList(Pointer ptr);

    public abstract int unregisterGraph(PointerPointer extraPointers, long graphId);

    public abstract void deleteIntArray(Pointer pointer);

    public abstract void deletePointerArray(Pointer pointer);

    public abstract void deleteVariablesSet(Pointer pointer);

    // GraphState creation
    public abstract Pointer getGraphState(long id);

    public abstract void deleteGraphState(Pointer state);

    public abstract int estimateThreshold(PointerPointer extraPointers, Pointer x, LongPointer xShapeInfo, int N, float threshold);

    // this method executes op that requires scope to be present: if/while/cond/whatever
    public abstract int execCustomOpWithScope(PointerPointer extraPointers, Pointer state, long opHash, long[] scopes, int numScopes, PointerPointer inputBuffers, PointerPointer inputShapes, int numInputs, PointerPointer outputBuffers, PointerPointer outputShapes, int numOutputs);
}
