//
// Created by agibsonccc on 2/21/16.
//

#ifndef NATIVEOPERATIONS_NATIVEOPS_H
#define NATIVEOPERATIONS_NATIVEOPS_H

#ifndef thread_local
# if __STDC_VERSION__ >= 201112 && !defined __STDC_NO_THREADS__
#  define thread_local _Thread_local
# elif defined _WIN32 && ( \
       defined _MSC_VER || \
       defined __ICL || \
       defined __DMC__ || \
       defined __BORLANDC__ )
#  define thread_local __declspec(thread)
/* note that ICC (linux) and Clang are covered by __GNUC__ */
# elif defined __GNUC__ || \
       defined __SUNPRO_C || \
       defined __xlC__
#  define thread_local __thread
# else
#  error "Cannot define thread_local"
# endif
#endif

#include <pointercast.h>

//DO NOT REMOVE: THIS IS AN EDITOR SEMANTICS THING FOR CLION
//IT DEFINES THE EXPORT MACRO FOR THE EDITOR AND THEN
//RE ADDS THE DEFINITION VIA dll.h
#ifdef  _WIN32
#define ND4J_EXPORT ND4J_EXPORT
#else
#define ND4J_EXPORT
#endif
#include <dll.h>

class ND4J_EXPORT NativeOps {


public:
    /**
       *
       * @param opNum
       * @param x
       * @param xShapeInfo
       * @param extraParams
       */
    double   execIndexReduceScalarDouble(Nd4jPointer *extraPointers,int opNum,
                                         Nd4jPointer x,
                                         Nd4jPointer xShapeInfo,
                                         Nd4jPointer extraParams);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfoBuffer
     * @param dimension
     * @param dimensionLength
     */
    void   execIndexReduceDouble(Nd4jPointer *extraPointers,int opNum,
                                 Nd4jPointer x,
                                 Nd4jPointer xShapeInfo,
                                 Nd4jPointer extraParams,
                                 Nd4jPointer result,
                                 Nd4jPointer resultShapeInfoBuffer,
                                 Nd4jPointer dimension, int dimensionLength);
    /**
     *
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
    void   execBroadcastDouble(
            Nd4jPointer *extraPointers,
            int opNum,
            Nd4jPointer x,
            Nd4jPointer xShapeInfo,
            Nd4jPointer y,
            Nd4jPointer yShapeInfo,
            Nd4jPointer result,
            Nd4jPointer resultShapeInfo,
            Nd4jPointer dimension, int dimensionLength);



    /**
     *
     * @param opNum
     * @param dx
     * @param xStride
     * @param y
     * @param yStride
     * @param result
     * @param resultStride
     * @param extraParams
     * @param n
     */
    void   execPairwiseTransformDouble(Nd4jPointer *extraPointers,int opNum,
                                       Nd4jPointer dx,
                                       int xStride,
                                       Nd4jPointer y,
                                       int yStride,
                                       Nd4jPointer result,
                                       int resultStride,
                                       Nd4jPointer extraParams, Nd4jIndex n);

    /**
     *
     * @param opNum
     * @param dx
     * @param xShapeInfo
     * @param y
     * @param yShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param extraParams
     * @param n
     * @param xIndexes
     * @param yIndexes
     * @param resultIndexes
     */
    void execPairwiseTransformDouble(Nd4jPointer *extraPointers,
                                     int opNum,
                                     Nd4jPointer dx,
                                     Nd4jPointer xShapeInfo,
                                     Nd4jPointer y,
                                     Nd4jPointer yShapeInfo,
                                     Nd4jPointer result,
                                     Nd4jPointer resultShapeInfo,
                                     Nd4jPointer extraParams,
                                     Nd4jPointer xIndexes,
                                     Nd4jPointer yIndexes,
                                     Nd4jPointer resultIndexes);

    /**
     *
     * @param opNum
     * @param dx
     * @param xShapeInfo
     * @param y
     * @param yShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param extraParams
     * @param n
     */
    void execPairwiseTransformDouble(
            Nd4jPointer *extraPointers,
            int opNum,
            Nd4jPointer dx,
            Nd4jPointer  xShapeInfo,
            Nd4jPointer y,
            Nd4jPointer  yShapeInfo,
            Nd4jPointer result,
            Nd4jPointer  resultShapeInfo,
            Nd4jPointer extraParams);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    void   execReduceDouble(Nd4jPointer *extraPointers,int opNum,
                            Nd4jPointer x,
                            Nd4jPointer xShapeInfo,
                            Nd4jPointer extraParams,
                            Nd4jPointer result,
                            Nd4jPointer resultShapeInfo);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    void   execReduceDouble(Nd4jPointer *extraPointers,int opNum,
                            Nd4jPointer x,
                            Nd4jPointer xShapeInfo,
                            Nd4jPointer extraParams,
                            Nd4jPointer result,
                            Nd4jPointer resultShapeInfo,
                            Nd4jPointer dimension,int dimensionLength);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @return
     */
    double execReduceScalarDouble(Nd4jPointer *extraPointers,int opNum,
                                  Nd4jPointer x,
                                  Nd4jPointer xShapeInfo,
                                  Nd4jPointer extraParams);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParamsVals
     * @param y
     * @param yShapeInfo
     * @param result
     * @param resultShapeInfo
     */
    void   execReduce3Double(Nd4jPointer *extraPointers,int opNum,
                             Nd4jPointer x,
                             Nd4jPointer xShapeInfo,
                             Nd4jPointer extraParamsVals,
                             Nd4jPointer y,
                             Nd4jPointer yShapeInfo,
                             Nd4jPointer result,
                             Nd4jPointer resultShapeInfo);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParamsVals
     * @param y
     * @param yShapeInfo
     */
    double   execReduce3ScalarDouble(Nd4jPointer *extraPointers,int opNum,
                                     Nd4jPointer x,
                                     Nd4jPointer xShapeInfo,
                                     Nd4jPointer extraParamsVals,
                                     Nd4jPointer y,
                                     Nd4jPointer yShapeInfo);
    /**
     *
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
    void   execReduce3Double(Nd4jPointer *extraPointers,int opNum,
                             Nd4jPointer x,
                             Nd4jPointer xShapeInfo,
                             Nd4jPointer extraParamsVals,
                             Nd4jPointer y,
                             Nd4jPointer yShapeInfo,
                             Nd4jPointer result,
                             Nd4jPointer resultShapeInfoBuffer,
                             Nd4jPointer dimension,
                             int dimensionLength);
    /**
     *
     * @param opNum
     * @param x
     * @param xStride
     * @param result
     * @param resultStride
     * @param scalar
     * @param extraParams
     * @param n
     */
    void   execScalarDouble(Nd4jPointer *extraPointers,int opNum,
                            Nd4jPointer x,
                            int xStride,
                            Nd4jPointer result,
                            int resultStride,
                            double scalar,
                            Nd4jPointer extraParams,
                            Nd4jIndex n);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param scalar
     * @param extraParams
     * @param n
     */
    void execScalarDouble(Nd4jPointer *extraPointers,int opNum,
                          Nd4jPointer x,
                          Nd4jPointer xShapeInfo,
                          Nd4jPointer result,
                          Nd4jPointer resultShapeInfo,
                          double scalar,
                          Nd4jPointer extraParams);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param scalar
     * @param extraParams
     * @param n
     * @param xIndexes
     * @param resultIndexes
     */
    void execScalarDouble(Nd4jPointer *extraPointers,int opNum,
                          Nd4jPointer x,
                          Nd4jPointer xShapeInfo,
                          Nd4jPointer result,
                          Nd4jPointer resultShapeInfo,
                          double scalar,
                          Nd4jPointer extraParams,
                          Nd4jIndex n,
                          Nd4jPointer xIndexes,
                          Nd4jPointer resultIndexes);
    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     */
    double   execSummaryStatsScalarDouble(Nd4jPointer *extraPointers,int opNum,Nd4jPointer x,
                                          Nd4jPointer xShapeInfo,
                                          Nd4jPointer extraParams,bool biasCorrected);
    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    void   execSummaryStatsDouble(Nd4jPointer *extraPointers,int opNum,
                                  Nd4jPointer x,
                                  Nd4jPointer xShapeInfo,
                                  Nd4jPointer extraParams,
                                  Nd4jPointer result,
                                  Nd4jPointer resultShapeInfo,bool biasCorrected);
    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfoBuffer
     * @param dimension
     * @param dimensionLength
     */
    void   execSummaryStatsDouble(Nd4jPointer *extraPointers,int opNum,Nd4jPointer x,
                                  Nd4jPointer xShapeInfo,
                                  Nd4jPointer extraParams,
                                  Nd4jPointer result,
                                  Nd4jPointer resultShapeInfoBuffer,
                                  Nd4jPointer dimension, int dimensionLength,bool biasCorrected);
    /**
     *
     * @param opNum
     * @param dx
     * @param xStride
     * @param result
     * @param resultStride
     * @param extraParams
     * @param n
     */
    void   execTransformDouble(Nd4jPointer *extraPointers,int opNum,
                               Nd4jPointer dx,
                               int xStride,
                               Nd4jPointer result,
                               int resultStride,
                               Nd4jPointer extraParams, Nd4jIndex n);

    /**
     *
     * @param opNum
     * @param dx
     * @param xShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param extraParams
     * @param n
     */
    void   execTransformDouble(Nd4jPointer *extraPointers,int opNum,
                               Nd4jPointer dx,
                               Nd4jPointer xShapeInfo,
                               Nd4jPointer result,
                               Nd4jPointer resultShapeInfo,
                               Nd4jPointer extraParams);

    /**
     *
     * @param opNum
     * @param dx
     * @param xShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param extraParams
     * @param n
     */
    void   execTransformDouble(Nd4jPointer *extraPointers,int opNum,
                               Nd4jPointer dx,
                               Nd4jPointer xShapeInfo,
                               Nd4jPointer result,
                               Nd4jPointer resultShapeInfo,
                               Nd4jPointer extraParams,
                               Nd4jPointer xIndexes,
                               Nd4jPointer resultIndexes);

    /**
    *
    * @param opNum
    * @param x
    * @param xShapeInfo
    * @param extraParams
    */
    float   execIndexReduceScalarFloat(Nd4jPointer *extraPointers,
                                       int opNum,
                                       Nd4jPointer x,
                                       Nd4jPointer xShapeInfo,
                                       Nd4jPointer extraParams);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfoBuffer
     * @param dimension
     * @param dimensionLength
     */
    void   execIndexReduceFloat(Nd4jPointer *extraPointers,int opNum,
                                Nd4jPointer x,
                                Nd4jPointer xShapeInfo,
                                Nd4jPointer extraParams,
                                Nd4jPointer result,
                                Nd4jPointer resultShapeInfoBuffer,
                                Nd4jPointer dimension, int dimensionLength);
    /**
     *
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
    void   execBroadcastFloat(
            Nd4jPointer *extraPointers,
            int opNum,
            Nd4jPointer x,
            Nd4jPointer xShapeInfo,
            Nd4jPointer y,
            Nd4jPointer yShapeInfo,
            Nd4jPointer result,
            Nd4jPointer resultShapeInfo,
            Nd4jPointer dimension, int dimensionLength);



    /**
     *
     * @param opNum
     * @param dx
     * @param xStride
     * @param y
     * @param yStride
     * @param result
     * @param resultStride
     * @param extraParams
     * @param n
     */
    void   execPairwiseTransformFloat(Nd4jPointer *extraPointers,int opNum,
                                      Nd4jPointer dx,
                                      int xStride,
                                      Nd4jPointer y,
                                      int yStride,
                                      Nd4jPointer result,
                                      int resultStride,
                                      Nd4jPointer extraParams, Nd4jIndex n);

    /**
     *
     * @param opNum
     * @param dx
     * @param xShapeInfo
     * @param y
     * @param yShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param extraParams
     * @param n
     * @param xIndexes
     * @param yIndexes
     * @param resultIndexes
     */
    void execPairwiseTransformFloat(Nd4jPointer *extraPointers,int opNum,
                                    Nd4jPointer dx,
                                    Nd4jPointer xShapeInfo,
                                    Nd4jPointer y,
                                    Nd4jPointer yShapeInfo,
                                    Nd4jPointer result,
                                    Nd4jPointer resultShapeInfo,
                                    Nd4jPointer extraParams,
                                    Nd4jPointer xIndexes,
                                    Nd4jPointer yIndexes,
                                    Nd4jPointer resultIndexes);

    /**
     *
     * @param opNum
     * @param dx
     * @param xShapeInfo
     * @param y
     * @param yShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param extraParams
     * @param n
     */
    void execPairwiseTransformFloat(Nd4jPointer *extraPointers,int opNum,
                                    Nd4jPointer dx,
                                    Nd4jPointer  xShapeInfo,
                                    Nd4jPointer y,
                                    Nd4jPointer  yShapeInfo,
                                    Nd4jPointer result,
                                    Nd4jPointer  resultShapeInfo,
                                    Nd4jPointer extraParams);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    void   execReduceFloat(Nd4jPointer *extraPointers,int opNum,
                           Nd4jPointer x,
                           Nd4jPointer xShapeInfo,
                           Nd4jPointer extraParams,
                           Nd4jPointer result,
                           Nd4jPointer resultShapeInfo);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    void   execReduceFloat(Nd4jPointer *extraPointers,int opNum,
                           Nd4jPointer x,
                           Nd4jPointer xShapeInfo,
                           Nd4jPointer extraParams,
                           Nd4jPointer result,
                           Nd4jPointer resultShapeInfo,
                           Nd4jPointer dimension,int dimensionLength);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @return
     */
    float execReduceScalarFloat(Nd4jPointer *extraPointers,int opNum,
                                Nd4jPointer x,
                                Nd4jPointer xShapeInfo,
                                Nd4jPointer extraParams);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParamsVals
     * @param y
     * @param yShapeInfo
     * @param result
     * @param resultShapeInfo
     */
    void   execReduce3Float(Nd4jPointer *extraPointers,int opNum,
                            Nd4jPointer x,
                            Nd4jPointer xShapeInfo,
                            Nd4jPointer extraParamsVals,
                            Nd4jPointer y,
                            Nd4jPointer yShapeInfo,
                            Nd4jPointer result,
                            Nd4jPointer resultShapeInfo);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParamsVals
     * @param y
     * @param yShapeInfo
     */
    float   execReduce3ScalarFloat(Nd4jPointer *extraPointers,int opNum,
                                   Nd4jPointer x,
                                   Nd4jPointer xShapeInfo,
                                   Nd4jPointer extraParamsVals,
                                   Nd4jPointer y,
                                   Nd4jPointer yShapeInfo);
    /**
     *
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
    void   execReduce3Float(Nd4jPointer *extraPointers,int opNum,
                            Nd4jPointer x,
                            Nd4jPointer xShapeInfo,
                            Nd4jPointer extraParamsVals,
                            Nd4jPointer y,
                            Nd4jPointer yShapeInfo,
                            Nd4jPointer result,
                            Nd4jPointer resultShapeInfoBuffer,
                            Nd4jPointer dimension,
                            int dimensionLength);
    /**
     *
     * @param opNum
     * @param x
     * @param xStride
     * @param result
     * @param resultStride
     * @param scalar
     * @param extraParams
     * @param n
     */
    void   execScalarFloat(Nd4jPointer *extraPointers,int opNum,
                           Nd4jPointer x,
                           int xStride,
                           Nd4jPointer result,
                           int resultStride,
                           double scalar,
                           Nd4jPointer extraParams,
                           Nd4jIndex n);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param scalar
     * @param extraParams
     * @param n
     */
    void execScalarFloat(Nd4jPointer *extraPointers,int opNum,
                         Nd4jPointer x,
                         Nd4jPointer xShapeInfo,
                         Nd4jPointer result,
                         Nd4jPointer resultShapeInfo,
                         float scalar,
                         Nd4jPointer extraParams);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param scalar
     * @param extraParams
     * @param n
     * @param xIndexes
     * @param resultIndexes
     */
    void execScalarFloat(Nd4jPointer *extraPointers,int opNum,
                         Nd4jPointer x,
                         Nd4jPointer xShapeInfo,
                         Nd4jPointer result,
                         Nd4jPointer resultShapeInfo,
                         double scalar,
                         Nd4jPointer extraParams,
                         Nd4jPointer xIndexes,
                         Nd4jPointer resultIndexes);
    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     */
    float   execSummaryStatsScalarFloat(Nd4jPointer *extraPointers,int opNum,Nd4jPointer x,
                                        Nd4jPointer xShapeInfo,
                                        Nd4jPointer extraParams,bool biasCorrected);
    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    void   execSummaryStatsFloat(Nd4jPointer *extraPointers,int opNum,
                                 Nd4jPointer x,
                                 Nd4jPointer xShapeInfo,
                                 Nd4jPointer extraParams,
                                 Nd4jPointer result,
                                 Nd4jPointer resultShapeInfo,bool biasCorrected);
    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfoBuffer
     * @param dimension
     * @param dimensionLength
     */
    void   execSummaryStatsFloat(Nd4jPointer *extraPointers,int opNum,Nd4jPointer x,
                                 Nd4jPointer xShapeInfo,
                                 Nd4jPointer extraParams,
                                 Nd4jPointer result,
                                 Nd4jPointer resultShapeInfoBuffer,
                                 Nd4jPointer dimension, int dimensionLength,bool biasCorrected);
    /**
     *
     * @param opNum
     * @param dx
     * @param xStride
     * @param result
     * @param resultStride
     * @param extraParams
     * @param n
     */
    void   execTransformFloat(Nd4jPointer *extraPointers,int opNum,
                              Nd4jPointer dx,
                              int xStride,
                              Nd4jPointer result,
                              int resultStride,
                              Nd4jPointer extraParams, Nd4jIndex n);

    /**
     *
     * @param opNum
     * @param dx
     * @param xShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param extraParams
     * @param n
     */
    void   execTransformFloat(Nd4jPointer *extraPointers,int opNum,
                              Nd4jPointer dx,
                              Nd4jPointer xShapeInfo,
                              Nd4jPointer result,
                              Nd4jPointer resultShapeInfo,
                              Nd4jPointer extraParams);

    /**
     *
     * @param opNum
     * @param dx
     * @param xShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param extraParams
     * @param n
     */
    void   execTransformFloat(Nd4jPointer *extraPointers,int opNum,
                              Nd4jPointer dx,
                              Nd4jPointer xShapeInfo,
                              Nd4jPointer result,
                              Nd4jPointer resultShapeInfo,
                              Nd4jPointer extraParams,
                              Nd4jPointer xIndexes,
                              Nd4jPointer resultIndexes);


    /**
* Append an input array
* to the end of a flat array
* in a particular order
* @param offset the offset of the array to start at
* @param order the order
* @param result the result array
* @param resultShapeInfo the shape info for te array
* @param input the input for the array
* @param inputShapeInfo the shape information for that array
*/
    void flattenFloat(
            Nd4jPointer *extraPointers,
            int offset,
            char order,
            Nd4jPointer result,
            Nd4jPointer resultShapeInfo,
            Nd4jPointer input,
            Nd4jPointer inputShapeInfo);

    /**
* Append an input array
* to the end of a flat array
* in a particular order
* @param offset the offset of the array to start at
* @param order the order
* @param result the result array
* @param resultShapeInfo the shape info for te array
* @param input the input for the array
* @param inputShapeInfo the shape information for that array
*/
    void flattenDouble(
            Nd4jPointer *extraPointers,
            int offset,
            char order,
            Nd4jPointer result,
            Nd4jPointer resultShapeInfo,
            Nd4jPointer input,
            Nd4jPointer inputShapeInfo);

   /**
    * Concatneate multi array of the same shape together
    * along a particular dimension
    */
    void concatFloat(
            Nd4jPointer *extraPointers,
            int dimension,
            int numArrays,
            Nd4jPointer *data,
            Nd4jPointer *inputShapeInfo,
            Nd4jPointer result,
            Nd4jPointer resultShapeInfo);
/**
    * Concatneate multi array of the same shape together
    * along a particular dimension
    */
    void concatDouble(
            Nd4jPointer *extraPointers,
            int dimension,
            int numArrays,
            Nd4jPointer *data,
            Nd4jPointer *inputShapeInfo,
            Nd4jPointer result,
            Nd4jPointer resultShapeInfo);

    /**
     * This method implementation exists only for cuda.
     * The other backends should have dummy method for JNI compatibility reasons.
     */
    void initializeDevicesAndFunctions();


    /**
     * This method acquires memory chunk of requested size on host side
     *
     * @param pointer pointer that'll be used for allocation
     * @param memorySize memory size, in bytes
     * @param flags optional parameter
     */
    Nd4jPointer mallocHost(long memorySize, int flags);

    /**
     * This method acquires memory chunk of requested size on specified device
     *
     * @param pointer pointer that'll be used for allocation
     * @param memorySize memory size, in bytes
     * @param ptrToDeviceId pointer to deviceId. For cuda that's just and int, for OpenCL that's pointer to device_id, etc
     * @param flags optional parameter
     */
    Nd4jPointer mallocDevice(long memorySize, Nd4jPointer ptrToDeviceId, int flags);

    /**
     * This method releases previously allocated host memory space
     *
     * @param pointer pointer that'll be freed
     */
    Nd4jPointer freeHost(Nd4jPointer pointer);

    /**
     * This method releases previously allocated memory space on device
     *
     * @param pointer pointer that'll be freed
     * @param ptrToDeviceId pointer to deviceId.
     */
    Nd4jPointer freeDevice(Nd4jPointer pointer, Nd4jPointer ptrToDeviceId);

    int ompGetNumThreads();

    void setOmpNumThreads(int threads);



    Nd4jPointer createContext();

    Nd4jPointer createStream();

    Nd4jPointer createEvent();

    Nd4jPointer createBlasHandle();

    Nd4jPointer registerEvent(Nd4jPointer event, Nd4jPointer stream);

    Nd4jPointer destroyEvent(Nd4jPointer event);

    Nd4jPointer setBlasStream(Nd4jPointer handle, Nd4jPointer stream);

    Nd4jPointer setDevice(Nd4jPointer ptrToDeviceId);

    Nd4jPointer streamSynchronize(Nd4jPointer stream);

    Nd4jPointer eventSynchronize(Nd4jPointer event);

    long getDeviceFreeMemory(Nd4jPointer ptrToDeviceId);

    Nd4jPointer memcpy(Nd4jPointer dst, Nd4jPointer src, long size, int flags, Nd4jPointer reserved);

    Nd4jPointer memcpyAsync(Nd4jPointer dst, Nd4jPointer src, long size, int flags, Nd4jPointer reserved);

    Nd4jPointer memset(Nd4jPointer dst, int value, long size, int flags, Nd4jPointer reserved);

    Nd4jPointer memsetAsync(Nd4jPointer dst, int value, long size, int flags, Nd4jPointer reserved);

    Nd4jPointer getAvailableDevices();

    void enableDebugMode(bool reallyEnable);

    void enableVerboseMode(bool reallyEnable);

    void setGridLimit(int gridSize);
};


#endif //NATIVEOPERATIONS_NATIVEOPS_H
