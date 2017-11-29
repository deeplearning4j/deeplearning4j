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
#include <types/float16.h>
#include <cnpy.h>

//DO NOT REMOVE: THIS IS AN EDITOR SEMANTICS THING FOR CLION
//IT DEFINES THE EXPORT MACRO FOR THE EDITOR AND THEN
//RE ADDS THE DEFINITION VIA dll.h
#ifdef  _WIN32
#define ND4J_EXPORT __declspec(dllexport)
#else
#define ND4J_EXPORT
#endif
#include <dll.h>
#include <helpers/BlasHelper.h>

/*
int tad_threshold = 1;
int element_threshold = 32;

bool debug = false;
bool verbose = false;
*/

#include <array/ShapeList.h>

class ND4J_EXPORT NativeOps {

public:


    /**
     *
     * @param num
     */
    void setElementThreshold(int num);

    /**
     *
     * @param num
     */
    void setTADThreshold(int num);

    /**
       *
       * @param opNum
       * @param x
       * @param xShapeInfo
       * @param extraParams
       */
    double   execIndexReduceScalarDouble(Nd4jPointer *extraPointers,int opNum,
                                         double *x,
                                         int *xInfo,
                                         double *extraParams);

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
                                 double *x,
                                 int *xInfo,
                                 double *extraParams,
                                 double *result,
                                 int *resultShapeInfoBuffer,
                                 int *dimension, int dimensionLength);
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
            double *x,
            int *xInfo,
            double *y,
            int *yInfo,
            double *result,
            int *resultShapeInfo,
            int *dimension,
            int dimensionLength);



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
    void   execPairwiseTransformDouble(Nd4jPointer *extraPointers,
                                       int opNum,
                                       double *dx,
                                       int xStride,
                                       double *y,
                                       int yStride,
                                       double *result,
                                       int resultStride,
                                       double *extraParams,
                                       Nd4jIndex n);

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
                                     double *dx,
                                     int *xInfo,
                                     double *y,
                                     int *yInfo,
                                     double *result,
                                     int *resultShapeInfo,
                                     double *extraParams,
                                     int *xIndexes,
                                     int *yIndexes,
                                     int *resultIndexes);

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
            double *dx,
            int *xShapeInfo,
            double *y,
            int *yShapeInfo,
            double *result,
            int *resultShapeInfo,
            double *extraParams);

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
                            double *x,
                            int *xInfo,
                            double *extraParams,
                            double *result,
                            int *resultShapeInfo);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    void   execReduceDouble(Nd4jPointer *extraPointers,
                            int opNum,
                            double *x,
                            int *xInfo,
                            double *extraParams,
                            double *result,
                            int *resultShapeInfo,
                            int *dimension,
                            int dimensionLength);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @return
     */
    double execReduceScalarDouble(Nd4jPointer *extraPointers,
                                  int opNum,
                                  double *x,
                                  int *xInfo,
                                  double *extraParams);

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
    void   execReduce3Double(Nd4jPointer *extraPointers,
                             int opNum,
                             double *x,
                             int *xInfo,
                             double *extraParamsVals,
                             double *y,
                             int *yInfo,
                             double *result,
                             int *resultShapeInfo);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParamsVals
     * @param y
     * @param yShapeInfo
     */
    double   execReduce3ScalarDouble(Nd4jPointer *extraPointers,
                                     int opNum,
                                     double *x,
                                     int *xInfo,
                                     double *extraParamsVals,
                                     double *y,
                                     int *yInfo);
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
    void   execReduce3Double(Nd4jPointer *extraPointers,
                             int opNum,
                             double *x,
                             int *xInfo,
                             double *extraParamsVals,
                             double *y,
                             int *yInfo,
                             double *result,
                             int *resultShapeInfoBuffer,
                             int *dimension,
                             int dimensionLength);

    void execReduce3AllDouble(Nd4jPointer *extraPointers,
                             int opNum,
                             double *x,
                             int *xInfo,
                             double *extraParamsVals,
                             double *y,
                             int *yInfo,
                             double *result,
                             int *resultShapeInfoBuffer,
                             int *dimension,
                             int dimensionLength,
                             int *xTadShapeInfo,
                             Nd4jIndex *xOffsets,
                             int *yTadShapeInfo,
                             Nd4jIndex *yOffsets);

    void execReduce3AllFloat(Nd4jPointer *extraPointers,
                              int opNum,
                              float *x,
                              int *xInfo,
                              float *extraParamsVals,
                              float *y,
                              int *yInfo,
                              float *result,
                              int *resultShapeInfoBuffer,
                              int *dimension,
                              int dimensionLength,
                              int *xTadShapeInfo,
                             Nd4jIndex *xOffsets,
                              int *yTadShapeInfo,
                             Nd4jIndex *yOffsets);

    void execReduce3AllHalf(Nd4jPointer *extraPointers,
                              int opNum,
                              float16 *x,
                              int *xInfo,
                              float16 *extraParamsVals,
                              float16 *y,
                              int *yInfo,
                              float16 *result,
                              int *resultShapeInfoBuffer,
                              int *dimension,
                              int dimensionLength,
                              int *xTadShapeInfo,
                            Nd4jIndex *xOffsets,
                              int *yTadShapeInfo,
                            Nd4jIndex *yOffsets);





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
    void   execScalarDouble(Nd4jPointer *extraPointers,
                            int opNum,
                            double *x,
                            int xStride,
                            double *result,
                            int resultStride,
                            double scalar,
                            double *extraParams,
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
    void execScalarDouble(Nd4jPointer *extraPointers,
                          int opNum,
                          double *x,
                          int *xInfo,
                          double *result,
                          int *resultShapeInfo,
                          double scalar,
                          double *extraParams);

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
    void execScalarDouble(Nd4jPointer *extraPointers,
                          int opNum,
                          double *x,
                          int *xInfo,
                          double *result,
                          int *resultShapeInfo,
                          double scalar,
                          double *extraParams,
                          Nd4jIndex n,
                          int *xIndexes,
                          int *resultIndexes);
    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     */
    double   execSummaryStatsScalarDouble(Nd4jPointer *extraPointers,
                                          int opNum,
                                          double *x,
                                          int *xInfo,
                                          double *extraParams,
                                          bool biasCorrected);
    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    void   execSummaryStatsDouble(Nd4jPointer *extraPointers,
                                  int opNum,
                                  double *x,
                                  int *xInfo,
                                  double *extraParams,
                                  double *result,
                                  int *resultShapeInfo,
                                  bool biasCorrected);
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
    void   execSummaryStatsDouble(Nd4jPointer *extraPointers,
                                  int opNum,
                                  double *x,
                                  int *xInfo,
                                  double *extraParams,
                                  double *result,
                                  int *resultShapeInfoBuffer,
                                  int *dimension,
                                  int dimensionLength,
                                  bool biasCorrected);
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
                               double *dx,
                               int xStride,
                               double *result,
                               int resultStride,
                               double *extraParams, Nd4jIndex n);

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
                               double *dx,
                               int *xInfo,
                               double *result,
                               int *resultShapeInfo,
                               double *extraParams);

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
    void   execTransformDouble(Nd4jPointer *extraPointers,
                               int opNum,
                               double* dx,
                               int *xShapeInfo,
                               double* result,
                               int *resultShapeInfo,
                               double* extraParams,
                               int *xIndexes,
                               int *resultIndexes);

    /**
    *
    * @param opNum
    * @param x
    * @param xShapeInfo
    * @param extraParams
    */
    float   execIndexReduceScalarFloat(Nd4jPointer *extraPointers,
                                       int opNum,
                                       float *x,
                                       int *xShapeInfo,
                                       float *extraParams);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @return
     */
    float execIndexReduceScalarHalf(Nd4jPointer *extraPointers,
                                    int opNum,
                                    float16 *x,
                                    int *xShapeInfo,
                                    float16 *extraParams);

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
                                float *x,
                                int *xShapeInfo,
                                float *extraParams,
                                float *result,
                                int *resultShapeInfoBuffer,
                                int *dimension, int dimensionLength);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfoBuffer
     * @param dimension
     * @param dimensionLength
     */
    void   execIndexReduceHalf(Nd4jPointer *extraPointers,int opNum,
                               float16 *x,
                               int *xShapeInfo,
                               float16 *extraParams,
                               float16 *result,
                               int *resultShapeInfoBuffer,
                               int *dimension, int dimensionLength);
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
            float *x,
            int *xShapeInfo,
            float *y,
            int *yShapeInfo,
            float *result,
            int *resultShapeInfo,
            int *dimension,
            int dimensionLength);

    /**
     *
     * @param extraPointers
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
    void   execBroadcastHalf(
            Nd4jPointer *extraPointers,
            int opNum,
            float16 *x,
            int *xShapeInfo,
            float16 *y,
            int *yShapeInfo,
            float16 *result,
            int *resultShapeInfo,
            int *dimension, int dimensionLength);



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
    void   execPairwiseTransformFloat(Nd4jPointer *extraPointers,
                                      int opNum,
                                      float *dx,
                                      int xStride,
                                      float *y,
                                      int yStride,
                                      float *result,
                                      int resultStride,
                                      float *extraParams,
                                      Nd4jIndex n);

    /**
     *
     * @param extraPointers
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
    void   execPairwiseTransformHalf(Nd4jPointer *extraPointers,
                                     int opNum,
                                     float16 *dx,
                                     int xStride,
                                     float16 *y,
                                     int yStride,
                                     float16 *result,
                                     int resultStride,
                                     float16 *extraParams,
                                     Nd4jIndex n);

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
    void execPairwiseTransformFloat(Nd4jPointer *extraPointers,
                                    int opNum,
                                    float *dx,
                                    int *xShapeInfo,
                                    float *y,
                                    int *yShapeInfo,
                                    float *result,
                                    int *resultShapeInfo,
                                    float *extraParams,
                                    int *xIndexes,
                                    int *yIndexes,
                                    int *resultIndexes);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param dx
     * @param xShapeInfo
     * @param y
     * @param yShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param extraParams
     * @param xIndexes
     * @param yIndexes
     * @param resultIndexes
     */
    void execPairwiseTransformHalf(Nd4jPointer *extraPointers,
                                   int opNum,
                                   float16 *dx,
                                   int *xShapeInfo,
                                   float16 *y,
                                   int *yShapeInfo,
                                   float16 *result,
                                   int *resultShapeInfo,
                                   float16 *extraParams,
                                   int *xIndexes,
                                   int *yIndexes,
                                   int *resultIndexes);

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
    void execPairwiseTransformFloat(Nd4jPointer *extraPointers,
                                    int opNum,
                                    float *dx,
                                    int *xShapeInfo,
                                    float *y,
                                    int *yShapeInfo,
                                    float *result,
                                    int *resultShapeInfo,
                                    float *extraParams);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param dx
     * @param xShapeInfo
     * @param y
     * @param yShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param extraParams
     */
    void execPairwiseTransformHalf(Nd4jPointer *extraPointers,
                                   int opNum,
                                   float16 *dx,
                                   int *xShapeInfo,
                                   float16 *y,
                                   int *yShapeInfo,
                                   float16 *result,
                                   int *resultShapeInfo,
                                   float16 *extraParams);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    void   execReduceFloat(Nd4jPointer *extraPointers,
                           int opNum,
                           float *x,
                           int *xShapeInfo,
                           float *extraParams,
                           float *result,
                           int *resultShapeInfo);


    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    void   execReduceHalf(Nd4jPointer *extraPointers,
                          int opNum,
                          float16 *x,
                          int *xShapeInfo,
                          float16 *extraParams,
                          float16 *result,
                          int *resultShapeInfo);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    void   execReduceFloat(Nd4jPointer *extraPointers,
                           int opNum,
                           float *x,
                           int *xShapeInfo,
                           float *extraParams,
                           float *result,
                           int *resultShapeInfo,
                           int *dimension,
                           int dimensionLength);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     * @param dimension
     * @param dimensionLength
     */

    void   execReduceHalf(Nd4jPointer *extraPointers,
                          int opNum,
                          float16 *x,
                          int *xShapeInfo,
                          float16 *extraParams,
                          float16 *result,
                          int *resultShapeInfo,
                          int *dimension,
                          int dimensionLength);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @return
     */
    float execReduceScalarFloat(Nd4jPointer *extraPointers,
                                int opNum,
                                float *x,
                                int *xShapeInfo,
                                float *extraParams);


    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @return
     */
    float execReduceScalarHalf(Nd4jPointer *extraPointers,
                               int opNum,
                               float16 *x,
                               int *xShapeInfo,
                               float16 *extraParams);

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
                            float *x,
                            int *xShapeInfo,
                            float *extraParamsVals,
                            float *y,
                            int *yShapeInfo,
                            float *result,
                            int *resultShapeInfo);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParamsVals
     * @param y
     * @param yShapeInfo
     * @param result
     * @param resultShapeInfo
     */
    void   execReduce3Half(Nd4jPointer *extraPointers,int opNum,
                           float16 *x,
                           int *xShapeInfo,
                           float16 *extraParamsVals,
                           float16 *y,
                           int *yShapeInfo,
                           float16 *result,
                           int *resultShapeInfo);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParamsVals
     * @param y
     * @param yShapeInfo
     */
    float   execReduce3ScalarFloat(Nd4jPointer *extraPointers,
                                   int opNum,
                                   float *x,
                                   int *xShapeInfo,
                                   float *extraParamsVals,
                                   float *y,
                                   int *yShapeInfo);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParamsVals
     * @param y
     * @param yShapeInfo
     * @return
     */
    float   execReduce3ScalarHalf(Nd4jPointer *extraPointers,
                                  int opNum,
                                  float16 *x,
                                  int *xShapeInfo,
                                  float16 *extraParamsVals,
                                  float16 *y,
                                  int *yShapeInfo);

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
    void   execReduce3Float(Nd4jPointer *extraPointers,
                            int opNum,
                            float *x,
                            int *xShapeInfo,
                            float *extraParamsVals,
                            float *y,
                            int *yShapeInfo,
                            float *result,
                            int *resultShapeInfoBuffer,
                            int *dimension,
                            int dimensionLength);

    /**
     *
     * @param extraPointers
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
    void   execReduce3Half(Nd4jPointer *extraPointers,
                           int opNum,
                           float16 *x,
                           int *xShapeInfo,
                           float16 *extraParamsVals,
                           float16 *y,
                           int *yShapeInfo,
                           float16 *result,
                           int *resultShapeInfoBuffer,
                           int *dimension,
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
    void   execScalarFloat(Nd4jPointer *extraPointers,
                           int opNum,
                           float *x,
                           int xStride,
                           float *result,
                           int resultStride,
                           float scalar,
                           float *extraParams,
                           Nd4jIndex n);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xStride
     * @param result
     * @param resultStride
     * @param scalar
     * @param extraParams
     * @param n
     */
    void   execScalarHalf(Nd4jPointer *extraPointers,
                          int opNum,
                          float16 *x,
                          int xStride,
                          float16 *result,
                          int resultStride,
                          float scalar,
                          float16 *extraParams,
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
    void execScalarFloat(Nd4jPointer *extraPointers,
                         int opNum,
                         float *x,
                         int *xShapeInfo,
                         float *result,
                         int *resultShapeInfo,
                         float scalar,
                         float *extraParams);


    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param scalar
     * @param extraParams
     */
    void execScalarHalf(Nd4jPointer *extraPointers,
                        int opNum,
                        float16 *x,
                        int *xShapeInfo,
                        float16 *result,
                        int *resultShapeInfo,
                        float scalar,
                        float16 *extraParams);

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
    void execScalarFloat(Nd4jPointer *extraPointers,
                         int opNum,
                         float *x,
                         int *xShapeInfo,
                         float *result,
                         int *resultShapeInfo,
                         float scalar,
                         float *extraParams,
                         int *xIndexes,
                         int *resultIndexes);


    /*
     * Special case: scalarOp alang dimension
     */
    /**
     *
     * @param extraPointers
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
    void execScalarFloat(Nd4jPointer *extraPointers,int opNum,
                         float *x,
                         int *xShapeInfo,
                         float *z,
                         int *zShapeInfo,
                         float *scalars,
                         float *extraParams,
                         int *dimension,
                         int dimensionLength);

    /**
     *
     * @param extraPointers
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
    void execScalarDouble(Nd4jPointer *extraPointers,int opNum,
                          double *x,
                          int *xShapeInfo,
                          double *z,
                          int *zShapeInfo,
                          double *scalars,
                          double *extraParams,
                          int *dimension,
                          int dimensionLength);

    /**
     *
     * @param extraPointers
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
    void execScalarHalf(Nd4jPointer *extraPointers,int opNum,
                        float16 *x,
                        int *xShapeInfo,
                        float16 *z,
                        int *zShapeInfo,
                        float16 *scalars,
                        float16 *extraParams,
                        int *dimension,
                        int dimensionLength);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     */
    float   execSummaryStatsScalarFloat(Nd4jPointer *extraPointers,int opNum,float *x,
                                        int *xShapeInfo,
                                        float *extraParams,bool biasCorrected);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param biasCorrected
     * @return
     */
    float   execSummaryStatsScalarHalf(Nd4jPointer *extraPointers,
                                       int opNum,
                                       float16 *x,
                                       int *xShapeInfo,
                                       float16 *extraParams,
                                       bool biasCorrected);

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
                                 float *x,
                                 int *xShapeInfo,
                                 float *extraParams,
                                 float *result,
                                 int *resultShapeInfo,bool biasCorrected);


    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     * @param biasCorrected
     */
    void   execSummaryStatsHalf(Nd4jPointer *extraPointers,
                                int opNum,
                                float16 *x,
                                int *xShapeInfo,
                                float16 *extraParams,
                                float16 *result,
                                int *resultShapeInfo,bool biasCorrected);

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
    void   execSummaryStatsFloat(Nd4jPointer *extraPointers,
                                 int opNum,
                                 float *x,
                                 int *xShapeInfo,
                                 float *extraParams,
                                 float *result,
                                 int *resultShapeInfoBuffer,
                                 int *dimension,
                                 int dimensionLength,
                                 bool biasCorrected);


    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfoBuffer
     * @param dimension
     * @param dimensionLength
     * @param biasCorrected
     */
    void   execSummaryStatsHalf(Nd4jPointer *extraPointers,
                                int opNum,
                                float16 *x,
                                int *xShapeInfo,
                                float16 *extraParams,
                                float16 *result,
                                int *resultShapeInfoBuffer,
                                int *dimension,
                                int dimensionLength,
                                bool biasCorrected);

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
    void   execTransformFloat(Nd4jPointer *extraPointers,
                              int opNum,
                              float *dx,
                              int xStride,
                              float *result,
                              int resultStride,
                              float *extraParams,
                              Nd4jIndex n);


    /**
     *
     * @param extraPointers
     * @param opNum
     * @param dx
     * @param xStride
     * @param result
     * @param resultStride
     * @param extraParams
     * @param n
     */
    void   execTransformHalf(Nd4jPointer *extraPointers,
                             int opNum,
                             float16 *dx,
                             int xStride,
                             float16 *result,
                             int resultStride,
                             float16 *extraParams,
                             Nd4jIndex n);

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
    void   execTransformFloat(Nd4jPointer *extraPointers,
                              int opNum,
                              float *dx,
                              int *xShapeInfo,
                              float *result,
                              int *resultShapeInfo,
                              float *extraParams);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param dx
     * @param xShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param extraParams
     */
    void   execTransformHalf(Nd4jPointer *extraPointers,
                             int opNum,
                             float16 *dx,
                             int *xShapeInfo,
                             float16 *result,
                             int *resultShapeInfo,
                             float16 *extraParams);

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
    void   execTransformFloat(Nd4jPointer *extraPointers,
                              int opNum,
                              float *dx,
                              int *xShapeInfo,
                              float *result,
                              int *resultShapeInfo,
                              float *extraParams,
                              int *xIndexes,
                              int *resultIndexes);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param dx
     * @param xShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param extraParams
     * @param xIndexes
     * @param resultIndexes
     */
    void   execTransformHalf(Nd4jPointer *extraPointers,
                             int opNum,
                             float16 *dx,
                             int *xShapeInfo,
                             float16 *result,
                             int *resultShapeInfo,
                             float16 *extraParams,
                             int *xIndexes,
                             int *resultIndexes);


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
            float *result,
            int *resultShapeInfo,
            float *input,
            int *inputShapeInfo);


    /**
     *
     * @param extraPointers
     * @param offset
     * @param order
     * @param result
     * @param resultShapeInfo
     * @param input
     * @param inputShapeInfo
     */
    void flattenHalf(
            Nd4jPointer *extraPointers,
            int offset,
            char order,
            float16 *result,
            int *resultShapeInfo,
            float16 *input,
            int *inputShapeInfo);

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
            double *result,
            int *resultShapeInfo,
            double *input,
            int *inputShapeInfo);

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
            float *result,
            int *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers);
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
            double *result,
            int *resultShapeInfo,
            Nd4jPointer *tadPointers,
            Nd4jPointer *offsetPointers);

    /**
     *
     * @param extraPointers
     * @param dimension
     * @param numArrays
     * @param data
     * @param inputShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param tadPointers
     * @param offsetPointers
     */
    void concatHalf(
            Nd4jPointer *extraPointers,
            int dimension,
            int numArrays,
            Nd4jPointer *data,
            Nd4jPointer *inputShapeInfo,
            float16 *result,
            int *resultShapeInfo,
            Nd4jPointer *tadPointers,
            Nd4jPointer *offsetPointers);


    void specialConcatFloat(
            Nd4jPointer *extraPointers,
            int dimension,
            int numArrays,
            Nd4jPointer *data,
            Nd4jPointer *inputShapeInfo,
            float *result,
            int *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers);
/**
    * Concatneate multi array of the same shape together
    * along a particular dimension
    */
    void specialConcatDouble(
            Nd4jPointer *extraPointers,
            int dimension,
            int numArrays,
            Nd4jPointer *data,
            Nd4jPointer *inputShapeInfo,
            double *result,
            int *resultShapeInfo,
            Nd4jPointer *tadPointers,
            Nd4jPointer *offsetPointers);

    /**
     *
     * @param extraPointers
     * @param dimension
     * @param numArrays
     * @param data
     * @param inputShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param tadPointers
     * @param offsetPointers
     */
    void specialConcatHalf(
            Nd4jPointer *extraPointers,
            int dimension,
            int numArrays,
            Nd4jPointer *data,
            Nd4jPointer *inputShapeInfo,
            float16 *result,
            int *resultShapeInfo,
            Nd4jPointer *tadPointers,
            Nd4jPointer *offsetPointers);

    /**
     * This method implementation exists only for cuda.
     * The other backends should have dummy method for JNI compatibility reasons.
     */
    void initializeDevicesAndFunctions();

    void initializeFunctions(Nd4jPointer *functions);

    /**
     * This method acquires memory chunk of requested size on host side
     *
     * @param pointer pointer that'll be used for allocation
     * @param memorySize memory size, in bytes
     * @param flags optional parameter
     */
    Nd4jPointer mallocHost(Nd4jIndex memorySize, int flags);

    /**
     * This method acquires memory chunk of requested size on specified device
     *
     * @param pointer pointer that'll be used for allocation
     * @param memorySize memory size, in bytes
     * @param ptrToDeviceId pointer to deviceId. For cuda that's just and int, for OpenCL that's pointer to device_id, etc
     * @param flags optional parameter
     */
    Nd4jPointer mallocDevice(Nd4jIndex memorySize, Nd4jPointer ptrToDeviceId, int flags);

    /**
     * This method releases previously allocated host memory space
     *
     * @param pointer pointer that'll be freed
     */
    int freeHost(Nd4jPointer pointer);

    /**
     * This method releases previously allocated memory space on device
     *
     * @param pointer pointer that'll be freed
     * @param ptrToDeviceId pointer to deviceId.
     */
    int freeDevice(Nd4jPointer pointer, Nd4jPointer ptrToDeviceId);

    /**
     *
     * @return
     */
    int ompGetMaxThreads();

    /**
     *
     * @return
     */
    int ompGetNumThreads();

    /**
     *
     * @param threads
     */
    void setOmpNumThreads(int threads);

    /**
     *
     * @param threads
     */
    void setOmpMinThreads(int threads);




    /**
     *
     * @return
     */
    Nd4jPointer createContext();

    /**
     *
     * @return
     */
    Nd4jPointer createStream();

    /**
     *
     * @return
     */
    Nd4jPointer createEvent();

    /**
     *
     * @param event
     * @param stream
     * @return
     */
    int registerEvent(Nd4jPointer event, Nd4jPointer stream);

    /**
     *
     * @param event
     * @return
     */
    int destroyEvent(Nd4jPointer event);

    /**
     *
     * @param ptrToDeviceId
     * @return
     */
    int setDevice(Nd4jPointer ptrToDeviceId);

    /**
     *
     * @return
     */
    int getDevice();

    /**
     *
     * @param stream
     * @return
     */
    int streamSynchronize(Nd4jPointer stream);

    /**
     *
     * @param event
     * @return
     */
    int eventSynchronize(Nd4jPointer event);

    /**
     *
     * @param ptrToDeviceId
     * @return
     */
    Nd4jIndex getDeviceFreeMemory(Nd4jPointer ptrToDeviceId);

    /**
     *
     * @param ptrToDeviceId
     * @return
     */
    Nd4jIndex getDeviceTotalMemory(Nd4jPointer ptrToDeviceId);

    /**
     *
     * @param ptrToDeviceId
     * @return
     */
    int getDeviceMajor(Nd4jPointer ptrToDeviceId);

    /**
     *
     * @param ptrToDeviceId
     * @return
     */
    int getDeviceMinor(Nd4jPointer ptrToDeviceId);

    /**
     *
     * @param ptrToDeviceId
     * @return
     */
    const char * getDeviceName(Nd4jPointer ptrToDeviceId);

    /**
     *
     * @param dst
     * @param src
     * @param size
     * @param flags
     * @param reserved
     * @return
     */
    int memcpy(Nd4jPointer dst,
               Nd4jPointer src,
               Nd4jIndex size,
               int flags,
               Nd4jPointer reserved);

    /**
     *
     * @param dst
     * @param src
     * @param size
     * @param flags
     * @param reserved
     * @return
     */
    int memcpyAsync(Nd4jPointer dst,
                    Nd4jPointer src,
                    Nd4jIndex size,
                    int flags,
                    Nd4jPointer reserved);

    /**
     *
     * @param dst
     * @param value
     * @param size
     * @param flags
     * @param reserved
     * @return
     */
    int memset(Nd4jPointer dst,
               int value,
               Nd4jIndex size,
               int flags,
               Nd4jPointer reserved);

    /**
     *
     * @param dst
     * @param value
     * @param size
     * @param flags
     * @param reserved
     * @return
     */
    int memsetAsync(Nd4jPointer dst,
                    int value,
                    Nd4jIndex size,
                    int flags,
                    Nd4jPointer reserved);

    /**
     *
     * @param dst
     * @param src
     * @param size
     * @param flags
     * @param reserved
     * @return
     */
    int memcpyConstantAsync(Nd4jIndex dst,
                            Nd4jPointer src,
                            Nd4jIndex size,
                            int flags,
                            Nd4jPointer reserved);

    /**
     *
     * @return
     */
    Nd4jPointer getConstantSpace();

    /**
     *
     * @return
     */
    int getAvailableDevices();

    /**
     *
     * @param reallyEnable
     */
    void enableDebugMode(bool reallyEnable);

    /**
     *
     * @param reallyEnable
     */
    void enableVerboseMode(bool reallyEnable);

    /**
     *
     * @param gridSize
     */
    void setGridLimit(int gridSize);

    /**
     *
     * @param xShapeInfo
     * @param dimension
     * @param dimensionLength
     * @param targetBuffer
     * @param offsetsBuffer
     */
    void tadOnlyShapeInfo(int *xShapeInfo,
                          int *dimension,
                          int dimensionLength,
                          int *targetBuffer,
                          Nd4jIndex *offsetsBuffer);

    /*
     * PullRow special op
     */

    /**
     *
     * @param extraPointers
     * @param x
     * @param xShapeInfo
     * @param z
     * @param zShapeInfo
     * @param n
     * @param indexes
     * @param tadShapeInfo
     * @param tadOffsets
     * @param zTadShapeInfo
     * @param zTadOffsets
     */
    void pullRowsHalf(Nd4jPointer *extraPointers,
                      float16 *x,
                      int *xShapeInfo,
                      float16 *z,
                      int *zShapeInfo,
                      int n,
                      int *indexes,
                      int *tadShapeInfo,
                      Nd4jIndex *tadOffsets,
                      int *zTadShapeInfo,
                      Nd4jIndex *zTadOffsets);

    /**
     *
     * @param extraPointers
     * @param x
     * @param xShapeInfo
     * @param z
     * @param zShapeInfo
     * @param n
     * @param indexes
     * @param tadShapeInfo
     * @param tadOffsets
     * @param zTadShapeInfo
     * @param zTadOffsets
     */
    void pullRowsFloat(Nd4jPointer *extraPointers,
                       float *x,
                       int *xShapeInfo,
                       float* z,
                       int *zShapeInfo,
                       int n,
                       int *indexes,
                       int *tadShapeInfo,
                       Nd4jIndex *tadOffsets,
                       int *zTadShapeInfo,
                       Nd4jIndex *zTadOffsets);

    /**
     *
     * @param extraPointers
     * @param x
     * @param xShapeInfo
     * @param z
     * @param zShapeInfo
     * @param n
     * @param indexes
     * @param tadShapeInfo
     * @param tadOffsets
     * @param zTadShapeInfo
     * @param zTadOffsets
     */
    void pullRowsDouble(Nd4jPointer *extraPointers,
                        double *x,
                        int *xShapeInfo,
                        double *z,
                        int *zShapeInfo,
                        int n,
                        int *indexes,
                        int *tadShapeInfo,
                        Nd4jIndex *tadOffsets,
                        int *zTadShapeInfo,
                        Nd4jIndex *zTadOffsets);

    /**
     * Array averaging op
     */
    /**
     *
     * @param extras
     * @param dx
     * @param dz
     * @param n
     * @param length
     * @param propagate
     */
    void averageHalf(Nd4jPointer *extras,
                     Nd4jPointer *dx,
                     float16 *dz,
                     int n,
                     Nd4jIndex length,
                     bool propagate);

    /**
     *
     * @param extras
     * @param dx
     * @param dz
     * @param n
     * @param length
     * @param propagate
     */
    void averageFloat(Nd4jPointer *extras,
                      Nd4jPointer *dx,
                      float *dz,
                      int n,
                      Nd4jIndex length,
                      bool propagate);

    /**
     *
     * @param extras
     * @param dx
     * @param dz
     * @param n
     * @param length
     * @param propagate
     */
    void averageDouble(Nd4jPointer *extras,
                       Nd4jPointer *dx,
                       double *dz,
                       int n,
                       Nd4jIndex length,
                       bool propagate);


    void accumulateHalf(Nd4jPointer *extras,
                          Nd4jPointer *dx,
                          float16 *dz,
                          int n,
                          Nd4jIndex length);


    void accumulateFloat(Nd4jPointer *extras,
                          Nd4jPointer *dx,
                          float *dz,
                          int n,
                          Nd4jIndex length);

    void accumulateDouble(Nd4jPointer *extras,
                       Nd4jPointer *dx,
                       double *dz,
                       int n,
                       Nd4jIndex length);


    /**
     * P2P enabler
     */
    /**
     *
     * @param enable
     */
    void enableP2P(bool enable);

    /**
     *
     */
    void checkP2P();

    /**
     *
     * @return
     */
    bool isP2PAvailable();

    /**
     * Shuffle methods
     */

    /**
     *
     * @param extras
     * @param dx
     * @param xShapeInfo
     * @param dz
     * @param zShapeInfo
     * @param N
     * @param shuffleMap
     * @param tadShapeInfo
     * @param tadOffsets
     */
    void shuffleDouble(Nd4jPointer *extras,
                       Nd4jPointer *dx,
                       Nd4jPointer *xShapeInfo,
                       Nd4jPointer *dz,
                       Nd4jPointer *zShapeInfo,
                       int N,
                       int *shuffleMap,
                       Nd4jPointer *tadShapeInfo,
                       Nd4jPointer *tadOffsets);

    /**
     *
     * @param extras
     * @param dx
     * @param xShapeInfo
     * @param dz
     * @param zShapeInfo
     * @param N
     * @param shuffleMap
     * @param tadShapeInfo
     * @param tadOffsets
     */
    void shuffleFloat(Nd4jPointer *extras,
                      Nd4jPointer *dx,
                      Nd4jPointer *xShapeInfo,
                      Nd4jPointer *dz,
                      Nd4jPointer *zShapeInfo,
                      int N,
                      int *shuffleMap,
                      Nd4jPointer *tadShapeInfo,
                      Nd4jPointer *tadOffsets);


    /**
     *
     * @param extras
     * @param dx
     * @param xShapeInfo
     * @param dz
     * @param zShapeInfo
     * @param N
     * @param shuffleMap
     * @param tadShapeInfo
     * @param tadOffsets
     */
    void shuffleHalf(Nd4jPointer *extras,
                     Nd4jPointer *dx,
                     Nd4jPointer *xShapeInfo,
                     Nd4jPointer *dz,
                     Nd4jPointer *zShapeInfo,
                     int N,
                     int *shuffleMap,
                     Nd4jPointer *tadShapeInfo,
                     Nd4jPointer *tadOffsets);

    /**
     * Type Conversions
     */

    /**
     *
     * @param extras
     * @param srcType
     * @param x
     * @param N
     * @param dstType
     * @param z
     */
    void convertTypes(Nd4jPointer *extras, int srcType, Nd4jPointer x, Nd4jIndex N, int dstType, Nd4jPointer z);


    /**
     *
     * @return
     */
    bool isExperimentalEnabled();

    /**
     * Aggregate
     */

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param arguments
     * @param numArguments
     * @param shapeArguments
     * @param numShapeArguments
     * @param indexArguments
     * @param numIndexArguments
     * @param intArrays
     * @param numIntArrays
     * @param realArguments
     * @param numRealArguments
     */
    void execAggregateFloat(Nd4jPointer *extraPointers,
                            int opNum,
                            float **arguments,
                            int numArguments,
                            int **shapeArguments,
                            int numShapeArguments,
                            int *indexArguments,
                            int numIndexArguments,
                            int **intArrays,
                            int numIntArrays,
                            float *realArguments,
                            int numRealArguments);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param arguments
     * @param numArguments
     * @param shapeArguments
     * @param numShapeArguments
     * @param indexArguments
     * @param numIndexArguments
     * @param intArrays
     * @param numIntArrays
     * @param realArguments
     * @param numRealArguments
     */
    void execAggregateDouble(Nd4jPointer *extraPointers,
                             int opNum,
                             double **arguments,
                             int numArguments,
                             int **shapeArguments,
                             int numShapeArguments,
                             int *indexArguments,
                             int numIndexArguments,
                             int **intArrays,
                             int numIntArrays,
                             double *realArguments,
                             int numRealArguments);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param arguments
     * @param numArguments
     * @param shapeArguments
     * @param numShapeArguments
     * @param indexArguments
     * @param numIndexArguments
     * @param intArrays
     * @param numIntArrays
     * @param realArguments
     * @param numRealArguments
     */
    void execAggregateHalf(Nd4jPointer *extraPointers,
                           int opNum,
                           float16 **arguments,
                           int numArguments,
                           int **shapeArguments,
                           int numShapeArguments,
                           int *indexArguments,
                           int numIndexArguments,
                           int **intArrays,
                           int numIntArrays,
                           float16 *realArguments,
                           int numRealArguments);


    /**
     *
     * @param extraPointers
     * @param numAggregates
     * @param opNum
     * @param maxArgs
     * @param maxShapes
     * @param maxIntArrays
     * @param maxIntArraySize
     * @param maxIdx
     * @param maxReals
     * @param ptrToArguments
     */
    void execAggregateBatchFloat(Nd4jPointer *extraPointers,
                                 int numAggregates,
                                 int opNum,
                                 int maxArgs,
                                 int maxShapes,
                                 int maxIntArrays,
                                 int maxIntArraySize,
                                 int maxIdx,
                                 int maxReals,
                                 void *ptrToArguments);

    /**
     *
     * @param extraPointers
     * @param numAggregates
     * @param opNum
     * @param maxArgs
     * @param maxShapes
     * @param maxIntArrays
     * @param maxIntArraySize
     * @param maxIdx
     * @param maxReals
     * @param ptrToArguments
     */
    void execAggregateBatchDouble(Nd4jPointer *extraPointers,
                                  int numAggregates,
                                  int opNum,
                                  int maxArgs,
                                  int maxShapes,
                                  int maxIntArrays,
                                  int maxIntArraySize,
                                  int maxIdx,
                                  int maxReals,
                                  void *ptrToArguments);

    /**
     *
     * @param extraPointers
     * @param numAggregates
     * @param opNum
     * @param maxArgs
     * @param maxShapes
     * @param maxIntArrays
     * @param maxIntArraySize
     * @param maxIdx
     * @param maxReals
     * @param ptrToArguments
     */
    void execAggregateBatchHalf(Nd4jPointer *extraPointers,
                                int numAggregates,
                                int opNum,
                                int maxArgs,
                                int maxShapes,
                                int maxIntArrays,
                                int maxIntArraySize,
                                int maxIdx,
                                int maxReals,
                                void *ptrToArguments);

    /**
     * Random operations
     */

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param state
     * @param z
     * @param zShapeBuffer
     * @param extraArguments
     */
    void execRandomFloat(Nd4jPointer *extraPointers,
                         int opNum,
                         Nd4jPointer state,
                         float *z,
                         int *zShapeBuffer,
                         float *extraArguments);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param state
     * @param x
     * @param xShapeBuffer
     * @param y
     * @param yShapeBuffer
     * @param z
     * @param zShapeBuffer
     * @param extraArguments
     */
    void execRandomFloat(Nd4jPointer *extraPointers,
                         int opNum,
                         Nd4jPointer state,
                         float *x,
                         int *xShapeBuffer,
                         float *y,
                         int *yShapeBuffer,
                         float *z,
                         int *zShapeBuffer,
                         float *extraArguments);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param state
     * @param x
     * @param xShapeBuffer
     * @param z
     * @param zShapeBuffer
     * @param extraArguments
     */
    void execRandomFloat(Nd4jPointer *extraPointers,
                         int opNum,
                         Nd4jPointer state,
                         float *x,
                         int *xShapeBuffer,
                         float *z,
                         int *zShapeBuffer,
                         float *extraArguments);


    /**
     *
     * @param extraPointers
     * @param opNum
     * @param state
     * @param z
     * @param zShapeBuffer
     * @param extraArguments
     */
    void execRandomDouble(Nd4jPointer *extraPointers,
                          int opNum,
                          Nd4jPointer state,
                          double *z,
                          int *zShapeBuffer,
                          double *extraArguments);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param state
     * @param x
     * @param xShapeBuffer
     * @param y
     * @param yShapeBuffer
     * @param z
     * @param zShapeBuffer
     * @param extraArguments
     */
    void execRandomDouble(Nd4jPointer *extraPointers,
                          int opNum,
                          Nd4jPointer state,
                          double *x,
                          int *xShapeBuffer,
                          double *y,
                          int *yShapeBuffer,
                          double *z,
                          int *zShapeBuffer,
                          double *extraArguments);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param state
     * @param x
     * @param xShapeBuffer
     * @param z
     * @param zShapeBuffer
     * @param extraArguments
     */
    void execRandomDouble(Nd4jPointer *extraPointers,
                          int opNum,
                          Nd4jPointer state,
                          double *x,
                          int *xShapeBuffer,
                          double *z,
                          int *zShapeBuffer,
                          double *extraArguments);


    /**
     *
     * @param extraPointers
     * @param opNum
     * @param state
     * @param z
     * @param zShapeBuffer
     * @param extraArguments
     */
    void execRandomHalf(Nd4jPointer *extraPointers,
                        int opNum,
                        Nd4jPointer state,
                        float16 *z,
                        int *zShapeBuffer,
                        float16 *extraArguments);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param state
     * @param x
     * @param xShapeBuffer
     * @param y
     * @param yShapeBuffer
     * @param z
     * @param zShapeBuffer
     * @param extraArguments
     */
    void execRandomHalf(Nd4jPointer *extraPointers,
                        int opNum,
                        Nd4jPointer state,
                        float16 *x,
                        int *xShapeBuffer,
                        float16 *y,
                        int *yShapeBuffer,
                        float16 *z,
                        int *zShapeBuffer,
                        float16 *extraArguments);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param state
     * @param x
     * @param xShapeBuffer
     * @param z
     * @param zShapeBuffer
     * @param extraArguments
     */
    void execRandomHalf(Nd4jPointer *extraPointers,
                        int opNum,
                        Nd4jPointer state,
                        float16 *x,
                        int *xShapeBuffer,
                        float16 *z,
                        int *zShapeBuffer,
                        float16 *extraArguments);



    /**
     *
     * @param extraPointers
     * @param seed
     * @param bufferSize
     * @param ptrToBuffer
     * @return
     */
    Nd4jPointer initRandom(Nd4jPointer *extraPointers,
                           long seed,
                           long bufferSize,
                           Nd4jPointer ptrToBuffer);

    /**
     *
     * @param extraPointers
     * @param seed
     * @param ptrRandom
     */
    void refreshBuffer(Nd4jPointer *extraPointers,
                       long seed,
                       Nd4jPointer ptrRandom);

    /**
     *
     * @param extraPointers
     * @param seed
     * @param ptrRandom
     */
    void reSeedBuffer(Nd4jPointer *extraPointers,
                      long seed,
                      Nd4jPointer ptrRandom);

    /**
     *
     * @param ptrRandom
     */
    void destroyRandom(Nd4jPointer ptrRandom);

    /**
     * Grid operations
     */

    /**
     *
     * @param extras
     * @param opTypeA
     * @param opNumA
     * @param opTypeB
     * @param opNumB
     * @param N
     * @param dx
     * @param xStride
     * @param dy
     * @param yStride
     * @param dz
     * @param zStride
     * @param extraA
     * @param extraB
     * @param scalarA
     * @param scalarB
     */
    void execMetaPredicateStridedFloat(Nd4jPointer *extras,
                                       const int opTypeA,
                                       const int opNumA,
                                       const int opTypeB,
                                       const int opNumB,
                                       long N,
                                       float *dx,
                                       int xStride,
                                       float *dy,
                                       int yStride,
                                       float *dz,
                                       int zStride,
                                       float *extraA,
                                       float *extraB,
                                       float scalarA,
                                       float scalarB);

    /**
     *
     * @param extras
     * @param opTypeA
     * @param opNumA
     * @param opTypeB
     * @param opNumB
     * @param N
     * @param dx
     * @param xShapeInfo
     * @param dy
     * @param yShapeInfo
     * @param dz
     * @param zShapeInfo
     * @param extraA
     * @param extraB
     * @param scalarA
     * @param scalarB
     */
    void execMetaPredicateShapeFloat(Nd4jPointer *extras,
                                     const int opTypeA,
                                     const int opNumA,
                                     const int opTypeB,
                                     const int opNumB,
                                     long N,
                                     float *dx,
                                     int *xShapeInfo,
                                     float *dy,
                                     int *yShapeInfo,
                                     float *dz,
                                     int *zShapeInfo,
                                     float *extraA,
                                     float *extraB,
                                     float scalarA,
                                     float scalarB);

    /**
     *
     * @param extras
     * @param opTypeA
     * @param opNumA
     * @param opTypeB
     * @param opNumB
     * @param N
     * @param dx
     * @param xStride
     * @param dy
     * @param yStride
     * @param dz
     * @param zStride
     * @param extraA
     * @param extraB
     * @param scalarA
     * @param scalarB
     */
    void execMetaPredicateStridedDouble(Nd4jPointer *extras,
                                        const int opTypeA,
                                        const int opNumA,
                                        const int opTypeB,
                                        const int opNumB,
                                        long N,
                                        double *dx,
                                        int xStride,
                                        double *dy,
                                        int yStride,
                                        double *dz,
                                        int zStride,
                                        double *extraA,
                                        double *extraB,
                                        double scalarA,
                                        double scalarB);

    /**
     *
     * @param extras
     * @param opTypeA
     * @param opNumA
     * @param opTypeB
     * @param opNumB
     * @param N
     * @param dx
     * @param xShapeInfo
     * @param dy
     * @param yShapeInfo
     * @param dz
     * @param zShapeInfo
     * @param extraA
     * @param extraB
     * @param scalarA
     * @param scalarB
     */
    void execMetaPredicateShapeDouble(Nd4jPointer *extras,
                                      const int opTypeA,
                                      const int opNumA,
                                      const int opTypeB,
                                      const int opNumB,
                                      long N,
                                      double *dx,
                                      int *xShapeInfo,
                                      double *dy,
                                      int *yShapeInfo,
                                      double *dz,
                                      int *zShapeInfo,
                                      double *extraA,
                                      double *extraB,
                                      double scalarA,
                                      double scalarB);

    /**
     *
     * @param extras
     * @param opTypeA
     * @param opNumA
     * @param opTypeB
     * @param opNumB
     * @param N
     * @param dx
     * @param xStride
     * @param dy
     * @param yStride
     * @param dz
     * @param zStride
     * @param extraA
     * @param extraB
     * @param scalarA
     * @param scalarB
     */
    void execMetaPredicateStridedHalf(Nd4jPointer *extras,
                                      const int opTypeA,
                                      const int opNumA,
                                      const int opTypeB,
                                      const int opNumB,
                                      long N,
                                      float16 *dx,
                                      int xStride,
                                      float16 *dy,
                                      int yStride,
                                      float16 *dz,
                                      int zStride,
                                      float16 *extraA,
                                      float16 *extraB,
                                      float scalarA,
                                      float scalarB);

    /**
     *
     * @param extras
     * @param opTypeA
     * @param opNumA
     * @param opTypeB
     * @param opNumB
     * @param N
     * @param dx
     * @param xShapeInfo
     * @param dy
     * @param yShapeInfo
     * @param dz
     * @param zShapeInfo
     * @param extraA
     * @param extraB
     * @param scalarA
     * @param scalarB
     */
    void execMetaPredicateShapeHalf(Nd4jPointer *extras,
                                    const int opTypeA,
                                    const int opNumA,
                                    const int opTypeB,
                                    const int opNumB,
                                    long N,
                                    float16 *dx,
                                    int *xShapeInfo,
                                    float16 *dy,
                                    int *yShapeInfo,
                                    float16 *dz,
                                    int *zShapeInfo,
                                    float16 *extraA,
                                    float16 *extraB,
                                    float scalarA,
                                    float scalarB);


    /**
     *
     * @param extras
     * @param opTypeA
     * @param opNumA
     * @param opTypeB
     * @param opNumB
     * @param dx
     * @param xShapeInfo
     * @param dy
     * @param yShapeInfo
     * @param dz
     * @param zShapeInfo
     * @param dimension
     * @param dimensionLength
     * @param tadShapeInfo
     * @param tadOffsets
     * @param extraA
     * @param extraB
     * @param scalarA
     * @param scalarB
     * @param scalarReturned
     */
    void execMetaPredicateReduceFloat(Nd4jPointer *extras,
                                      const int opTypeA,
                                      const int opNumA,
                                      const int opTypeB,
                                      const int opNumB,
                                      float *dx,
                                      int *xShapeInfo,
                                      float *dy,
                                      int *yShapeInfo,
                                      float *dz,
                                      int *zShapeInfo,
                                      int *dimension,
                                      int dimensionLength,
                                      int *tadShapeInfo,
                                      Nd4jIndex *tadOffsets,
                                      float *extraA,
                                      float *extraB,
                                      float scalarA,
                                      float scalarB,
                                      bool scalarReturned);



    /**
     * Get the shape buffer from a
     * numpy array.
     * **Warning** this allocates memory
     * @param npyArray
     * @return
     */
    Nd4jPointer shapeBufferForNumpy(Nd4jPointer npyArray);

    /**
     * Data buffer for numpy
     * @param npArray
     * @return
     */
    Nd4jPointer dataPointForNumpy(Nd4jPointer npArray);

    /**
     * Create a pointer to an NDarray struct
     * @param path  the path to create the ndarray
     * struct from
     * @return  a pointer to the ndarray struct
     */
    Nd4jPointer numpyFromFile(std::string path);

    /**
     * This method releases pointer.
     *
     * PLEASE NOTE: This method shouldn't be ever called for anything but numpy arrays created from FILE
     *
     * @param npyArray
     */
    void releaseNumpy(Nd4jPointer npyArray);

    /**
     * Return the length of a shape buffer
     * based on the pointer
     * @param buffer  the buffer pointer to check
     * @return
     */
    int lengthForShapeBufferPointer(Nd4jPointer buffer);

    /**
     * Get the element size for a numpy array
     * @param npyArray  the numpy array's address
     * to get the length for
     * @return
     */
    int elementSizeForNpyArray(Nd4jPointer npyArray);


    /**
   * The pointer to get the address for
   *
   * @param address the address to get the pointer
   * @return the pointer for the given address
   */

    Nd4jPointer pointerForAddress(Nd4jIndex address);

    /**
     * This method takes single N-dimensional tensor, and copies its TADs to target arrays
     *
     * @param x
     * @param xShapeInfo
     * @param targets
     * @param zShapeInfo
     * @return
     */
    void tearDouble(Nd4jPointer *extraPointers, double *x, int *xShapeInfo, Nd4jPointer *targets, int *zShapeInfo, int *tadShapeInfo, Nd4jIndex *tadOffsets);

    /**
     * This method takes single N-dimensional tensor, and copies its TADs to target arrays
     *
     * @param x
     * @param xShapeInfo
     * @param targets
     * @param zShapeInfo
     * @return
     */
    void tearFloat(Nd4jPointer *extraPointers, float *x, int *xShapeInfo, Nd4jPointer *targets, int *zShapeInfo, int *tadShapeInfo, Nd4jIndex *tadOffsets);

    /**
     * This method takes single N-dimensional tensor, and copies its TADs to target arrays
     *
     * @param x
     * @param xShapeInfo
     * @param targets
     * @param zShapeInfo
     * @return
     */
    void tearHalf(Nd4jPointer *extraPointers, float16 *x, int *xShapeInfo, Nd4jPointer *targets, int *zShapeInfo, int *tadShapeInfo, Nd4jIndex *tadOffsets);


    Nd4jIndex encodeBitmapFloat(Nd4jPointer *extraPointers, float *dx, Nd4jIndex N, int *dz, float threshold);

    Nd4jIndex encodeBitmapDouble(Nd4jPointer *extraPointers, double *dx, Nd4jIndex N, int *dz, float threshold);

    Nd4jIndex encodeBitmapHalf(Nd4jPointer *extraPointers, float16 *dx, Nd4jIndex N, int *dz, float threshold);

    void decodeBitmapFloat(Nd4jPointer *extraPointers, void *dx, Nd4jIndex N, float *dz);

    void decodeBitmapDouble(Nd4jPointer *extraPointers, void *dx, Nd4jIndex N, double *dz);

    void decodeBitmapHalf(Nd4jPointer *extraPointers, void *dx, Nd4jIndex N, float16 *dz);


    void encodeThresholdP1Double(Nd4jPointer *extraPointers, double *dx, Nd4jIndex N, int *dz, float threshold);

    void encodeThresholdP1Half(Nd4jPointer *extraPointers, float16 *dx, Nd4jIndex N, int *dz, float threshold);

    void encodeThresholdP1Float(Nd4jPointer *extraPointers, float *dx, Nd4jIndex N, int *dz, float threshold);


    void encodeThresholdP2Int(Nd4jPointer *extraPointers, int *dx, Nd4jIndex N, int *dz);


    void encodeThresholdP3Float(Nd4jPointer *extraPointers, float *dx, int *offsets, Nd4jIndex N, int *dz);

    void encodeThresholdP3Double(Nd4jPointer *extraPointers, double *dx, int *offsets, Nd4jIndex N, int *dz);

    void encodeThresholdP3Half(Nd4jPointer *extraPointers, float16 *dx, int *offsets, Nd4jIndex N, int *dz);


    void decodeThresholdFloat(Nd4jPointer *extraPointers, void *dx, Nd4jIndex N, float *dz);

    void decodeThresholdDouble(Nd4jPointer *extraPointers, void *dx, Nd4jIndex N, double *dz);

    void decodeThresholdHalf(Nd4jPointer *extraPointers, void *dx, Nd4jIndex N, float16 *dz);



    void sortFloat(Nd4jPointer *extraPointers, float *x, int *xShapeInfo, bool descending);

    void sortDouble(Nd4jPointer *extraPointers, double *x, int *xShapeInfo, bool descending);

    void sortHalf(Nd4jPointer *extraPointers, float16 *x, int *xShapeInfo, bool descending);



    void sortTadFloat(Nd4jPointer *extraPointers, float *x, int *xShapeInfo, int *dimension, int dimensionLength, int *tadShapeInfo, Nd4jIndex *tadOffsets, bool descending);

    void sortTadDouble(Nd4jPointer *extraPointers, double *x, int *xShapeInfo, int *dimension, int dimensionLength, int *tadShapeInfo, Nd4jIndex *tadOffsets, bool descending);

    void sortTadHalf(Nd4jPointer *extraPointers, float16 *x, int *xShapeInfo, int *dimension, int dimensionLength, int *tadShapeInfo, Nd4jIndex *tadOffsets, bool descending);


    // special sort impl for sorting out COO indices and values
    void sortCooIndicesFloat(Nd4jPointer *extraPointers, int *indices, float *values, Nd4jIndex length, int rank);

    void sortCooIndicesDouble(Nd4jPointer *extraPointers, int *indices, double *values, Nd4jIndex length, int rank);

    void sortCooIndicesHalf(Nd4jPointer *extraPointers, int *indices, float16 *values, Nd4jIndex length, int rank);


    Nd4jIndex* mmapFile(Nd4jPointer *extraPointers, const char *fileName, Nd4jIndex length);

    void munmapFile(Nd4jPointer *extraPointers, Nd4jIndex* ptrMap, Nd4jIndex length);


    // flatbuffers execution
    Nd4jPointer executeFlatGraphFloat(Nd4jPointer *extraPointers, Nd4jPointer flatBufferPointer);

    // protobuf execution
    Nd4jPointer executeProtoGraphFloat(Nd4jPointer *extraPointers, Nd4jPointer protoBufferPointer);
    Nd4jPointer executeProtoGraphFloat(Nd4jPointer *extraPointers, const char *fileName);

    const char* getAllCustomOps();

    // customOp executioner
    int execCustomOpFloat(Nd4jPointer* extraPointers, Nd4jIndex hash, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputs, Nd4jPointer* outputBuffers, Nd4jPointer* outputShapes, int numOutputs, float* tArgs, int numTArgs, int *iArgs, int numIArgs, bool isInplace);
    int execCustomOpDouble(Nd4jPointer* extraPointers, Nd4jIndex hash, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputs, Nd4jPointer* outputBuffers, Nd4jPointer* outputShapes, int numOutputs, double* tArgs, int numTArgs, int *iArgs, int numIArgs, bool isInplace);
    int execCustomOpHalf(Nd4jPointer* extraPointers, Nd4jIndex hash, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputs, Nd4jPointer* outputBuffers, Nd4jPointer* outputShapes, int numOutputs, float16* tArgs, int numTArgs, int *iArgs, int numIArgs, bool isInplace);

    Nd4jPointer* calculateOutputShapesFloat(Nd4jPointer* extraPointers, Nd4jIndex hash, Nd4jPointer* inputShapes, int numInputShapes, float* tArgs, int numTArgs, int *iArgs, int numIArgs);
    Nd4jPointer* calculateOutputShapesHalf(Nd4jPointer* extraPointers, Nd4jIndex hash, Nd4jPointer* inputShapes, int numInputShapes, float16* tArgs, int numTArgs, int *iArgs, int numIArgs);
    Nd4jPointer* calculateOutputShapesDouble(Nd4jPointer* extraPointers, Nd4jIndex hash, Nd4jPointer* inputShapes, int numInputShapes, double* tArgs, int numTArgs, int *iArgs, int numIArgs);
};


#endif //NATIVEOPERATIONS_NATIVEOPS_H
