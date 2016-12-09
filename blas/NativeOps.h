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

//DO NOT REMOVE: THIS IS AN EDITOR SEMANTICS THING FOR CLION
//IT DEFINES THE EXPORT MACRO FOR THE EDITOR AND THEN
//RE ADDS THE DEFINITION VIA dll.h
#ifdef  _WIN32
#define ND4J_EXPORT __declspec(dllexport)
#else
#define ND4J_EXPORT
#endif
#include <dll.h>

class ND4J_EXPORT NativeOps {


public:


    void setElementThreshold(int num);

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
    void   execPairwiseTransformDouble(Nd4jPointer *extraPointers,int opNum,
                                       double *dx,
                                       int xStride,
                                       double *y,
                                       int yStride,
                                       double *result,
                                       int resultStride,
                                       double *extraParams, Nd4jIndex n);

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
    void   execReduceDouble(Nd4jPointer *extraPointers,int opNum,
                            double *x,
                            int *xInfo,
                            double *extraParams,
                            double *result,
                            int *resultShapeInfo,
                            int *dimension,int dimensionLength);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @return
     */
    double execReduceScalarDouble(Nd4jPointer *extraPointers,int opNum,
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
    void   execReduce3Double(Nd4jPointer *extraPointers,int opNum,
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
    double   execReduce3ScalarDouble(Nd4jPointer *extraPointers,int opNum,
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
    void   execReduce3Double(Nd4jPointer *extraPointers,int opNum,
                             double *x,
                             int *xInfo,
                             double *extraParamsVals,
                             double *y,
                             int *yInfo,
                             double *result,
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
    void   execScalarDouble(Nd4jPointer *extraPointers,int opNum,
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
    void execScalarDouble(Nd4jPointer *extraPointers,int opNum,
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
    void execScalarDouble(Nd4jPointer *extraPointers,int opNum,
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
    double   execSummaryStatsScalarDouble(Nd4jPointer *extraPointers,int opNum,double *x,
                                          int *xInfo,
                                          double *extraParams,bool biasCorrected);
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
                                  double *x,
                                  int *xInfo,
                                  double *extraParams,
                                  double *result,
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
    void   execSummaryStatsDouble(Nd4jPointer *extraPointers,int opNum,double *x,
                                  int *xInfo,
                                  double *extraParams,
                                  double *result,
                                  int *resultShapeInfoBuffer,
                                  int *dimension, int dimensionLength,bool biasCorrected);
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
    void   execTransformDouble(Nd4jPointer *extraPointers,int opNum,
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
            int *dimension, int dimensionLength);

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
    void   execPairwiseTransformFloat(Nd4jPointer *extraPointers,int opNum,
                                      float *dx,
                                      int xStride,
                                      float *y,
                                      int yStride,
                                      float *result,
                                      int resultStride,
                                      float *extraParams, Nd4jIndex n);

    void   execPairwiseTransformHalf(Nd4jPointer *extraPointers,int opNum,
                                      float16 *dx,
                                      int xStride,
                                      float16 *y,
                                      int yStride,
                                      float16 *result,
                                      int resultStride,
                                      float16 *extraParams, Nd4jIndex n);

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

    void execPairwiseTransformHalf(Nd4jPointer *extraPointers,int opNum,
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
    void execPairwiseTransformFloat(Nd4jPointer *extraPointers,int opNum,
                                    float *dx,
                                    int *xShapeInfo,
                                    float *y,
                                    int *yShapeInfo,
                                    float *result,
                                    int *resultShapeInfo,
                                    float *extraParams);

    void execPairwiseTransformHalf(Nd4jPointer *extraPointers,int opNum,
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
    void   execReduceFloat(Nd4jPointer *extraPointers,int opNum,
                           float *x,
                           int *xShapeInfo,
                           float *extraParams,
                           float *result,
                           int *resultShapeInfo);

    void   execReduceHalf(Nd4jPointer *extraPointers,int opNum,
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
    void   execReduceFloat(Nd4jPointer *extraPointers,int opNum,
                           float *x,
                           int *xShapeInfo,
                           float *extraParams,
                           float *result,
                           int *resultShapeInfo,
                           int *dimension,int dimensionLength);

    void   execReduceHalf(Nd4jPointer *extraPointers,int opNum,
                           float16 *x,
                           int *xShapeInfo,
                           float16 *extraParams,
                           float16 *result,
                           int *resultShapeInfo,
                           int *dimension,int dimensionLength);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @return
     */
    float execReduceScalarFloat(Nd4jPointer *extraPointers,int opNum,
                                float *x,
                                int *xShapeInfo,
                                float *extraParams);

    float execReduceScalarHalf(Nd4jPointer *extraPointers,int opNum,
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
    float   execReduce3ScalarFloat(Nd4jPointer *extraPointers,int opNum,
                                   float *x,
                                   int *xShapeInfo,
                                   float *extraParamsVals,
                                   float *y,
                                   int *yShapeInfo);

    float   execReduce3ScalarHalf(Nd4jPointer *extraPointers,int opNum,
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
    void   execReduce3Float(Nd4jPointer *extraPointers,int opNum,
                            float *x,
                            int *xShapeInfo,
                            float *extraParamsVals,
                            float *y,
                            int *yShapeInfo,
                            float *result,
                            int *resultShapeInfoBuffer,
                            int *dimension,
                            int dimensionLength);

    void   execReduce3Half(Nd4jPointer *extraPointers,int opNum,
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
    void   execScalarFloat(Nd4jPointer *extraPointers,int opNum,
                           float *x,
                           int xStride,
                           float *result,
                           int resultStride,
                           float scalar,
                           float *extraParams,
                           Nd4jIndex n);

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
    void execScalarFloat(Nd4jPointer *extraPointers,int opNum,
                         float *x,
                         int *xShapeInfo,
                         float *result,
                         int *resultShapeInfo,
                         float scalar,
                         float *extraParams);


    void execScalarHalf(Nd4jPointer *extraPointers,int opNum,
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
    void execScalarFloat(Nd4jPointer *extraPointers,int opNum,
                         float *x,
                         int *xShapeInfo,
                         float *result,
                         int *resultShapeInfo,
                         double scalar,
                         float *extraParams,
                         int *xIndexes,
                         int *resultIndexes);


    /*
     * Special case: scalarOp alang dimension
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

    void execScalarDouble(Nd4jPointer *extraPointers,int opNum,
                         double *x,
                         int *xShapeInfo,
                         double *z,
                         int *zShapeInfo,
                         double *scalars,
                         double *extraParams,
                         int *dimension,
                         int dimensionLength);

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

    float   execSummaryStatsScalarHalf(Nd4jPointer *extraPointers,int opNum,float16 *x,
                                        int *xShapeInfo,
                                        float16 *extraParams,bool biasCorrected);

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


    void   execSummaryStatsHalf(Nd4jPointer *extraPointers,int opNum,
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
    void   execSummaryStatsFloat(Nd4jPointer *extraPointers,int opNum,float *x,
                                 int *xShapeInfo,
                                 float *extraParams,
                                 float *result,
                                 int *resultShapeInfoBuffer,
                                 int *dimension, int dimensionLength,bool biasCorrected);


    void   execSummaryStatsHalf(Nd4jPointer *extraPointers,int opNum,float16 *x,
                                 int *xShapeInfo,
                                 float16 *extraParams,
                                 float16 *result,
                                 int *resultShapeInfoBuffer,
                                 int *dimension, int dimensionLength,bool biasCorrected);

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
                              float *dx,
                              int xStride,
                              float *result,
                              int resultStride,
                              float *extraParams, Nd4jIndex n);


    void   execTransformHalf(Nd4jPointer *extraPointers,int opNum,
                              float16 *dx,
                              int xStride,
                              float16 *result,
                              int resultStride,
                              float16 *extraParams, Nd4jIndex n);

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
                              float *dx,
                              int *xShapeInfo,
                              float *result,
                              int *resultShapeInfo,
                              float *extraParams);

    void   execTransformHalf(Nd4jPointer *extraPointers,int opNum,
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
    void   execTransformFloat(Nd4jPointer *extraPointers,int opNum,
                              float *dx,
                              int *xShapeInfo,
                              float *result,
                              int *resultShapeInfo,
                              float *extraParams,
                              int *xIndexes,
                              int *resultIndexes);

    void   execTransformHalf(Nd4jPointer *extraPointers,int opNum,
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
            int *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers);

    void concatHalf(
            Nd4jPointer *extraPointers,
            int dimension,
            int numArrays,
            Nd4jPointer *data,
            Nd4jPointer *inputShapeInfo,
            float16 *result,
            int *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers);

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

    int ompGetMaxThreads();

    int ompGetNumThreads();

    void setOmpNumThreads(int threads);

    void setOmpMinThreads(int threads);



    Nd4jPointer createContext();

    Nd4jPointer createStream();

    Nd4jPointer createEvent();

    int registerEvent(Nd4jPointer event, Nd4jPointer stream);

    int destroyEvent(Nd4jPointer event);

    int setDevice(Nd4jPointer ptrToDeviceId);

    int getDevice();

    int streamSynchronize(Nd4jPointer stream);

    int eventSynchronize(Nd4jPointer event);

    Nd4jIndex getDeviceFreeMemory(Nd4jPointer ptrToDeviceId);

    Nd4jIndex getDeviceTotalMemory(Nd4jPointer ptrToDeviceId);

    int getDeviceMajor(Nd4jPointer ptrToDeviceId);

    int getDeviceMinor(Nd4jPointer ptrToDeviceId);

    const char * getDeviceName(Nd4jPointer ptrToDeviceId);

    int memcpy(Nd4jPointer dst, Nd4jPointer src, Nd4jIndex size, int flags, Nd4jPointer reserved);

    int memcpyAsync(Nd4jPointer dst, Nd4jPointer src, Nd4jIndex size, int flags, Nd4jPointer reserved);

    int memset(Nd4jPointer dst, int value, Nd4jIndex size, int flags, Nd4jPointer reserved);

    int memsetAsync(Nd4jPointer dst, int value, Nd4jIndex size, int flags, Nd4jPointer reserved);

    int memcpyConstantAsync(Nd4jIndex dst, Nd4jPointer src, Nd4jIndex size, int flags, Nd4jPointer reserved);

    Nd4jPointer getConstantSpace();

    int getAvailableDevices();

    void enableDebugMode(bool reallyEnable);

    void enableVerboseMode(bool reallyEnable);

    void setGridLimit(int gridSize);

    void tadOnlyShapeInfo(int *xShapeInfo, int *dimension, int dimensionLength, int *targetBuffer, int *offsetsBuffer);

    /*
     * PullRow special op
     */

    void pullRowsHalf(Nd4jPointer *extraPointers, float16 *x, int *xShapeInfo, float16 *z, int *zShapeInfo, int n, int *indexes, int *tadShapeInfo, int *tadOffsets, int *zTadShapeInfo, int *zTadOffsets);

    void pullRowsFloat(Nd4jPointer *extraPointers, float *x, int *xShapeInfo, float* z, int *zShapeInfo, int n, int *indexes, int *tadShapeInfo, int *tadOffsets, int *zTadShapeInfo, int *zTadOffsets);

    void pullRowsDouble(Nd4jPointer *extraPointers, double *x, int *xShapeInfo, double *z, int *zShapeInfo, int n, int *indexes, int *tadShapeInfo, int *tadOffsets, int *zTadShapeInfo, int *zTadOffsets);

    /**
     * Array averaging op
     */
    void averageHalf(Nd4jPointer *extras, Nd4jPointer *dx, float16 *dz, int n, Nd4jIndex length, bool propagate);

    void averageFloat(Nd4jPointer *extras, Nd4jPointer *dx, float *dz, int n, Nd4jIndex length, bool propagate);

    void averageDouble(Nd4jPointer *extras, Nd4jPointer *dx, double *dz, int n, Nd4jIndex length, bool propagate);


    /**
     * P2P enabler
     */
    void enableP2P(bool enable);

    void checkP2P();

    bool isP2PAvailable();

    /**
     * Shuffle methods
     */

    void shuffleDouble(Nd4jPointer *extras, Nd4jPointer *dx, Nd4jPointer *xShapeInfo, Nd4jPointer *dz, Nd4jPointer *zShapeInfo, int N, int *shuffleMap, Nd4jPointer *tadShapeInfo, Nd4jPointer *tadOffsets);

    void shuffleFloat(Nd4jPointer *extras, Nd4jPointer *dx, Nd4jPointer *xShapeInfo, Nd4jPointer *dz, Nd4jPointer *zShapeInfo, int N, int *shuffleMap, Nd4jPointer *tadShapeInfo, Nd4jPointer *tadOffsets);

    void shuffleHalf(Nd4jPointer *extras, Nd4jPointer *dx, Nd4jPointer *xShapeInfo, Nd4jPointer *dz, Nd4jPointer *zShapeInfo, int N, int *shuffleMap, Nd4jPointer *tadShapeInfo, Nd4jPointer *tadOffsets);

    /**
     * Type Conversions
     */

    void convertTypes(Nd4jPointer *extras, int srcType, Nd4jPointer x, long N, int dstType, Nd4jPointer z);


    bool isExperimentalEnabled();

    /**
     * Aggregate
     */
    void execAggregateFloat(Nd4jPointer *extraPointers,int opNum, float **arguments, int numArguments, int **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays, float *realArguments, int numRealArguments);

    void execAggregateDouble(Nd4jPointer *extraPointers,int opNum, double **arguments, int numArguments, int **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays, double *realArguments, int numRealArguments);

    void execAggregateHalf(Nd4jPointer *extraPointers,int opNum, float16 **arguments, int numArguments, int **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays, float16 *realArguments, int numRealArguments);



    void execAggregateBatchFloat(Nd4jPointer *extraPointers, int numAggregates, int opNum, int maxArgs, int maxShapes, int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals, void *ptrToArguments);

    void execAggregateBatchDouble(Nd4jPointer *extraPointers, int numAggregates, int opNum, int maxArgs, int maxShapes, int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals, void *ptrToArguments);

    void execAggregateBatchHalf(Nd4jPointer *extraPointers, int numAggregates, int opNum, int maxArgs, int maxShapes, int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals, void *ptrToArguments);

    /**
     * Random operations
     */

    void execRandomFloat(Nd4jPointer *extraPointers, int opNum, Nd4jPointer state, float *z, int *zShapeBuffer, float *extraArguments);

    void execRandomFloat(Nd4jPointer *extraPointers, int opNum, Nd4jPointer state, float *x, int *xShapeBuffer, float *y, int *yShapeBuffer, float *z, int *zShapeBuffer, float *extraArguments);

    void execRandomFloat(Nd4jPointer *extraPointers, int opNum, Nd4jPointer state, float *x, int *xShapeBuffer, float *z, int *zShapeBuffer, float *extraArguments);


    void execRandomDouble(Nd4jPointer *extraPointers, int opNum, Nd4jPointer state, double *z, int *zShapeBuffer, double *extraArguments);

    void execRandomDouble(Nd4jPointer *extraPointers, int opNum, Nd4jPointer state, double *x, int *xShapeBuffer, double *y, int *yShapeBuffer, double *z, int *zShapeBuffer, double *extraArguments);

    void execRandomDouble(Nd4jPointer *extraPointers, int opNum, Nd4jPointer state, double *x, int *xShapeBuffer, double *z, int *zShapeBuffer, double *extraArguments);


    void execRandomHalf(Nd4jPointer *extraPointers, int opNum, Nd4jPointer state, float16 *z, int *zShapeBuffer, float16 *extraArguments);

    void execRandomHalf(Nd4jPointer *extraPointers, int opNum, Nd4jPointer state, float16 *x, int *xShapeBuffer, float16 *y, int *yShapeBuffer, float16 *z, int *zShapeBuffer, float16 *extraArguments);

    void execRandomHalf(Nd4jPointer *extraPointers, int opNum, Nd4jPointer state, float16 *x, int *xShapeBuffer, float16 *z, int *zShapeBuffer, float16 *extraArguments);



    Nd4jPointer initRandom(Nd4jPointer *extraPointers, long seed, long bufferSize, Nd4jPointer ptrToBuffer);

    void refreshBuffer(Nd4jPointer *extraPointers, long seed, Nd4jPointer ptrRandom);

    void reSeedBuffer(Nd4jPointer *extraPointers, long seed, Nd4jPointer ptrRandom);

    void destroyRandom(Nd4jPointer ptrRandom);

    /**
     * Grid operations
     */

    void execMetaPredicateStridedFloat(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, long N, float *dx, int xStride, float *dy, int yStride, float *dz, int zStride, float *extraA, float *extraB, float scalarA, float scalarB);

    void execMetaPredicateShapeFloat(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, long N, float *dx, int *xShapeInfo, float *dy, int *yShapeInfo, float *dz, int *zShapeInfo, float *extraA, float *extraB, float scalarA, float scalarB);

    void execMetaPredicateStridedDouble(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, long N, double *dx, int xStride, double *dy, int yStride, double *dz, int zStride, double *extraA, double *extraB, double scalarA, double scalarB);

    void execMetaPredicateShapeDouble(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, long N, double *dx, int *xShapeInfo, double *dy, int *yShapeInfo, double *dz, int *zShapeInfo, double *extraA, double *extraB, double scalarA, double scalarB);

    void execMetaPredicateStridedHalf(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, long N, float16 *dx, int xStride, float16 *dy, int yStride, float16 *dz, int zStride, float16 *extraA, float16 *extraB, float scalarA, float scalarB);

    void execMetaPredicateShapeHalf(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, long N, float16 *dx, int *xShapeInfo, float16 *dy, int *yShapeInfo, float16 *dz, int *zShapeInfo, float16 *extraA, float16 *extraB, float scalarA, float scalarB);


    void execMetaPredicateReduceFloat(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, float *dx, int *xShapeInfo, float *dy, int *yShapeInfo, float *dz, int *zShapeInfo, int *dimension, int dimensionLength, int *tadShapeInfo, int *tadOffsets, float *extraA, float *extraB, float scalarA, float scalarB, bool scalarReturned);
};


#endif //NATIVEOPERATIONS_NATIVEOPS_H
