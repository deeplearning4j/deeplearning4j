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

#include <dll.h>
#include <pointercast.h>



class ND4J_EXPORT NativeOps {
#ifdef __CUDACC__
cudaDeviceProp *deviceProperties;
        cudaFuncAttributes *funcAttributes;
#endif

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
        void   execBroadcastDouble(Nd4jPointer *extraPointers,int opNum,
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
        Nd4jPointer extraParams, int n);

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
        int n);

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
        int n,
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
        Nd4jPointer extraParams, int n);

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
        void   execBroadcastFloat(Nd4jPointer *extraPointers,int opNum,
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
        Nd4jPointer extraParams, int n);

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
        int n);

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
        Nd4jPointer extraParams, int n);

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
         * This method implementation exists only for cuda.
         * The other backends should have dummy method for JNI compatibility reasons.
         */
        void initializeDevicesAndFunctions();

};


#endif //NATIVEOPERATIONS_NATIVEOPS_H
