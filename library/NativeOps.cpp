//
// Created by agibsonccc on 1/30/16.
//
#include "NativeOpExcutioner.h"
#include <NativeOps.h>


class DoubleNativeOpExecutioner : public NativeOpExcutioner<double> {
private:
    static DoubleNativeOpExecutioner *INSTANCE;
public:
    static DoubleNativeOpExecutioner getInstance() {
        if(INSTANCE == NULL)
            INSTANCE = new DoubleNativeOpExecutioner();
        return INSTANCE;
    }
};

class FloatNativeOpExecutioner : public NativeOpExcutioner<float> {
private:
    static FloatNativeOpExecutioner *INSTANCE;
public:
    static FloatNativeOpExecutioner getInstance() {
        if(INSTANCE == NULL)
            INSTANCE = new FloatNativeOpExecutioner();
        return INSTANCE;
    }
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Class:     org_nd4j_linalg_cpu_ops_NativeOps
 * Method:    execIndexReduceScalar
 * Signature: (ILjava/nio/DoubleBuffer;Ljava/nio/IntBuffer;Ljava/nio/DoubleBuffer;)D
 */
JNIEXPORT jdouble JNICALL Java_org_nd4j_linalg_cpu_ops_NativeOps_execIndexReduceScalar__ILjava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_DoubleBuffer_2
        (JNIEnv *env, jobject obj, jint opNum, jobject x, jobject xShapeInfo, jobject extraParams) {
    int *xShapeBuff = env->GetDirectBufferAddress(xShapeInfo);
    double *extraParamsBuff = env->GetDirectBufferAddress(extraParams);
    double *x = env->GetDirectBufferAddress(x);
    return DoubleNativeOpExecutioner::getInstance().execIndexReduceScalar(opNum,x,xShapeBuff,extraParamsBuff);
}

/*
 * Class:     org_nd4j_linalg_cpu_ops_NativeOps
 * Method:    execIndexReduce
     * Signature: (ILjava/nio/DoubleBuffer;Ljava/nio/IntBuffer;Ljava/nio/DoubleBuffer;Ljava/nio/DoubleBuffer;Ljava/nio/IntBuffer;Ljava/nio/IntBuffer;I)V
 */
JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_ops_NativeOps_execIndexReduce__ILjava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_DoubleBuffer_2Ljava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_IntBuffer_2I
        (JNIEnv *env, jobject obj, jint opNum, jobject x, jobject xShapeInfo, jobject extraParams, jobject result, jobject resultShapeInfo, jobject dimension, jint dimensionLength) {
    int *xShapeBuff = env->GetDirectBufferAddress(xShapeInfo);
    double *xBuff = env->GetDirectBufferAddress(x);
    double *extraParamsBuff = env->GetDirectBufferAddress(extraParams);
    double *resultBuff = env->GetDirectBufferAddress(result);
    int *resultShapeBuff = env->GetDirectBufferAddress(resultShapeInfo);
    int *dimensionBuff = env->GetDirectBufferAddress(dimension);
    DoubleNativeOpExecutioner::getInstance().execIndexReduce(opNum,xBuff,xShapeBuff,extraParamsBuff,resultBuff,resultShapeBuff,dimensionBuff,dimensionLength);
}

/*
 * Class:     org_nd4j_linalg_cpu_ops_NativeOps
 * Method:    execBroadcast
 * Signature: (ILjava/nio/DoubleBuffer;Ljava/nio/IntBuffer;Ljava/nio/DoubleBuffer;Ljava/nio/IntBuffer;Ljava/nio/DoubleBuffer;Ljava/nio/IntBuffer;Ljava/nio/IntBuffer;I)V
 */
JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_ops_NativeOps_execBroadcast__ILjava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_IntBuffer_2I
        (JNIEnv *env, jobject obj, jint opNum,
         jobject x, jobject xShapeInfo, jobject y,
         jobject yShapeInfo, jobject result, jobject resultShapeInfo, jobject dimension, jint dimensionLength) {
    double *xBuff = env->GetDirectBufferAddress(x);
    int *xShapeBuff = env->GetDirectBufferAddress(xShapeInfo);
    double *yBuff = env->GetDirectBufferAddress(y);
    int *yShapeBuff = env->GetDirectBufferAddress(yShapeInfo);
    double *resultBuff = env->GetDirectBufferAddress(result);
    int *resultShapeBuff = env->GetDirectBufferAddress(resultShapeInfo);
    int *dimensionBuff = env->GetDirectBufferAddress(dimension);

    DoubleNativeOpExecutioner::getInstance().execBroadcast(
            opNum,xBuff,xShapeBuff,yBuff,yShapeBuff,resultBuff,resultShapeBuff,dimensionBuff,dimensionLength);
}

/*
 * Class:     org_nd4j_linalg_cpu_ops_NativeOps
 * Method:    execPairwiseTransform
 * Signature: (ILjava/nio/DoubleBuffer;ILjava/nio/DoubleBuffer;ILjava/nio/DoubleBuffer;ILjava/nio/DoubleBuffer;I)V
 */
JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_ops_NativeOps_execPairwiseTransform__ILjava_nio_DoubleBuffer_2ILjava_nio_DoubleBuffer_2ILjava_nio_DoubleBuffer_2ILjava_nio_DoubleBuffer_2I
        (JNIEnv *env, jobject obj, jint opNum, jobject dx, jint xStride,
         jobject y, jint yStride, jobject result, jint resultStride, jobject extraParams, jint n) {
    double *dxBuff = env->GetDirectBufferAddress(dx);
    double *yBuff = env->GetDirectBufferAddress(y);
    double *resultBuff = env->GetDirectBufferAddress(result);
    double *extraParamsBuff = env->GetDirectBufferAddress(extraParams);
    DoubleNativeOpExecutioner::getInstance()->execPairwiseTransform(opNum,
                                                                    dxBuff,
                                                                    xStride,
                                                                    yBuff,
                                                                    yStride,
                                                                    resultBuff,
                                                                    resultStride,
                                                                    extraParamsBuff,n);
}

/*
 * Class:     org_nd4j_linalg_cpu_ops_NativeOps
 * Method:    execReduce
 * Signature: (ILjava/nio/DoubleBuffer;Ljava/nio/IntBuffer;Ljava/nio/DoubleBuffer;Ljava/nio/DoubleBuffer;Ljava/nio/IntBuffer;)V
 */
JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_ops_NativeOps_execReduce__ILjava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_DoubleBuffer_2Ljava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2
        (JNIEnv *env, jobject obj, jint opNum, jobject x,
        		jobject xShapeInfo,
        		jobject extraParams,
        		jobject result,
        		object resultShapeInfo) {
    double *xBuff = env->GetDirectBufferAddress(x);
    int *xShapeBuff = env->GetDirectBufferAddress(xShapeInfo);
    double *extraParamsBuff = env->GetDirectBufferAddress(extraParams);
    double *resultBuff = env->GetDirectBufferAddress(result);
    int *resultShapeBuff = env->GetDirectBufferAddress(resultShapeInfo);
    DoubleNativeOpExecutioner::getInstance().execReduce(opNum,
                                                        xBuff,
                                                        xShapeBuff,
                                                        extraParamsBuff,resultBuff,resultShapeBuff);
}

/*
 * Class:     org_nd4j_linalg_cpu_ops_NativeOps
 * Method:    execReduce
 * Signature: (ILjava/nio/DoubleBuffer;Ljava/nio/IntBuffer;Ljava/nio/DoubleBuffer;Ljava/nio/DoubleBuffer;Ljava/nio/IntBuffer;Ljava/nio/IntBuffer;I)V
 */
JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_ops_NativeOps_execReduce__ILjava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_DoubleBuffer_2Ljava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_IntBuffer_2I
        (JNIEnv *env, jobject obj,
         jint opNum,
         jobject x,
         jobject xShapeInfo,
         jobject extraParams,
         jobject result, jobject resultShapeInfo, jobject dimension, jint dimensionLength) {
    double *xBuff = env->GetDirectBufferAddress(x);
    int *xShapeBuff = env->GetDirectBufferAddress(xShapeInfo);
    double *extraParamsBuff = env->GetDirectBufferAddress(extraParams);
    double *resultBuff = env->GetDirectBufferAddress(result);
    int *resultShapeBuff = env->GetDirectBufferAddress(resultShapeInfo);
    int *dimensionBuff = env->GetDirectBufferAddress(dimension);

    DoubleNativeOpExecutioner::getInstance().execReduce(opNum,xBuff,xShapeBuff,extraParamsBuff,
                                                        resultBuff,resultShapeBuff,dimensionBuff,dimensionLength);

}

/*
 * Class:     org_nd4j_linalg_cpu_ops_NativeOps
 * Method:    execReduceScalar
 * Signature: (ILjava/nio/DoubleBuffer;Ljava/nio/IntBuffer;Ljava/nio/DoubleBuffer;)D
 */
JNIEXPORT jdouble JNICALL Java_org_nd4j_linalg_cpu_ops_NativeOps_execReduceScalar__ILjava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_DoubleBuffer_2
        (JNIEnv *env, jobject obj, jint opNum, jobject x, jobject xShapeInfo, jobject extraParams) {
    double *xBuff = env->GetDirectBufferAddress(x);
    int *xShapeBuff = env->GetDirectBufferAddress(xShapeInfo);
    double *extraParamsBuff = env->GetDirectBufferAddress(extraParams);
    return DoubleNativeOpExecutioner::getInstance().execReduceScalar(opNum,xBuff,xShapeBuff,extraParamsBuff);

}

/*
 * Class:     org_nd4j_linalg_cpu_ops_NativeOps
 * Method:    execReduce3
 * Signature: (ILjava/nio/DoubleBuffer;Ljava/nio/IntBuffer;Ljava/nio/DoubleBuffer;Ljava/nio/DoubleBuffer;Ljava/nio/IntBuffer;Ljava/nio/DoubleBuffer;Ljava/nio/IntBuffer;)V
 */
JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_ops_NativeOps_execReduce3__ILjava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_DoubleBuffer_2Ljava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2
        (JNIEnv *env,
         jobject obj,
         jint opNum,
         jobject x, jobject xShapeInfo,
         jobject extraParams,
         jobject y,
         jobject yShapeInfo, jobject result, jobject resultShapeInfo) {

}

/*
 * Class:     org_nd4j_linalg_cpu_ops_NativeOps
 * Method:    execReduce3Scalar
 * Signature: (ILjava/nio/DoubleBuffer;Ljava/nio/IntBuffer;Ljava/nio/DoubleBuffer;Ljava/nio/DoubleBuffer;Ljava/nio/IntBuffer;)D
 */
JNIEXPORT jdouble JNICALL Java_org_nd4j_linalg_cpu_ops_NativeOps_execReduce3Scalar__ILjava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_DoubleBuffer_2Ljava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2
        (JNIEnv *env, jobject obj,
         jint opNum, jobject x,
         jobject xShapeInfo,
         jobject extraParams,
         jobject y,
         jobject yShapeInfo) {
    double *xBuff = env->GetDirectBufferAddress(x);
    int *xShapeBuff = env->GetDirectBufferAddress(xShapeInfo);
    double *extraParamsBuff = env->GetDirectBufferAddress(extraParams);
    double *yBuff = env->GetDirectBufferAddress(y);
    int *yShapeBuff = env->GetDirectBufferAddress(yShapeInfo);
    return DoubleNativeOpExecutioner::getInstance().execReduce3(opNum,xBuff,xShapeBuff,
                                                                extraParamsBuff,yBuff,yShapeBuff);
}

/*
 * Class:     org_nd4j_linalg_cpu_ops_NativeOps
 * Method:    execReduce3
 * Signature: (ILjava/nio/DoubleBuffer;Ljava/nio/IntBuffer;Ljava/nio/DoubleBuffer;Ljava/nio/DoubleBuffer;Ljava/nio/IntBuffer;Ljava/nio/DoubleBuffer;Ljava/nio/IntBuffer;Ljava/nio/IntBuffer;I)V
 */
JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_ops_NativeOps_execReduce3__ILjava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_DoubleBuffer_2Ljava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_IntBuffer_2I
        (JNIEnv *env, jobject obj,
         jint opNum,
         jobject x,
         jobject xShapeInfo,
         jobject extraParams,
         jobject y,
         jobject yShapeInfo,
         jobject result,
         jobject resultShapeInfo,
         jobject dimension,
         jint dimensionLength) {
    double *xBuff = env->GetDirectBufferAddress(x);
    int *xShapeBuff = env->GetDirectBufferAddress(xShapeInfo);
    double *extraParamsBuff = env->GetDirectBufferAddress(extraParams);
    double *yBuff = env->GetDirectBufferAddress(y);
    int *yShapeBuff = env->GetDirectBufferAddress(yShapeInfo);
    double *resultBuff = env->GetDirectBufferAddress(result);
    int *resultShapeBuff = env->GetDirectBufferAddress(resultShapeInfo);
    int *dimensionBuff = env->GetDirectBufferAddress(dimension);
    DoubleNativeOpExecutioner::getInstance().execReduce3(opNum,xBuff,xShapeBuff,extraParamsBuff,yBuff,yShapeBuff,resultBuff,resultShapeBuff,dimensionBuff,dimensionLength);

}

/*
 * Class:     org_nd4j_linalg_cpu_ops_NativeOps
 * Method:    execScalar
 * Signature: (ILjava/nio/DoubleBuffer;ILjava/nio/DoubleBuffer;IDLjava/nio/DoubleBuffer;I)D
 */
JNIEXPORT jdouble JNICALL Java_org_nd4j_linalg_cpu_ops_NativeOps_execScalar__ILjava_nio_DoubleBuffer_2ILjava_nio_DoubleBuffer_2IDLjava_nio_DoubleBuffer_2I
        (JNIEnv *env, jobject obj,
         jint opNum,
         jobject x,
         jint xStride,
         jobject result,
         jint resultStride,
         jdouble scalar,
         jobject extraParams,
         jint n) {
    double *xBuff = env->GetDirectBufferAddress(x);
    double *resultBuff = env->GetDirectBufferAddress(result);
    double *extraParamsBuff = env->GetDirectBufferAddress(extraParams);
    DoubleNativeOpExecutioner::getInstance().execScalar(opNum,
                                                        xBuff,xStride,
                                                        resultBuff,resultStride,
                                                        scalar,extraParamsBuff,n);

}

/*
 * Class:     org_nd4j_linalg_cpu_ops_NativeOps
 * Method:    execSummaryStatsScalar
 * Signature: (ILjava/nio/DoubleBuffer;Ljava/nio/IntBuffer;Ljava/nio/DoubleBuffer;)D
 */
JNIEXPORT jdouble JNICALL Java_org_nd4j_linalg_cpu_ops_NativeOps_execSummaryStatsScalar__ILjava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_DoubleBuffer_2
        (JNIEnv *env,
         jobject obj,
         jint opNum,
         jobject x, jobject xShapeInfo, jobject extraParams) {
    double *xBuff = env->GetDirectBufferAddress(x);
    int *xShapeBuff = env->GetDirectBufferAddress(xShapeInfo);
    double *extraParamsBuff = env->GetDirectBufferAddress(extraParams);
    return DoubleNativeOpExecutioner::getInstance().execSummaryStatsScalar(opNum,xBuff,xShapeBuff);
}

/*
 * Class:     org_nd4j_linalg_cpu_ops_NativeOps
 * Method:    execSummaryStats
 * Signature: (ILjava/nio/DoubleBuffer;Ljava/nio/IntBuffer;Ljava/nio/DoubleBuffer;Ljava/nio/DoubleBuffer;Ljava/nio/IntBuffer;)V
 */
JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_ops_NativeOps_execSummaryStats__ILjava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_DoubleBuffer_2Ljava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2
        (JNIEnv *env, jobject obj, jint opNum,
         jobject x,
         jobject xShapeInfo, jobject extraParams,
         jobject result, jobject resultShapeInfo) {
    double *xBuff = env->GetDirectBufferAddress(x);
    int *xShapeBuff = env->GetDirectBufferAddress(xShapeInfo);
    double *extraParamsBuff = env->GetDirectBufferAddress(extraParams);
    double *resultBuff = env->GetDirectBufferAddress(result);
    int *resultShapeBuff = env->GetDirectBufferAddress(resultShapeInfo);

}

/*
 * Class:     org_nd4j_linalg_cpu_ops_NativeOps
 * Method:    execSummaryStats
 * Signature: (ILjava/nio/DoubleBuffer;Ljava/nio/IntBuffer;Ljava/nio/DoubleBuffer;Ljava/nio/DoubleBuffer;Ljava/nio/IntBuffer;Ljava/nio/IntBuffer;I)V
 */
JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_ops_NativeOps_execSummaryStats__ILjava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_DoubleBuffer_2Ljava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_IntBuffer_2I
        (JNIEnv *env, jobject obj,
         jint opNum,
         jobject x,
         jobject xShapeInfo,
         jobject extraParams,
         jobject result,
         jobject resultShapeInfo, jobject dimension, jint dimensionLength) {
    double *xBuff = env->GetDirectBufferAddress(x);
    int *xShapeBuff = env->GetDirectBufferAddress(xShapeInfo);
    double *extraParamsBuff = env->GetDirectBufferAddress(extraParams);
    double *resultBuff = env->GetDirectBufferAddress(result);
    int *resultShapeBuff = env->GetDirectBufferAddress(resultShapeInfo);
    int *dimensionBuff = env->GetDirectBufferAddress(dimension);
    DoubleNativeOpExecutioner::getInstance().execSummaryStats(opNum,xBuff,xShapeBuff,extraParamsBuff,resultBuff,resultShapeBuff,dimensionBuff,dimensionLength);
}

/*
 * Class:     org_nd4j_linalg_cpu_ops_NativeOps
 * Method:    execTransform
 * Signature: (ILjava/nio/DoubleBuffer;ILjava/nio/DoubleBuffer;ILjava/nio/DoubleBuffer;I)V
 */
JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_ops_NativeOps_execTransform__ILjava_nio_DoubleBuffer_2ILjava_nio_DoubleBuffer_2ILjava_nio_DoubleBuffer_2I
        (JNIEnv *env,
         jobject obj,
         jint opNum,
         jobject x, jint xStride,
         jobject result,
         jint resultStride, jobject extraParams,
         jint n) {
    double *xBuff = env->GetDirectBufferAddress(x);
    double *extraParamsBuff = env->GetDirectBufferAddress(extraParams);

    double *resultBuff = env->GetDirectBufferAddress(result);
    DoubleNativeOpExecutioner::getInstance().execTransform(opNum,xBuff,xStride,resultBuff,
                                                           resultStride,extraParamsBuff,n);


}

/*
 * Class:     org_nd4j_linalg_cpu_ops_NativeOps
 * Method:    execIndexReduceScalar
 * Signature: (ILjava/nio/FloatBuffer;Ljava/nio/IntBuffer;Ljava/nio/FloatBuffer;)F
 */
JNIEXPORT jfloat JNICALL Java_org_nd4j_linalg_cpu_ops_NativeOps_execIndexReduceScalar__ILjava_nio_FloatBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_FloatBuffer_2
        (JNIEnv *env, jobject obj, jint opNum, jobject x, jobject xShapeInfo, jobject extraParams) {
    int *xShapeBuff = env->GetDirectBufferAddress(xShapeInfo);
    float *extraParamsBuff = env->GetDirectBufferAddress(extraParams);
    float *x = env->GetDirectBufferAddress(x);
    return FloatNativeOpExecutioner::getInstance().execIndexReduceScalar(opNum,x,xShapeBuff,extraParamsBuff);

}

/*
 * Class:     org_nd4j_linalg_cpu_ops_NativeOps
 * Method:    execIndexReduce
 * Signature: (ILjava/nio/FloatBuffer;Ljava/nio/IntBuffer;Ljava/nio/FloatBuffer;Ljava/nio/FloatBuffer;Ljava/nio/IntBuffer;Ljava/nio/IntBuffer;I)V
 */
JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_ops_NativeOps_execIndexReduce__ILjava_nio_FloatBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_FloatBuffer_2Ljava_nio_FloatBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_IntBuffer_2I
        (JNIEnv *env, jobject obj, jint opNum,
         jobject x, jobject xShapeInfo,
         jobject extraParams, jobject result, jobject resultShapeInfo, jobject dimension, jint dimensionLength) {
    int *xShapeBuff = env->GetDirectBufferAddress(xShapeInfo);
    float *xBuff = env->GetDirectBufferAddress(x);
    float *extraParamsBuff = env->GetDirectBufferAddress(extraParams);
    float *resultBuff = env->GetDirectBufferAddress(result);
    int *resultShapeBuff = env->GetDirectBufferAddress(resultShapeInfo);
    int *dimensionBuff = env->GetDirectBufferAddress(dimension);
    FloatNativeOpExecutioner::getInstance().execIndexReduce(opNum,xBuff,xShapeBuff,extraParamsBuff,resultBuff,resultShapeBuff,dimensionBuff,dimensionLength);

}

/*
 * Class:     org_nd4j_linalg_cpu_ops_NativeOps
 * Method:    execBroadcast
 * Signature: (ILjava/nio/FloatBuffer;Ljava/nio/IntBuffer;Ljava/nio/FloatBuffer;Ljava/nio/IntBuffer;Ljava/nio/FloatBuffer;Ljava/nio/IntBuffer;Ljava/nio/IntBuffer;I)V
 */
JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_ops_NativeOps_execBroadcast__ILjava_nio_FloatBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_FloatBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_FloatBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_IntBuffer_2I
        (JNIEnv *env, jobject obj, jint opNum, jobject x,
         jobject xShapeInfo,
         jobject y,
         jobject yShapeInfo, jobject result,
         jobject resultShapeInfo,
         jobject dimension, jint dimensionLength) {
    float *xBuff = env->GetDirectBufferAddress(x);
    int *xShapeBuff = env->GetDirectBufferAddress(xShapeInfo);
    float *yBuff = env->GetDirectBufferAddress(y);
    int *yShapeBuff = env->GetDirectBufferAddress(yShapeInfo);
    float *resultBuff = env->GetDirectBufferAddress(result);
    int *resultShapeBuff = env->GetDirectBufferAddress(resultShapeInfo);
    int *dimensionBuff = env->GetDirectBufferAddress(dimension);

    FloatNativeOpExecutioner::getInstance().execBroadcast(
            opNum,xBuff,
            xShapeBuff,
            yBuff,
            yShapeBuff,
            resultBuff,
            resultShapeBuff,dimensionBuff,dimensionLength);
}

/*
 * Class:     org_nd4j_linalg_cpu_ops_NativeOps
 * Method:    execPairwiseTransform
 * Signature: (ILjava/nio/FloatBuffer;ILjava/nio/FloatBuffer;ILjava/nio/FloatBuffer;ILjava/nio/FloatBuffer;I)V
 */
JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_ops_NativeOps_execPairwiseTransform__ILjava_nio_FloatBuffer_2ILjava_nio_FloatBuffer_2ILjava_nio_FloatBuffer_2ILjava_nio_FloatBuffer_2I
        (JNIEnv *env,
         jobject obj,
         jint opNum,
         jobject dx, jint xStride,
         jobject y, jint yStride,
         jobject result,
         jint resultStride, jobject extraParams, jint n) {
    float *dxBuff = env->GetDirectBufferAddress(dx);
    float *yBuff = env->GetDirectBufferAddress(y);
    float *resultBuff = env->GetDirectBufferAddress(result);
    float *extraParamsBuff = env->GetDirectBufferAddress(extraParams);
    FloatNativeOpExecutioner::getInstance()->execPairwiseTransform(opNum,
                                                                    dxBuff,
                                                                    xStride,
                                                                    yBuff,
                                                                    yStride,
                                                                    resultBuff,
                                                                    resultStride,
                                                                    extraParamsBuff,n);

}

/*
 * Class:     org_nd4j_linalg_cpu_ops_NativeOps
 * Method:    execReduce
 * Signature: (ILjava/nio/FloatBuffer;Ljava/nio/IntBuffer;Ljava/nio/FloatBuffer;Ljava/nio/FloatBuffer;Ljava/nio/IntBuffer;Ljava/nio/IntBuffer;I)V
 */
JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_ops_NativeOps_execReduce__ILjava_nio_FloatBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_FloatBuffer_2Ljava_nio_FloatBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_IntBuffer_2I
        (JNIEnv *env, jobject obj,
         jint opNum, jobject x,
         jobject xShapeInfo, jobject extraParams,
         jobject result, jobject resultShapeInfo,
         jobject dimension, jint dimensionLength) {
    float *xBuff = env->GetDirectBufferAddress(x);
    int *xShapeBuff = env->GetDirectBufferAddress(xShapeInfo);
    float *extraParamsBuff = env->GetDirectBufferAddress(extraParams);
    float *resultBuff = env->GetDirectBufferAddress(result);
    int *resultShapeBuff = env->GetDirectBufferAddress(resultShapeInfo);
    FloatNativeOpExecutioner::getInstance().execReduce(opNum,
                                                        xBuff,
                                                        xShapeBuff,
                                                        extraParamsBuff,resultBuff,resultShapeBuff);


}

/*
 * Class:     org_nd4j_linalg_cpu_ops_NativeOps
 * Method:    execReduceScalar
 * Signature: (ILjava/nio/FloatBuffer;Ljava/nio/IntBuffer;Ljava/nio/FloatBuffer;)F
 */
JNIEXPORT jfloat JNICALL Java_org_nd4j_linalg_cpu_ops_NativeOps_execReduceScalar__ILjava_nio_FloatBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_FloatBuffer_2
        (JNIEnv *env, jobject obj, jint opNum, jobject, jobject, jobject) {
    float *xBuff = env->GetDirectBufferAddress(x);
    int *xShapeBuff = env->GetDirectBufferAddress(xShapeInfo);
    float *extraParamsBuff = env->GetDirectBufferAddress(extraParams);
    return FloatNativeOpExecutioner::getInstance().execReduceScalar(opNum,xBuff,xShapeBuff,extraParamsBuff);


}

/*
 * Class:     org_nd4j_linalg_cpu_ops_NativeOps
 * Method:    execReduce3
 * Signature: (ILjava/nio/FloatBuffer;Ljava/nio/IntBuffer;Ljava/nio/FloatBuffer;Ljava/nio/FloatBuffer;Ljava/nio/IntBuffer;Ljava/nio/FloatBuffer;Ljava/nio/IntBuffer;)V
 */
JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_ops_NativeOps_execReduce3__ILjava_nio_FloatBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_FloatBuffer_2Ljava_nio_FloatBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_FloatBuffer_2Ljava_nio_IntBuffer_2
        (JNIEnv *env, jobject obj, jint opNum,
         jobject x,
         jobject xShapeInfo,
         jobject extraParams,
         jobject y,
         jobject yShapeInfo,
         jobject result,
         jobject resultShapeInfo) {

}

/*
 * Class:     org_nd4j_linalg_cpu_ops_NativeOps
 * Method:    execReduce3Scalar
 * Signature: (ILjava/nio/FloatBuffer;Ljava/nio/IntBuffer;Ljava/nio/FloatBuffer;Ljava/nio/FloatBuffer;Ljava/nio/IntBuffer;)F
 */
JNIEXPORT jfloat JNICALL Java_org_nd4j_linalg_cpu_ops_NativeOps_execReduce3Scalar__ILjava_nio_FloatBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_FloatBuffer_2Ljava_nio_FloatBuffer_2Ljava_nio_IntBuffer_2
        (JNIEnv *env,
         jobject obj,
         jint opNum,
         jobject x,
         jobject xShapeInfo,
         jobject extraParams,
         jobject y,
         jobject yShapeInfo) {
    float *xBuff = env->GetDirectBufferAddress(x);
    int *xShapeBuff = env->GetDirectBufferAddress(xShapeInfo);
    float *extraParamsBuff = env->GetDirectBufferAddress(extraParams);
    float *yBuff = env->GetDirectBufferAddress(y);
    int *yShapeBuff = env->GetDirectBufferAddress(yShapeInfo);
    return FloatNativeOpExecutioner::getInstance().execReduce3(opNum,xBuff,xShapeBuff,
                                                                extraParamsBuff,yBuff,yShapeBuff);

}

/*
 * Class:     org_nd4j_linalg_cpu_ops_NativeOps
 * Method:    execReduce3
 * Signature: (ILjava/nio/FloatBuffer;Ljava/nio/IntBuffer;Ljava/nio/FloatBuffer;Ljava/nio/FloatBuffer;Ljava/nio/IntBuffer;Ljava/nio/FloatBuffer;Ljava/nio/IntBuffer;Ljava/nio/IntBuffer;I)V
 */
JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_ops_NativeOps_execReduce3__ILjava_nio_FloatBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_FloatBuffer_2Ljava_nio_FloatBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_FloatBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_IntBuffer_2I
        (JNIEnv *env, jobject obj,
         jint opNum,
         jobject x,
         jobject xShapeInfo,
         jobject extraParams,
         jobject y,
         jobject yShapeInfo,
         jobject result,
         jobject resultShapeInfo,
         jobject dimension,
         jint dimensionLength) {
    float *xBuff = env->GetDirectBufferAddress(x);
    int *xShapeBuff = env->GetDirectBufferAddress(xShapeInfo);
    float *extraParamsBuff = env->GetDirectBufferAddress(extraParams);
    float *yBuff = env->GetDirectBufferAddress(y);
    int *yShapeBuff = env->GetDirectBufferAddress(yShapeInfo);
    float *resultBuff = env->GetDirectBufferAddress(result);
    int *resultShapeBuff = env->GetDirectBufferAddress(resultShapeInfo);
    int *dimensionBuff = env->GetDirectBufferAddress(dimension);
    FloatNativeOpExecutioner::getInstance().execReduce3(opNum,xBuff,xShapeBuff,extraParamsBuff,yBuff,yShapeBuff,resultBuff,resultShapeBuff,dimensionBuff,dimensionLength);

}

/*
 * Class:     org_nd4j_linalg_cpu_ops_NativeOps
 * Method:    execScalar
 * Signature: (ILjava/nio/FloatBuffer;ILjava/nio/FloatBuffer;IFLjava/nio/FloatBuffer;I)F
 */
JNIEXPORT jfloat JNICALL Java_org_nd4j_linalg_cpu_ops_NativeOps_execScalar__ILjava_nio_FloatBuffer_2ILjava_nio_FloatBuffer_2IFLjava_nio_FloatBuffer_2I
        (JNIEnv *env, jobject obj, jint opNum,
         jobject x, jint xStride,
         jobject result,
         jint resultStride,
         jfloat scalar,
         jobject extraParams, jint n) {
    float *xBuff = env->GetDirectBufferAddress(x);
    int *xShapeBuff = env->GetDirectBufferAddress(xShapeInfo);
    float *extraParamsBuff = env->GetDirectBufferAddress(extraParams);
    float *yBuff = env->GetDirectBufferAddress(y);
    int *yShapeBuff = env->GetDirectBufferAddress(yShapeInfo);
    float *resultBuff = env->GetDirectBufferAddress(result);
    int *resultShapeBuff = env->GetDirectBufferAddress(resultShapeInfo);
    int *dimensionBuff = env->GetDirectBufferAddress(dimension);
    FloatNativeOpExecutioner::getInstance().execReduce3(opNum,xBuff,xShapeBuff,extraParamsBuff,yBuff,yShapeBuff,resultBuff,resultShapeBuff,dimensionBuff,dimensionLength);

}

/*
 * Class:     org_nd4j_linalg_cpu_ops_NativeOps
 * Method:    execSummaryStatsScalar
 * Signature: (ILjava/nio/FloatBuffer;Ljava/nio/IntBuffer;Ljava/nio/FloatBuffer;)F
 */
JNIEXPORT jfloat JNICALL Java_org_nd4j_linalg_cpu_ops_NativeOps_execSummaryStatsScalar__ILjava_nio_FloatBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_FloatBuffer_2
        (JNIEnv *env, jobject obj, jint opNum, jobject x, jobject xShapeInfo, jobject extraParams) {
    float *xBuff = env->GetDirectBufferAddress(x);
    int *xShapeBuff = env->GetDirectBufferAddress(xShapeInfo);
    float *extraParamsBuff = env->GetDirectBufferAddress(extraParams);
    return FloatNativeOpExecutioner::getInstance().execSummaryStatsScalar(opNum,xBuff,xShapeBuff);

}

/*
 * Class:     org_nd4j_linalg_cpu_ops_NativeOps
 * Method:    execSummaryStats
 * Signature: (ILjava/nio/FloatBuffer;Ljava/nio/IntBuffer;Ljava/nio/FloatBuffer;Ljava/nio/FloatBuffer;Ljava/nio/IntBuffer;)V
 */
JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_ops_NativeOps_execSummaryStats__ILjava_nio_FloatBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_FloatBuffer_2Ljava_nio_FloatBuffer_2Ljava_nio_IntBuffer_2
        (JNIEnv *env, jobject obj, jint opNum, jobject, jobject, jobject, jobject, jobject) {
    //no op
}

/*
 * Class:     org_nd4j_linalg_cpu_ops_NativeOps
 * Method:    execSummaryStats
 * Signature: (ILjava/nio/FloatBuffer;Ljava/nio/IntBuffer;Ljava/nio/FloatBuffer;Ljava/nio/FloatBuffer;Ljava/nio/IntBuffer;Ljava/nio/IntBuffer;I)V
 */
JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_ops_NativeOps_execSummaryStats__ILjava_nio_FloatBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_FloatBuffer_2Ljava_nio_FloatBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_IntBuffer_2I
        (JNIEnv *env,
         jobject obj,
         jint, jobject,
         jobject,
         jobject, jobject, jobject, jobject, jint) {
    float *xBuff = env->GetDirectBufferAddress(x);
    int *xShapeBuff = env->GetDirectBufferAddress(xShapeInfo);
    float *extraParamsBuff = env->GetDirectBufferAddress(extraParams);
    float *resultBuff = env->GetDirectBufferAddress(result);
    int *resultShapeBuff = env->GetDirectBufferAddress(resultShapeInfo);
    int *dimensionBuff = env->GetDirectBufferAddress(dimension);
    FloatNativeOpExecutioner::getInstance().execSummaryStats(opNum,xBuff,xShapeBuff,extraParamsBuff,resultBuff,resultShapeBuff,dimensionBuff,dimensionLength);

}

/*
 * Class:     org_nd4j_linalg_cpu_ops_NativeOps
 * Method:    execTransform
 * Signature: (ILjava/nio/FloatBuffer;ILjava/nio/FloatBuffer;ILjava/nio/FloatBuffer;I)V
 */
JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_ops_NativeOps_execTransform__ILjava_nio_FloatBuffer_2ILjava_nio_FloatBuffer_2ILjava_nio_FloatBuffer_2I
        (JNIEnv *env, jobject obj, jint opNum, jobject x, jint xStride, jobject result, jint resultStride,
         jobject extraParams, jint n) {
    float *xBuff = env->GetDirectBufferAddress(x);
    float *extraParamsBuff = env->GetDirectBufferAddress(extraParams);

    float *resultBuff = env->GetDirectBufferAddress(result);
    FloatNativeOpExecutioner::getInstance().execTransform(opNum,xBuff,xStride,resultBuff,
                                                           resultStride,extraParamsBuff,n);



}

#ifdef __cplusplus
}
#endif
