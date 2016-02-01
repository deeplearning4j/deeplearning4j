//
// Created by agibsonccc on 1/30/16.
//
#include "NativeOps.h"
#include "NativeOpExcutioner.h"

class DoubleNativeOpExecutioner : public NativeOpExcutioner<double> {
public:
    static DoubleNativeOpExecutioner  getInstance() {
        DoubleNativeOpExecutioner INSTANCE;
        return INSTANCE;
    }
};

class FloatNativeOpExecutioner : public NativeOpExcutioner<float> {
public:
    static FloatNativeOpExecutioner  getInstance() {
        FloatNativeOpExecutioner INSTANCE;
        return INSTANCE;
    }
};

#ifdef __cplusplus
extern "C" {
#endif





/*
 * Class:     org_nd4j_linalg_cpu_nativecpu_ops_NativeOps
 * Method:    execIndexReduceScalar
 * Signature: (ILjava/nio/DoubleBuffer;Ljava/nio/IntBuffer;Ljava/nio/DoubleBuffer;)D
 */
JNIEXPORT jdouble JNICALL Java_org_nd4j_linalg_cpu_nativecpu_ops_NativeOps_execIndexReduceScalar__ILjava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_DoubleBuffer_2
        (JNIEnv *env, jobject obj, jint opNum, jobject x, jobject xShapeInfo, jobject extraParams) {
    int *xShapeBuff = (int *) env->GetDirectBufferAddress(xShapeInfo);
    double *extraParamsBuff = extraParams != NULL ? (double *) env->GetDirectBufferAddress(extraParams) : NULL;
    double *xBuff = (double *) env->GetDirectBufferAddress(x);
    return DoubleNativeOpExecutioner::getInstance().execIndexReduceScalar(opNum,xBuff,xShapeBuff,extraParamsBuff);
}

/*
 * Class:     org_nd4j_linalg_cpu_nativecpu_ops_NativeOps
 * Method:    execIndexReduce
     * Signature: (ILjava/nio/DoubleBuffer;Ljava/nio/IntBuffer;Ljava/nio/DoubleBuffer;Ljava/nio/DoubleBuffer;Ljava/nio/IntBuffer;Ljava/nio/IntBuffer;I)V
 */
JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_ops_NativeOps_execIndexReduce__ILjava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_DoubleBuffer_2Ljava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_IntBuffer_2I
        (JNIEnv *env, jobject obj, jint opNum, jobject x, jobject xShapeInfo, jobject extraParams, jobject result, jobject resultShapeInfo, jobject dimension, jint dimensionLength) {
    int *xShapeBuff = (int *) env->GetDirectBufferAddress(xShapeInfo);
    double *xBuff = (double *) env->GetDirectBufferAddress(x);
    double *extraParamsBuff = extraParams != NULL ? (double *) env->GetDirectBufferAddress(extraParams) : NULL;
    double *resultBuff = (double *) env->GetDirectBufferAddress(result);
    int *resultShapeBuff = (int *) env->GetDirectBufferAddress(resultShapeInfo);
    int *dimensionBuff = (int *) env->GetDirectBufferAddress(dimension);
    DoubleNativeOpExecutioner::getInstance().execIndexReduce(opNum,xBuff,xShapeBuff,extraParamsBuff,resultBuff,resultShapeBuff,dimensionBuff,dimensionLength);
}

/*
 * Class:     org_nd4j_linalg_cpu_nativecpu_ops_NativeOps
 * Method:    execBroadcast
 * Signature: (ILjava/nio/DoubleBuffer;Ljava/nio/IntBuffer;Ljava/nio/DoubleBuffer;Ljava/nio/IntBuffer;Ljava/nio/DoubleBuffer;Ljava/nio/IntBuffer;Ljava/nio/IntBuffer;I)V
 */
JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_ops_NativeOps_execBroadcast__ILjava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_IntBuffer_2I
        (JNIEnv *env, jobject obj, jint opNum,
         jobject x, jobject xShapeInfo, jobject y,
         jobject yShapeInfo, jobject result, jobject resultShapeInfo, jobject dimension, jint dimensionLength) {
    double *xBuff = (double *) env->GetDirectBufferAddress(x);
    int *xShapeBuff = (int *) env->GetDirectBufferAddress(xShapeInfo);
    double *yBuff = (double *) env->GetDirectBufferAddress(y);
    int *yShapeBuff = (int *) env->GetDirectBufferAddress(yShapeInfo);
    double *resultBuff = (double *) env->GetDirectBufferAddress(result);
    int *resultShapeBuff = (int *) env->GetDirectBufferAddress(resultShapeInfo);
    int *dimensionBuff = (int *) env->GetDirectBufferAddress(dimension);

    DoubleNativeOpExecutioner::getInstance().execBroadcast(
            opNum,xBuff,xShapeBuff,yBuff,yShapeBuff,resultBuff,resultShapeBuff,dimensionBuff,dimensionLength);
}

/*
 * Class:     org_nd4j_linalg_cpu_nativecpu_ops_NativeOps
 * Method:    execPairwiseTransform
 * Signature: (ILjava/nio/DoubleBuffer;ILjava/nio/DoubleBuffer;ILjava/nio/DoubleBuffer;ILjava/nio/DoubleBuffer;I)V
 */
JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_ops_NativeOps_execPairwiseTransform__ILjava_nio_DoubleBuffer_2ILjava_nio_DoubleBuffer_2ILjava_nio_DoubleBuffer_2ILjava_nio_DoubleBuffer_2I
        (JNIEnv *env, jobject obj, jint opNum, jobject dx, jint xStride,
         jobject y, jint yStride, jobject result, jint resultStride, jobject extraParams, jint n) {
    double *dxBuff = (double *) env->GetDirectBufferAddress(dx);
    double *yBuff = (double *) env->GetDirectBufferAddress(y);
    double *resultBuff = (double *) env->GetDirectBufferAddress(result);
    double *extraParamsBuff = extraParams != NULL ? (double *) env->GetDirectBufferAddress(extraParams) : NULL;
    DoubleNativeOpExecutioner::getInstance().execPairwiseTransform(opNum,
                                                                   dxBuff,
                                                                   xStride,
                                                                   yBuff,
                                                                   yStride,
                                                                   resultBuff,
                                                                   resultStride,
                                                                   extraParamsBuff,n);
}

/*
 * Class:     org_nd4j_linalg_cpu_nativecpu_ops_NativeOps
 * Method:    execReduce
 * Signature: (ILjava/nio/DoubleBuffer;Ljava/nio/IntBuffer;Ljava/nio/DoubleBuffer;Ljava/nio/DoubleBuffer;Ljava/nio/IntBuffer;)V
 */
JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_ops_NativeOps_execReduce__ILjava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_DoubleBuffer_2Ljava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2
        (JNIEnv *env, jobject obj, jint opNum, jobject x,
         jobject xShapeInfo,
         jobject extraParams,
         jobject result,
         jobject resultShapeInfo) {

}

/*
 * Class:     org_nd4j_linalg_cpu_nativecpu_ops_NativeOps
 * Method:    execReduce
 * Signature: (ILjava/nio/DoubleBuffer;Ljava/nio/IntBuffer;Ljava/nio/DoubleBuffer;Ljava/nio/DoubleBuffer;Ljava/nio/IntBuffer;Ljava/nio/IntBuffer;I)V
 */
JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_ops_NativeOps_execReduce__ILjava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_DoubleBuffer_2Ljava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_IntBuffer_2I
        (JNIEnv *env, jobject obj,
         jint opNum,
         jobject x,
         jobject xShapeInfo,
         jobject extraParams,
         jobject result, jobject resultShapeInfo, jobject dimension, jint dimensionLength) {
    double *xBuff = (double *) env->GetDirectBufferAddress(x);
    int *xShapeBuff = (int *) env->GetDirectBufferAddress(xShapeInfo);
    double *extraParamsBuff = extraParams != NULL ? (double *) env->GetDirectBufferAddress(extraParams) : NULL;
    double *resultBuff = (double *) env->GetDirectBufferAddress(result);
    int *resultShapeBuff = (int *) env->GetDirectBufferAddress(resultShapeInfo);
    int *dimensionBuff = (int *) env->GetDirectBufferAddress(dimension);

    DoubleNativeOpExecutioner::getInstance().execReduce(opNum,xBuff,xShapeBuff,extraParamsBuff,
                                                        resultBuff,resultShapeBuff,dimensionBuff,dimensionLength);

}

/*
 * Class:     org_nd4j_linalg_cpu_nativecpu_ops_NativeOps
 * Method:    execReduceScalar
 * Signature: (ILjava/nio/DoubleBuffer;Ljava/nio/IntBuffer;Ljava/nio/DoubleBuffer;)D
 */
JNIEXPORT jdouble JNICALL Java_org_nd4j_linalg_cpu_nativecpu_ops_NativeOps_execReduceScalar__ILjava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_DoubleBuffer_2
        (JNIEnv *env, jobject obj, jint opNum, jobject x, jobject xShapeInfo, jobject extraParams) {
    double *xBuff = (double *) env->GetDirectBufferAddress(x);
    int *xShapeBuff = (int *) env->GetDirectBufferAddress(xShapeInfo);
    double *extraParamsBuff = extraParams != NULL ? (double * ) env->GetDirectBufferAddress(extraParams) : NULL;
    return DoubleNativeOpExecutioner::getInstance().execReduceScalar(opNum,xBuff,xShapeBuff,extraParamsBuff);

}

/*
 * Class:     org_nd4j_linalg_cpu_nativecpu_ops_NativeOps
 * Method:    execReduce3
 * Signature: (ILjava/nio/DoubleBuffer;Ljava/nio/IntBuffer;Ljava/nio/DoubleBuffer;Ljava/nio/DoubleBuffer;Ljava/nio/IntBuffer;Ljava/nio/DoubleBuffer;Ljava/nio/IntBuffer;)V
 */
JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_ops_NativeOps_execReduce3__ILjava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_DoubleBuffer_2Ljava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2
        (JNIEnv *env,
         jobject obj,
         jint opNum,
         jobject x, jobject xShapeInfo,
         jobject extraParams,
         jobject y,
         jobject yShapeInfo, jobject result, jobject resultShapeInfo) {

}

/*
 * Class:     org_nd4j_linalg_cpu_nativecpu_ops_NativeOps
 * Method:    execReduce3Scalar
 * Signature: (ILjava/nio/DoubleBuffer;Ljava/nio/IntBuffer;Ljava/nio/DoubleBuffer;Ljava/nio/DoubleBuffer;Ljava/nio/IntBuffer;)D
 */
JNIEXPORT jdouble JNICALL Java_org_nd4j_linalg_cpu_nativecpu_ops_NativeOps_execReduce3Scalar__ILjava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_DoubleBuffer_2Ljava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2
        (JNIEnv *env, jobject obj,
         jint opNum, jobject x,
         jobject xShapeInfo,
         jobject extraParams,
         jobject y,
         jobject yShapeInfo) {
    double *xBuff = (double *) env->GetDirectBufferAddress(x);
    int *xShapeBuff = (int *) env->GetDirectBufferAddress(xShapeInfo);
    double *extraParamsBuff = extraParams != NULL ? (double *) env->GetDirectBufferAddress(extraParams) : NULL;
    double *yBuff = (double *) env->GetDirectBufferAddress(y);
    int *yShapeBuff = (int *) env->GetDirectBufferAddress(yShapeInfo);
    return DoubleNativeOpExecutioner::getInstance().execReduce3Scalar(opNum,xBuff,xShapeBuff,
                                                                      extraParamsBuff,yBuff,yShapeBuff);
}

/*
 * Class:     org_nd4j_linalg_cpu_nativecpu_ops_NativeOps
 * Method:    execReduce3
 * Signature: (ILjava/nio/DoubleBuffer;Ljava/nio/IntBuffer;Ljava/nio/DoubleBuffer;Ljava/nio/DoubleBuffer;Ljava/nio/IntBuffer;Ljava/nio/DoubleBuffer;Ljava/nio/IntBuffer;Ljava/nio/IntBuffer;I)V
 */
JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_ops_NativeOps_execReduce3__ILjava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_DoubleBuffer_2Ljava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_IntBuffer_2I
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
    double *xBuff = (double *) env->GetDirectBufferAddress(x);
    int *xShapeBuff = (int *) env->GetDirectBufferAddress(xShapeInfo);
    double *extraParamsBuff = extraParams != NULL ? (double *) env->GetDirectBufferAddress(extraParams) : NULL;
    double *yBuff = (double *) env->GetDirectBufferAddress(y);
    int *yShapeBuff = (int *) env->GetDirectBufferAddress(yShapeInfo);
    double *resultBuff = (double *) env->GetDirectBufferAddress(result);
    int *resultShapeBuff = (int *) env->GetDirectBufferAddress(resultShapeInfo);
    int *dimensionBuff = (int *) env->GetDirectBufferAddress(dimension);
    DoubleNativeOpExecutioner::getInstance().execReduce3(opNum,xBuff,xShapeBuff,extraParamsBuff,yBuff,yShapeBuff,resultBuff,resultShapeBuff,dimensionBuff,dimensionLength);

}

/*
 * Class:     org_nd4j_linalg_cpu_nativecpu_ops_NativeOps
 * Method:    execScalar
 * Signature: (ILjava/nio/DoubleBuffer;ILjava/nio/DoubleBuffer;IDLjava/nio/DoubleBuffer;I)D
 */
JNIEXPORT jdouble JNICALL Java_org_nd4j_linalg_cpu_nativecpu_ops_NativeOps_execScalar__ILjava_nio_DoubleBuffer_2ILjava_nio_DoubleBuffer_2IDLjava_nio_DoubleBuffer_2I
        (JNIEnv *env, jobject obj,
         jint opNum,
         jobject x,
         jint xStride,
         jobject result,
         jint resultStride,
         jdouble scalar,
         jobject extraParams,
         jint n) {
    double *xBuff = (double *) env->GetDirectBufferAddress(x);
    double *resultBuff = (double *) env->GetDirectBufferAddress(result);
    double *extraParamsBuff = extraParams != NULL ? (double *) env->GetDirectBufferAddress(extraParams) : NULL;
    DoubleNativeOpExecutioner::getInstance().execScalar(opNum,
                                                        xBuff,xStride,
                                                        resultBuff,resultStride,
                                                        scalar,extraParamsBuff,n);
    return 0.0;

}

/*
 * Class:     org_nd4j_linalg_cpu_nativecpu_ops_NativeOps
 * Method:    execSummaryStatsScalar
 * Signature: (ILjava/nio/DoubleBuffer;Ljava/nio/IntBuffer;Ljava/nio/DoubleBuffer;)D
 */
JNIEXPORT jdouble JNICALL Java_org_nd4j_linalg_cpu_nativecpu_ops_NativeOps_execSummaryStatsScalar__ILjava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_DoubleBuffer_2
        (JNIEnv *env,
         jobject obj,
         jint opNum,
         jobject x, jobject xShapeInfo, jobject extraParams) {
    double *xBuff = (double *) env->GetDirectBufferAddress(x);
    int *xShapeBuff = (int *) env->GetDirectBufferAddress(xShapeInfo);
    double *extraParamsBuff = extraParams != NULL ? (double *) env->GetDirectBufferAddress(extraParams) : NULL;
    return DoubleNativeOpExecutioner::getInstance().execSummaryStatsScalar(opNum,xBuff,xShapeBuff,extraParamsBuff);
}

/*
 * Class:     org_nd4j_linalg_cpu_nativecpu_ops_NativeOps
 * Method:    execSummaryStats
 * Signature: (ILjava/nio/DoubleBuffer;Ljava/nio/IntBuffer;Ljava/nio/DoubleBuffer;Ljava/nio/DoubleBuffer;Ljava/nio/IntBuffer;)V
 */
JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_ops_NativeOps_execSummaryStats__ILjava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_DoubleBuffer_2Ljava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2
        (JNIEnv *env, jobject obj, jint opNum,
         jobject x,
         jobject xShapeInfo, jobject extraParams,
         jobject result, jobject resultShapeInfo) {


}

/*
 * Class:     org_nd4j_linalg_cpu_nativecpu_ops_NativeOps
 * Method:    execSummaryStats
 * Signature: (ILjava/nio/DoubleBuffer;Ljava/nio/IntBuffer;Ljava/nio/DoubleBuffer;Ljava/nio/DoubleBuffer;Ljava/nio/IntBuffer;Ljava/nio/IntBuffer;I)V
 */
JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_ops_NativeOps_execSummaryStats__ILjava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_DoubleBuffer_2Ljava_nio_DoubleBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_IntBuffer_2I
        (JNIEnv *env, jobject obj,
         jint opNum,
         jobject x,
         jobject xShapeInfo,
         jobject extraParams,
         jobject result,
         jobject resultShapeInfo, jobject dimension, jint dimensionLength) {
    double *xBuff = (double *) env->GetDirectBufferAddress(x);
    int *xShapeBuff = (int *) env->GetDirectBufferAddress(xShapeInfo);
    double *extraParamsBuff = extraParams != NULL ? (double *) env->GetDirectBufferAddress(extraParams) : NULL;
    double *resultBuff = (double *) env->GetDirectBufferAddress(result);
    int *resultShapeBuff = (int *) env->GetDirectBufferAddress(resultShapeInfo);
    int *dimensionBuff = (int *) env->GetDirectBufferAddress(dimension);
    DoubleNativeOpExecutioner::getInstance().execSummaryStats(opNum,xBuff,xShapeBuff,extraParamsBuff,resultBuff,resultShapeBuff,dimensionBuff,dimensionLength);
}

/*
 * Class:     org_nd4j_linalg_cpu_nativecpu_ops_NativeOps
 * Method:    execTransform
 * Signature: (ILjava/nio/DoubleBuffer;ILjava/nio/DoubleBuffer;ILjava/nio/DoubleBuffer;I)V
 */
JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_ops_NativeOps_execTransform__ILjava_nio_DoubleBuffer_2ILjava_nio_DoubleBuffer_2ILjava_nio_DoubleBuffer_2I
        (JNIEnv *env,
         jobject obj,
         jint opNum,
         jobject x, jint xStride,
         jobject result,
         jint resultStride, jobject extraParams,
         jint n) {
    double *xBuff = (double *) env->GetDirectBufferAddress(x);
    double *extraParamsBuff = extraParams != NULL ? (double *) env->GetDirectBufferAddress(extraParams) : NULL;

    double *resultBuff = (double *) env->GetDirectBufferAddress(result);
    DoubleNativeOpExecutioner::getInstance().execTransform(opNum,xBuff,xStride,resultBuff,
                                                           resultStride,extraParamsBuff,n);


}

/*
 * Class:     org_nd4j_linalg_cpu_nativecpu_ops_NativeOps
 * Method:    execIndexReduceScalar
 * Signature: (ILjava/nio/FloatBuffer;Ljava/nio/IntBuffer;Ljava/nio/FloatBuffer;)F
 */
JNIEXPORT jfloat JNICALL Java_org_nd4j_linalg_cpu_nativecpu_ops_NativeOps_execIndexReduceScalar__ILjava_nio_FloatBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_FloatBuffer_2
        (JNIEnv *env, jobject obj, jint opNum, jobject x, jobject xShapeInfo, jobject extraParams) {
    int *xShapeBuff = (int *) env->GetDirectBufferAddress(xShapeInfo);
    float *extraParamsBuff = extraParams != NULL ? (float *) env->GetDirectBufferAddress(extraParams) : NULL;
    float *xBuff = (float *) env->GetDirectBufferAddress(x);
    return FloatNativeOpExecutioner::getInstance().execIndexReduceScalar(opNum,xBuff,xShapeBuff,extraParamsBuff);

}

/*
 * Class:     org_nd4j_linalg_cpu_nativecpu_ops_NativeOps
 * Method:    execIndexReduce
 * Signature: (ILjava/nio/FloatBuffer;Ljava/nio/IntBuffer;Ljava/nio/FloatBuffer;Ljava/nio/FloatBuffer;Ljava/nio/IntBuffer;Ljava/nio/IntBuffer;I)V
 */
JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_ops_NativeOps_execIndexReduce__ILjava_nio_FloatBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_FloatBuffer_2Ljava_nio_FloatBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_IntBuffer_2I
        (JNIEnv *env, jobject obj, jint opNum,
         jobject x, jobject xShapeInfo,
         jobject extraParams, jobject result, jobject resultShapeInfo, jobject dimension, jint dimensionLength) {
    int *xShapeBuff = (int *) env->GetDirectBufferAddress(xShapeInfo);
    float *xBuff = (float *) env->GetDirectBufferAddress(x);
    float *extraParamsBuff = extraParams != NULL ? (float *) env->GetDirectBufferAddress(extraParams) : NULL;
    float *resultBuff = (float *) env->GetDirectBufferAddress(result);
    int *resultShapeBuff = (int *) env->GetDirectBufferAddress(resultShapeInfo);
    int *dimensionBuff = (int *) env->GetDirectBufferAddress(dimension);
    FloatNativeOpExecutioner::getInstance().execIndexReduce(opNum,xBuff,xShapeBuff,extraParamsBuff,resultBuff,resultShapeBuff,dimensionBuff,dimensionLength);

}

/*
 * Class:     org_nd4j_linalg_cpu_nativecpu_ops_NativeOps
 * Method:    execBroadcast
 * Signature: (ILjava/nio/FloatBuffer;Ljava/nio/IntBuffer;Ljava/nio/FloatBuffer;Ljava/nio/IntBuffer;Ljava/nio/FloatBuffer;Ljava/nio/IntBuffer;Ljava/nio/IntBuffer;I)V
 */
JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_ops_NativeOps_execBroadcast__ILjava_nio_FloatBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_FloatBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_FloatBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_IntBuffer_2I
        (JNIEnv *env, jobject obj, jint opNum, jobject x,
         jobject xShapeInfo,
         jobject y,
         jobject yShapeInfo, jobject result,
         jobject resultShapeInfo,
         jobject dimension, jint dimensionLength) {
    float *xBuff = (float *) env->GetDirectBufferAddress(x);
    int *xShapeBuff = (int *) env->GetDirectBufferAddress(xShapeInfo);
    float *yBuff = (float *) env->GetDirectBufferAddress(y);
    int *yShapeBuff = (int *) env->GetDirectBufferAddress(yShapeInfo);
    float *resultBuff = (float *) env->GetDirectBufferAddress(result);
    int *resultShapeBuff = (int *) env->GetDirectBufferAddress(resultShapeInfo);
    int *dimensionBuff = (int *) env->GetDirectBufferAddress(dimension);

    FloatNativeOpExecutioner::getInstance().execBroadcast(
            opNum,xBuff,
            xShapeBuff,
            yBuff,
            yShapeBuff,
            resultBuff,
            resultShapeBuff,dimensionBuff,dimensionLength);
}

/*
 * Class:     org_nd4j_linalg_cpu_nativecpu_ops_NativeOps
 * Method:    execPairwiseTransform
 * Signature: (ILjava/nio/FloatBuffer;ILjava/nio/FloatBuffer;ILjava/nio/FloatBuffer;ILjava/nio/FloatBuffer;I)V
 */
JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_ops_NativeOps_execPairwiseTransform__ILjava_nio_FloatBuffer_2ILjava_nio_FloatBuffer_2ILjava_nio_FloatBuffer_2ILjava_nio_FloatBuffer_2I
        (JNIEnv *env,
         jobject obj,
         jint opNum,
         jobject dx, jint xStride,
         jobject y, jint yStride,
         jobject result,
         jint resultStride, jobject extraParams, jint n) {
    float *dxBuff = (float *) env->GetDirectBufferAddress(dx);
    float *yBuff = (float *) env->GetDirectBufferAddress(y);
    float *resultBuff = (float *) env->GetDirectBufferAddress(result);
    float *extraParamsBuff = extraParams != NULL ? (float *) env->GetDirectBufferAddress(extraParams) : NULL;
    FloatNativeOpExecutioner::getInstance().execPairwiseTransform(opNum,
                                                                  dxBuff,
                                                                  xStride,
                                                                  yBuff,
                                                                  yStride,
                                                                  resultBuff,
                                                                  resultStride,
                                                                  extraParamsBuff,n);

}

/*
 * Class:     org_nd4j_linalg_cpu_nativecpu_ops_NativeOps
 * Method:    execReduce
 * Signature: (ILjava/nio/FloatBuffer;Ljava/nio/IntBuffer;Ljava/nio/FloatBuffer;Ljava/nio/FloatBuffer;Ljava/nio/IntBuffer;Ljava/nio/IntBuffer;I)V
 */
JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_ops_NativeOps_execReduce__ILjava_nio_FloatBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_FloatBuffer_2Ljava_nio_FloatBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_IntBuffer_2I
        (JNIEnv *env, jobject obj,
         jint opNum, jobject x,
         jobject xShapeInfo, jobject extraParams,
         jobject result, jobject resultShapeInfo,
         jobject dimension, jint dimensionLength) {
    float *xBuff = (float *) env->GetDirectBufferAddress(x);
    int *xShapeBuff = (int *) env->GetDirectBufferAddress(xShapeInfo);
    float *extraParamsBuff = extraParams != NULL ? (float *) env->GetDirectBufferAddress(extraParams) : NULL;
    float *resultBuff = (float *) env->GetDirectBufferAddress(result);
    int *resultShapeBuff = (int *) env->GetDirectBufferAddress(resultShapeInfo);
    int *dimensionBuff = (int *) env->GetDirectBufferAddress(dimension);
/*
 * int opNum,
                    T *x,
                    int *xShapeInfo,
                    T *extraParams,
                    T *result,
                    int *resultShapeInfo,int *dimension,int dimensionLength)
 */
    FloatNativeOpExecutioner::getInstance().execReduce(opNum,
                                                       xBuff,
                                                       xShapeBuff,
                                                       extraParamsBuff,resultBuff,
                                                       resultShapeBuff,dimensionBuff,dimensionLength);


}

/*
 * Class:     org_nd4j_linalg_cpu_nativecpu_ops_NativeOps
 * Method:    execReduceScalar
 * Signature: (ILjava/nio/FloatBuffer;Ljava/nio/IntBuffer;Ljava/nio/FloatBuffer;)F
 */
JNIEXPORT jfloat JNICALL Java_org_nd4j_linalg_cpu_nativecpu_ops_NativeOps_execReduceScalar__ILjava_nio_FloatBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_FloatBuffer_2
        (JNIEnv *env, jobject obj, jint opNum, jobject x, jobject xShapeInfo, jobject extraParams) {
    float *xBuff = (float *) env->GetDirectBufferAddress(x);
    int *xShapeBuff = (int *) env->GetDirectBufferAddress(xShapeInfo);
    float *extraParamsBuff = extraParams != NULL ? (float *) env->GetDirectBufferAddress(extraParams) : NULL;
    return FloatNativeOpExecutioner::getInstance().execReduceScalar(opNum,xBuff,xShapeBuff,extraParamsBuff);


}

/*
 * Class:     org_nd4j_linalg_cpu_nativecpu_ops_NativeOps
 * Method:    execReduce3
 * Signature: (ILjava/nio/FloatBuffer;Ljava/nio/IntBuffer;Ljava/nio/FloatBuffer;Ljava/nio/FloatBuffer;Ljava/nio/IntBuffer;Ljava/nio/FloatBuffer;Ljava/nio/IntBuffer;)V
 */
JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_ops_NativeOps_execReduce3__ILjava_nio_FloatBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_FloatBuffer_2Ljava_nio_FloatBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_FloatBuffer_2Ljava_nio_IntBuffer_2
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
 * Class:     org_nd4j_linalg_cpu_nativecpu_ops_NativeOps
 * Method:    execReduce3Scalar
 * Signature: (ILjava/nio/FloatBuffer;Ljava/nio/IntBuffer;Ljava/nio/FloatBuffer;Ljava/nio/FloatBuffer;Ljava/nio/IntBuffer;)F
 */
JNIEXPORT jfloat JNICALL Java_org_nd4j_linalg_cpu_nativecpu_ops_NativeOps_execReduce3Scalar__ILjava_nio_FloatBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_FloatBuffer_2Ljava_nio_FloatBuffer_2Ljava_nio_IntBuffer_2
        (JNIEnv *env,
         jobject obj,
         jint opNum,
         jobject x,
         jobject xShapeInfo,
         jobject extraParams,
         jobject y,
         jobject yShapeInfo) {
    float *xBuff = (float *) env->GetDirectBufferAddress(x);
    int *xShapeBuff = (int *) env->GetDirectBufferAddress(xShapeInfo);
    float *extraParamsBuff = extraParams != NULL ? (float *) env->GetDirectBufferAddress(extraParams) : NULL;
    float *yBuff = (float *) env->GetDirectBufferAddress(y);
    int *yShapeBuff = (int *) env->GetDirectBufferAddress(yShapeInfo);
    return FloatNativeOpExecutioner::getInstance().execReduce3Scalar(opNum,xBuff,xShapeBuff,
                                                                     extraParamsBuff,yBuff,yShapeBuff);

}

/*
 * Class:     org_nd4j_linalg_cpu_nativecpu_ops_NativeOps
 * Method:    execReduce3
 * Signature: (ILjava/nio/FloatBuffer;Ljava/nio/IntBuffer;Ljava/nio/FloatBuffer;Ljava/nio/FloatBuffer;Ljava/nio/IntBuffer;Ljava/nio/FloatBuffer;Ljava/nio/IntBuffer;Ljava/nio/IntBuffer;I)V
 */
JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_ops_NativeOps_execReduce3__ILjava_nio_FloatBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_FloatBuffer_2Ljava_nio_FloatBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_FloatBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_IntBuffer_2I
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
    float *xBuff = (float *) env->GetDirectBufferAddress(x);
    int *xShapeBuff = (int *) env->GetDirectBufferAddress(xShapeInfo);
    float *extraParamsBuff = extraParams != NULL ? (float *) env->GetDirectBufferAddress(extraParams) : NULL;
    float *yBuff =(float *) env->GetDirectBufferAddress(y);
    int *yShapeBuff = (int *) env->GetDirectBufferAddress(yShapeInfo);
    float *resultBuff = (float *) env->GetDirectBufferAddress(result);
    int *resultShapeBuff = (int *) env->GetDirectBufferAddress(resultShapeInfo);
    int *dimensionBuff = (int *) env->GetDirectBufferAddress(dimension);
    FloatNativeOpExecutioner::getInstance().execReduce3(opNum,xBuff,xShapeBuff,extraParamsBuff,yBuff,yShapeBuff,resultBuff,resultShapeBuff,dimensionBuff,dimensionLength);

}

/*
 * Class:     org_nd4j_linalg_cpu_nativecpu_ops_NativeOps
 * Method:    execScalar
 * Signature: (ILjava/nio/FloatBuffer;ILjava/nio/FloatBuffer;IFLjava/nio/FloatBuffer;I)F
 */
JNIEXPORT jfloat JNICALL Java_org_nd4j_linalg_cpu_nativecpu_ops_NativeOps_execScalar__ILjava_nio_FloatBuffer_2ILjava_nio_FloatBuffer_2IFLjava_nio_FloatBuffer_2I
        (JNIEnv *env, jobject obj,
         jint opNum,
         jobject x, jint xStride,
         jobject result,
         jint resultStride,
         jfloat scalar,
         jobject extraParams, jint n) {
    float *xBuff = (float *) env->GetDirectBufferAddress(x);
    float *extraParamsBuff = extraParams != NULL ? (float *) env->GetDirectBufferAddress(extraParams) : NULL;
    float *resultBuff = (float *) env->GetDirectBufferAddress(result);
    FloatNativeOpExecutioner::getInstance().execScalar(opNum,xBuff,xStride,resultBuff,resultStride,scalar,extraParamsBuff,n);
    return 0;
}

/*
 * Class:     org_nd4j_linalg_cpu_nativecpu_ops_NativeOps
 * Method:    execSummaryStatsScalar
 * Signature: (ILjava/nio/FloatBuffer;Ljava/nio/IntBuffer;Ljava/nio/FloatBuffer;)F
 */
JNIEXPORT jfloat JNICALL Java_org_nd4j_linalg_cpu_nativecpu_ops_NativeOps_execSummaryStatsScalar__ILjava_nio_FloatBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_FloatBuffer_2
        (JNIEnv *env, jobject obj, jint opNum, jobject x, jobject xShapeInfo, jobject extraParams) {
    float *xBuff = (float *) env->GetDirectBufferAddress(x);
    int *xShapeBuff = (int *) env->GetDirectBufferAddress(xShapeInfo);
    float *extraParamsBuff = extraParams != NULL ? (float *) env->GetDirectBufferAddress(extraParams) : NULL;
    return FloatNativeOpExecutioner::getInstance().execSummaryStatsScalar(opNum,xBuff,xShapeBuff,extraParamsBuff);

}

/*
 * Class:     org_nd4j_linalg_cpu_nativecpu_ops_NativeOps
 * Method:    execSummaryStats
 * Signature: (ILjava/nio/FloatBuffer;Ljava/nio/IntBuffer;Ljava/nio/FloatBuffer;Ljava/nio/FloatBuffer;Ljava/nio/IntBuffer;)V
 */
JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_ops_NativeOps_execSummaryStats__ILjava_nio_FloatBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_FloatBuffer_2Ljava_nio_FloatBuffer_2Ljava_nio_IntBuffer_2
        (JNIEnv *env, jobject obj, jint opNum, jobject, jobject, jobject, jobject, jobject) {
    //no op
}

/*
 * Class:     org_nd4j_linalg_cpu_nativecpu_ops_NativeOps
 * Method:    execSummaryStats
 * Signature: (ILjava/nio/FloatBuffer;Ljava/nio/IntBuffer;Ljava/nio/FloatBuffer;Ljava/nio/FloatBuffer;Ljava/nio/IntBuffer;Ljava/nio/IntBuffer;I)V
 */
JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_ops_NativeOps_execSummaryStats__ILjava_nio_FloatBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_FloatBuffer_2Ljava_nio_FloatBuffer_2Ljava_nio_IntBuffer_2Ljava_nio_IntBuffer_2I
        (JNIEnv *env,
         jobject obj,
         jint opNum, jobject x,
         jobject xShapeInfo,
         jobject extraParams, jobject result, jobject resultShapeInfo, jobject dimension, jint dimensionLength) {
    float *xBuff = (float *) env->GetDirectBufferAddress(x);
    int *xShapeBuff = (int *) env->GetDirectBufferAddress(xShapeInfo);
    float *extraParamsBuff = extraParams != NULL ? (float *) env->GetDirectBufferAddress(extraParams) : NULL;
    float *resultBuff = (float *) env->GetDirectBufferAddress(result);
    int *resultShapeBuff = (int *) env->GetDirectBufferAddress(resultShapeInfo);
    int *dimensionBuff = (int *) env->GetDirectBufferAddress(dimension);
    FloatNativeOpExecutioner::getInstance().execSummaryStats(opNum,xBuff,xShapeBuff,extraParamsBuff,resultBuff,resultShapeBuff,dimensionBuff,dimensionLength);

}

/*
 * Class:     org_nd4j_linalg_cpu_nativecpu_ops_NativeOps
 * Method:    execTransform
 * Signature: (ILjava/nio/FloatBuffer;ILjava/nio/FloatBuffer;ILjava/nio/FloatBuffer;I)V
 */
JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_ops_NativeOps_execTransform__ILjava_nio_FloatBuffer_2ILjava_nio_FloatBuffer_2ILjava_nio_FloatBuffer_2I
        (JNIEnv *env, jobject obj, jint opNum, jobject x, jint xStride, jobject result, jint resultStride,
         jobject extraParams, jint n) {
    float *xBuff = (float *) env->GetDirectBufferAddress(x);
    float *extraParamsBuff = extraParams != NULL ? (float *) env->GetDirectBufferAddress(extraParams) : NULL;

    float *resultBuff = (float *) env->GetDirectBufferAddress(result);
    FloatNativeOpExecutioner::getInstance().execTransform(opNum,xBuff,xStride,resultBuff,
                                                          resultStride,extraParamsBuff,n);



}

#ifdef __cplusplus
}
#endif
