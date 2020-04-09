package org.nd4j.linalg.api.ops.impl.image;


public enum ImageResizeMethods {

    //Note: ordinal (order) here matters for C++ level from here
    // https://github.com/KonduitAI/deeplearning4j/blob/master/libnd4j/include/ops/declarable/helpers/image_resize.h
    kResizeBilinear,
    kResizeBicubic,
    kResizeNearest,
    kResizeGaussian,
    kResizeLanczos5,
    kResizeMitchelcubic,
    kResizeArea
}
