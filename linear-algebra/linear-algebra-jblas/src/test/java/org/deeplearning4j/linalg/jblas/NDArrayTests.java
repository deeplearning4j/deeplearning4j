package org.deeplearning4j.linalg.jblas;


import org.deeplearning4j.linalg.api.complex.IComplexDouble;
import org.deeplearning4j.linalg.api.complex.IComplexNDArray;
import org.deeplearning4j.linalg.api.ndarray.DimensionSlice;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.api.ndarray.SliceOp;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.linalg.jblas.complex.ComplexNDArray;
import org.deeplearning4j.linalg.ops.reduceops.Ops;
import org.deeplearning4j.linalg.util.ArrayUtil;
import org.deeplearning4j.linalg.util.Shape;
import org.jblas.DoubleMatrix;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.*;

/**
 * NDArrayTests
 * @author Adam Gibson
 */
public class NDArrayTests extends org.deeplearning4j.linalg.api.test.NDArrayTests {
    private static Logger log = LoggerFactory.getLogger(NDArrayTests.class);

}
