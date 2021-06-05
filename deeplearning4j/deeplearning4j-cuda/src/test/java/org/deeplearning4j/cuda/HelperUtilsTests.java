package org.deeplearning4j.cuda;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.cuda.convolution.CudnnConvolutionHelper;
import org.deeplearning4j.cuda.convolution.subsampling.CudnnSubsamplingHelper;
import org.deeplearning4j.cuda.dropout.CudnnDropoutHelper;
import org.deeplearning4j.cuda.normalization.CudnnLocalResponseNormalizationHelper;
import org.deeplearning4j.cuda.recurrent.CudnnLSTMHelper;
import org.deeplearning4j.nn.conf.dropout.DropoutHelper;
import org.deeplearning4j.nn.layers.HelperUtils;
import org.deeplearning4j.nn.layers.convolution.ConvolutionHelper;
import org.deeplearning4j.nn.layers.convolution.subsampling.SubsamplingHelper;
import org.deeplearning4j.nn.layers.normalization.LocalResponseNormalizationHelper;
import org.deeplearning4j.nn.layers.recurrent.LSTMHelper;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.NativeTag;

import static org.junit.jupiter.api.Assertions.assertNotNull;


@Slf4j
@NativeTag
public class HelperUtilsTests extends BaseDL4JTest  {

    @Test
    public void testHelperCreation() {
        System.setProperty(HelperUtils.DISABLE_HELPER_PROPERTY,"false");

        assertNotNull(HelperUtils.createHelper(CudnnLSTMHelper.class.getName(),"", LSTMHelper.class,"layer-name",getDataType()));
        assertNotNull(HelperUtils.createHelper(CudnnDropoutHelper.class.getName(),"", DropoutHelper.class,"layer-name",getDataType()));
        assertNotNull(HelperUtils.createHelper(CudnnConvolutionHelper.class.getName(),"", ConvolutionHelper.class,"layer-name",getDataType()));
        assertNotNull(HelperUtils.createHelper(CudnnLocalResponseNormalizationHelper.class.getName(),"", LocalResponseNormalizationHelper.class,"layer-name",getDataType()));
        assertNotNull(HelperUtils.createHelper(CudnnSubsamplingHelper.class.getName(),"", SubsamplingHelper.class,"layer-name",getDataType()));

    }

}
