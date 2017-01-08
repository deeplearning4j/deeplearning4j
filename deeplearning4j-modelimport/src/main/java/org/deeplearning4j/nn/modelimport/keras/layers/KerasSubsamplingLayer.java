package org.deeplearning4j.nn.modelimport.keras.layers;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;

import java.util.Map;

/**
 * Created by davekale on 1/5/17.
 */
@Slf4j
public class KerasSubsamplingLayer extends KerasLayer {

    public KerasSubsamplingLayer(Map<String,Object> layerConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this(layerConfig, true);
    }

    public KerasSubsamplingLayer(Map<String,Object> layerConfig, boolean enforceTrainingConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);
        this.dl4jLayer = new SubsamplingLayer.Builder(mapPoolingType(this.className))
            .name(this.layerName)
            .dropOut(getDropoutFromConfig(layerConfig))
            .convolutionMode(getConvolutionModeFromConfig(layerConfig))
            .kernelSize(getKernelSizeFromConfig(layerConfig))
            .stride(getStrideFromConfig(layerConfig))
            .padding(getPaddingFromConfig(layerConfig))
            .build();
    }

    public SubsamplingLayer getSubsamplingLayer() {
        return (SubsamplingLayer)this.dl4jLayer;
    }
}
