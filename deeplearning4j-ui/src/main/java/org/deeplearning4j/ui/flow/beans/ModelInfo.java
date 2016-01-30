package org.deeplearning4j.ui.flow.beans;

import lombok.Data;
import lombok.NonNull;

import java.io.Serializable;
import java.util.List;

/**
 * This bean works as holder for unbounded list of layers. Each layer has it's own place in model's virtual coordinate space
 *
 * @author raver119@gmail.com
 */
@Data
public class ModelInfo implements Serializable {
    private long time;
    private List<LayerInfo> layers;

    public void addLayer(@NonNull LayerInfo layer) {
        this.layers.add(layer);
    }

    private LayerInfo getLayerInfoByName(String name) {
        for (LayerInfo layerInfo: layers) {
            if (layerInfo.getName().equalsIgnoreCase(name)) return layerInfo;
        }

        return null;
    }
}
