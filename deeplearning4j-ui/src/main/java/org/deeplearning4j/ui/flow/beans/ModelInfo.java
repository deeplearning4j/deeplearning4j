package org.deeplearning4j.ui.flow.beans;

import lombok.Data;
import lombok.NonNull;

import java.io.Serializable;
import java.util.List;

/**
 * This bean works as holder for unbounded list of layers. Each layer has it's own place in model's virtual coordinate space.
 * For now, coordinate space is limited to 2 dimensions
 *
 * @author raver119@gmail.com
 */
@Data
public class ModelInfo implements Serializable {
    private long time;

    // this should be table or map(pair(x,y)), but not a list
    private List<LayerInfo> layers;

    /**
     * This method maps given layer into model coordinate space
     * @param layer
     */
    public void addLayer(@NonNull LayerInfo layer) {
        // TODO: implement 2D mapping here
        this.layers.add(layer);
    }

    /**
     * This method returns LayerInfo for specified layer name
     * @param name
     * @return
     */
    public LayerInfo getLayerInfoByName(String name) {
        for (LayerInfo layerInfo: layers) {
            if (layerInfo.getName().equalsIgnoreCase(name)) return layerInfo;
        }
        return null;
    }

    /**
     * This method returns LayerInfo for specified grid coordinates
     * @param x
     * @param y
     * @return
     */
    public LayerInfo getLayerInfoByCoords(int x, int y) {
        return null;
    }
}
