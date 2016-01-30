package org.deeplearning4j.ui.flow.beans;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
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
    private final static long serialVersionUID = 119L;
    private long time = System.currentTimeMillis();

    // PLEASE NOTE: Inverted coords here -> Y, X LayerInfo
    private Table<Integer, Integer, LayerInfo> layers = HashBasedTable.create();

    /**
     * This method maps given layer into model coordinate space
     * @param layer
     */
    public void addLayer(@NonNull LayerInfo layer) {
        this.layers.put(layer.getY(), layer.getX(), layer);
    }

    /**
     * This method returns LayerInfo for specified layer name
     * @param name
     * @return
     */
    public LayerInfo getLayerInfoByName(String name) {
        for (LayerInfo layerInfo: layers.values()) {
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
        return layers.get(y, x);
    }
}
