package org.deeplearning4j.clustering.randomprojection;

import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;

@Data
public class RPTree {
    private RPNode root;
    private RPHyperPlanes rpHyperPlanes;
    private int dim;
    private int maxSize;

    public RPTree(int dim, int maxSize) {
        this.dim = dim;
        this.maxSize = maxSize;
        rpHyperPlanes = new RPHyperPlanes(dim);
        root = new RPNode(0);
    }


    public void buildTree(INDArray x) {
        for(int i = 0; i < x.rows(); i++) {
            root.getIndices().add(i);
        }

        RPUtils.buildTree(root,rpHyperPlanes,x,maxSize,0);
    }


    public void addNodeAtIndex(int idx,INDArray toAdd) {
        RPNode query = RPUtils.query(root,rpHyperPlanes,toAdd);
        query.getIndices().add(idx);
    }





}
