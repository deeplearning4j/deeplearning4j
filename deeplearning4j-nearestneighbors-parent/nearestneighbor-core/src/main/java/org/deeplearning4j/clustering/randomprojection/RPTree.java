package org.deeplearning4j.clustering.randomprojection;

import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@Data
public class RPTree {
    private RPNode root;
    private RPHyperPlanes rpHyperPlanes;
    private int dim;
    //also knows as leave size
    private int maxSize;
    private INDArray X;
    private String similarityFunction = "cosinesimilarity";


    /**
     *
     * @param dim the dimension of the vectors
     * @param maxSize the max size of the leaves
     *
     */
    public RPTree(int dim, int maxSize,String similarityFunction) {
        this.dim = dim;
        this.maxSize = maxSize;
        rpHyperPlanes = new RPHyperPlanes(dim);
        root = new RPNode(this,0);
        this.similarityFunction = similarityFunction;
    }

    /**
     *
     * @param dim the dimension of the vectors
     * @param maxSize the max size of the leaves
     *
     */
    public RPTree(int dim, int maxSize) {
       this(dim,maxSize,"cosinesimilarity");
    }


    public void buildTree(INDArray x) {
        this.X = x;
        for(int i = 0; i < x.rows(); i++) {
            root.getIndices().add(i);
        }

        RPUtils.buildTree(this,root,rpHyperPlanes,
                x,maxSize,0,similarityFunction);
    }



    public void addNodeAtIndex(int idx,INDArray toAdd) {
        RPNode query = RPUtils.query(root,rpHyperPlanes,toAdd,similarityFunction);
        query.getIndices().add(idx);
    }


    public List<RPNode> getLeaves() {
        List<RPNode> nodes = new ArrayList<>();
        RPUtils.scanForLeaves(nodes,getRoot());
        return nodes;
    }


    public INDArray query(INDArray query,int numResults) {
        return RPUtils.queryAll(query,X,Arrays.asList(this),numResults,similarityFunction);
    }

    public List<Integer> getCandidates(INDArray target) {
        return RPUtils.getCandidates(target,Arrays.asList(this),similarityFunction);
    }


}
