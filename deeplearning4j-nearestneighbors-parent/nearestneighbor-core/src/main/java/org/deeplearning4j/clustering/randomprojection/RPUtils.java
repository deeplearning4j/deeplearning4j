package org.deeplearning4j.clustering.randomprojection;

import com.google.common.primitives.Doubles;
import lombok.val;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * A port of:
 * https://github.com/lyst/rpforest
 * to nd4j
 */
public class RPUtils {



    public static INDArray queryAll(INDArray toQuery,INDArray X,List<RPTree> trees,int n) {
        int dim = X.size(1);
        List<Integer> candidates = getCandidates(toQuery, trees, dim);
        val sortedCandidates = sortCandidates(toQuery,X,dim,candidates);
        int numReturns = Math.min(n,sortedCandidates.size());

        INDArray result = Nd4j.create(numReturns);
        for(int i = 0; i < numReturns; i++) {
            result.putScalar(i,sortedCandidates.get(i).getSecond());
        }


        return result;
    }

    public static List<Pair<Double,Integer>> sortCandidates(INDArray x,INDArray X,int dim,List<Integer> candidates) {
        int prevIdx = -1;
        List<Pair<Double,Integer>> ret = new ArrayList<>();
        for(int i = 0; i < candidates.size(); i++) {
            if(candidates.get(i) != prevIdx) {
                ret.add(Pair.of(Transforms.cosineSim(x,X.slice(candidates.get(i))),candidates.get(i)));
            }

            prevIdx = i;
        }


        Collections.sort(ret, new Comparator<Pair<Double, Integer>>() {
            @Override
            public int compare(Pair<Double, Integer> doubleIntegerPair, Pair<Double, Integer> t1) {
                return Doubles.compare(doubleIntegerPair.getFirst(),t1.getFirst());
            }
        });

        return ret;
    }




    public static INDArray getAllCandidates(INDArray x,List<RPTree> trees,int dim,int num) {
        List<Integer> candidates = getCandidates(x,trees,dim);
        Collections.sort(candidates);

        int prevIdx = -1;
        int idxCount = 0;
        List<Pair<Integer,Integer>> scores = new ArrayList<>();
        for(int i = 0; i < candidates.size(); i++) {
            if(candidates.get(i) == prevIdx) {
                idxCount++;
            }
            else if(prevIdx != -1) {
                scores.add(Pair.of(idxCount,prevIdx));
                idxCount = 1;
            }

            prevIdx = i;
        }


        scores.add(Pair.of(idxCount,prevIdx));

        INDArray arr = Nd4j.create(scores.size());
        for(int i = 0; i < scores.size(); i++) {
            arr.putScalar(i,scores.get(i).getSecond());
        }

        return arr;
    }

    public static List<Integer> getCandidates(INDArray x,List<RPTree> roots,int dim) {
        List<Integer> ret = new ArrayList<>();
        for(RPTree tree : roots) {
            RPNode root = tree.getRoot();
            RPNode query = query(root,tree.getRpHyperPlanes(),x);
            ret.addAll(query.getIndices());
        }

        return ret;
    }

    public static  RPNode query(RPNode from,RPHyperPlanes planes,INDArray x) {
        if(from.getLeft() == null && from.getRight() == null) {
            return from;
        }

        INDArray hyperPlane = planes.getHyperPlaneAt(from.getDepth());
        double dist = Transforms.cosineSim(hyperPlane,x);
        if(dist <= from.getMedian()) {
            query(from.getLeft(),planes,x);
        }

        else {
            query(from.getRight(),planes,x);
        }

        return null;
    }


    public static void buildTree(RPNode from,
                                 RPHyperPlanes planes,
                                 INDArray X,
                                 int maxSize,
                                 int depth) {
        if(from.getIndices().size() <= maxSize) {
            //slimNode
            slimNode(from);
            return;
        }


        List<Double> distances = new ArrayList<>();
        RPNode left = new RPNode(depth + 1);
        RPNode right = new RPNode(depth + 1);

        if(depth >= planes.getWholeHyperPlane().rows()) {
            planes.addRandomHyperPlane();
        }


        INDArray hyperPlane = planes.getHyperPlaneAt(depth);



        for(int i = 0; i < from.getIndices().size(); i++) {
            double cosineSim = Transforms.cosineSim(hyperPlane,X.slice(from.getIndices().get(i)));
            distances.add(cosineSim);
        }

        Collections.sort(distances);
        from.setMedian(distances.get(distances.size() / 2));


        for(int i = 0; i < from.getIndices().size(); i++) {
            double cosineSim = Transforms.cosineSim(hyperPlane,X.slice(from.getIndices().get(i)));
            if(cosineSim <= from.getMedian()) {
                left.getIndices().add(from.getIndices().get(i));
            }
            else {
                right.getIndices().add(from.getIndices().get(i));
            }
        }

        //failed split
        if(left.getIndices().isEmpty() || right.getIndices().isEmpty()) {
            slimNode(from);
            return;
        }


        from.setLeft(left);
        from.setRight(right);
        slimNode(from);


        buildTree(left,planes,X,maxSize,depth + 1);
        buildTree(right,planes,X,maxSize,depth + 1);

    }





    public static void slimNode(RPNode node) {
        if(!node.getIndices().isEmpty()) {
            node.getIndices().clear();
        }

    }


}
