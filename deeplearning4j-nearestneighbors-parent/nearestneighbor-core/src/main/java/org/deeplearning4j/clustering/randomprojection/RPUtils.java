package org.deeplearning4j.clustering.randomprojection;

import com.google.common.primitives.Doubles;
import lombok.val;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.imports.converters.DifferentialFunctionClassHolder;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.impl.accum.distances.*;
import org.nd4j.linalg.exception.ND4JIllegalArgumentException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;

import java.util.*;

/**
 * A port of:
 * https://github.com/lyst/rpforest
 * to nd4j
 *
 * @author
 */
public class RPUtils {


    private static ThreadLocal<Map<String,DifferentialFunction>> functionInstances = new ThreadLocal<>();

    public static <T extends DifferentialFunction> DifferentialFunction getOp(String name,
                                                                              INDArray x,
                                                                              INDArray y,
                                                                              INDArray result) {
        Map<String,DifferentialFunction> ops = functionInstances.get();
        if(ops == null) {
            ops = new HashMap<>();
            functionInstances.set(ops);
        }

        switch(name) {
            case "cosinedistance":
                if(!ops.containsKey(name)) {
                    CosineDistance cosineDistance = new CosineDistance(x,y,result,x.length());
                    ops.put(name,cosineDistance);
                    return cosineDistance;
                }
                else {
                    CosineDistance cosineDistance = (CosineDistance) ops.get(name);
                    return cosineDistance;
                }
            case "cosinesimilarity":
                if(!ops.containsKey(name)) {
                    CosineSimilarity cosineSimilarity = new CosineSimilarity(x,y,result,x.length());
                    ops.put(name,cosineSimilarity);
                    return cosineSimilarity;
                }
                else {
                    CosineSimilarity cosineSimilarity = (CosineSimilarity) ops.get(name);
                    cosineSimilarity.setX(x);
                    cosineSimilarity.setY(y);
                    cosineSimilarity.setZ(result);
                    cosineSimilarity.setN(x.length());
                    return cosineSimilarity;

                }
            case "manhattan":
                if(!ops.containsKey(name)) {
                    ManhattanDistance manhattanDistance = new ManhattanDistance(x,y,result,x.length());
                    ops.put(name,manhattanDistance);
                    return manhattanDistance;
                }
                else {
                    ManhattanDistance manhattanDistance = (ManhattanDistance) ops.get(name);
                    manhattanDistance.setX(x);
                    manhattanDistance.setY(y);
                    manhattanDistance.setZ(result);
                    manhattanDistance.setN(x.length());
                    return  manhattanDistance;
                }
            case "jaccard":
                if(!ops.containsKey(name)) {
                    JaccardDistance jaccardDistance = new JaccardDistance(x,y,result,x.length());
                    ops.put(name,jaccardDistance);
                    return jaccardDistance;
                }
                else {
                    JaccardDistance jaccardDistance = (JaccardDistance) ops.get(name);
                    jaccardDistance.setX(x);
                    jaccardDistance.setY(y);
                    jaccardDistance.setZ(result);
                    jaccardDistance.setN(x.length());
                    return jaccardDistance;
                }
            case "hamming":
                if(!ops.containsKey(name)) {
                    HammingDistance hammingDistance = new HammingDistance(x,y,result,x.length());
                    ops.put(name,hammingDistance);
                    return hammingDistance;
                }
                else {
                    HammingDistance hammingDistance = (HammingDistance) ops.get(name);
                    hammingDistance.setX(x);
                    hammingDistance.setY(y);
                    hammingDistance.setZ(result);
                    hammingDistance.setN(x.length());
                    return hammingDistance;
                }
                //euclidean
            default:
                if(!ops.containsKey(name)) {
                    EuclideanDistance euclideanDistance = new EuclideanDistance(x,y,result,x.length());
                    ops.put(name,euclideanDistance);
                    return euclideanDistance;
                }
                else {
                    EuclideanDistance euclideanDistance = (EuclideanDistance) ops.get(name);
                    euclideanDistance.setX(x);
                    euclideanDistance.setY(y);
                    euclideanDistance.setZ(result);
                    euclideanDistance.setN(x.length());
                    return euclideanDistance;
                }
        }
    }


    /**
     * Query all trees using the given input and data
     * @param toQuery the query vector
     * @param X the input data to query
     * @param trees the trees to query
     * @param n the number of results to search for
     * @param similarityFunction the similarity function to use
     * @return the indices (in order) in the ndarray
     */
    public static List<Pair<Double,Integer>> queryAllWithDistances(INDArray toQuery,INDArray X,List<RPTree> trees,int n,String similarityFunction) {
        if(trees.isEmpty()) {
            throw new ND4JIllegalArgumentException("Trees is empty!");
        }

        List<Integer> candidates = getCandidates(toQuery, trees,similarityFunction);
        val sortedCandidates = sortCandidates(toQuery,X,candidates,similarityFunction);
        int numReturns = Math.min(n,sortedCandidates.size());
        List<Pair<Double,Integer>> ret = new ArrayList<>(numReturns);
        for(int i = 0; i < numReturns; i++) {
            ret.add(sortedCandidates.get(i));
        }

        return ret;
    }

    /**
     * Query all trees using the given input and data
     * @param toQuery the query vector
     * @param X the input data to query
     * @param trees the trees to query
     * @param n the number of results to search for
     * @param similarityFunction the similarity function to use
     * @return the indices (in order) in the ndarray
     */
    public static INDArray queryAll(INDArray toQuery,INDArray X,List<RPTree> trees,int n,String similarityFunction) {
        if(trees.isEmpty()) {
            throw new ND4JIllegalArgumentException("Trees is empty!");
        }

        List<Integer> candidates = getCandidates(toQuery, trees,similarityFunction);
        val sortedCandidates = sortCandidates(toQuery,X,candidates,similarityFunction);
        int numReturns = Math.min(n,sortedCandidates.size());

        INDArray result = Nd4j.create(numReturns);
        for(int i = 0; i < numReturns; i++) {
            result.putScalar(i,sortedCandidates.get(i).getSecond());
        }


        return result;
    }

    /**
     * Get the sorted distances given the
     * query vector, input data, given the list of possible search candidates
     * @param x the query vector
     * @param X the input data to use
     * @param candidates the possible search candidates
     * @param similarityFunction the similarity function to use
     * @return the sorted distances
     */
    public static List<Pair<Double,Integer>> sortCandidates(INDArray x,INDArray X,
                                                            List<Integer> candidates,
                                                            String similarityFunction) {
        int prevIdx = -1;
        List<Pair<Double,Integer>> ret = new ArrayList<>();
        for(int i = 0; i < candidates.size(); i++) {
            if(candidates.get(i) != prevIdx) {
                ret.add(Pair.of(computeDistance(similarityFunction,X.slice(candidates.get(i)),x),candidates.get(i)));
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



    /**
     * Get the search candidates as indices given the input
     * and similarity function
     * @param x the input data to search with
     * @param trees the trees to search
     * @param similarityFunction the function to use for similarity
     * @return the list of indices as the search results
     */
    public static INDArray getAllCandidates(INDArray x,List<RPTree> trees,String similarityFunction) {
        List<Integer> candidates = getCandidates(x,trees,similarityFunction);
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


    /**
     * Get the search candidates as indices given the input
     * and similarity function
     * @param x the input data to search with
     * @param roots the trees to search
     * @param similarityFunction the function to use for similarity
     * @return the list of indices as the search results
     */
    public static List<Integer> getCandidates(INDArray x,List<RPTree> roots,String similarityFunction) {
        Set<Integer> ret = new LinkedHashSet<>();
        for(RPTree tree : roots) {
            RPNode root = tree.getRoot();
            RPNode query = query(root,tree.getRpHyperPlanes(),x,similarityFunction);
            ret.addAll(query.getIndices());
        }

        return new ArrayList<>(ret);
    }


    /**
     * Query the tree starting from the given node
     * using the given hyper plane and similarity function
     * @param from the node to start from
     * @param planes the hyper plane to query
     * @param x the input data
     * @param similarityFunction the similarity function to use
     * @return the leaf node representing the given query from a
     * search in the tree
     */
    public static  RPNode query(RPNode from,RPHyperPlanes planes,INDArray x,String similarityFunction) {
        if(from.getLeft() == null && from.getRight() == null) {
            return from;
        }

        INDArray hyperPlane = planes.getHyperPlaneAt(from.getDepth());
        double dist = computeDistance(similarityFunction,x,hyperPlane);
        if(dist <= from.getMedian()) {
            return query(from.getLeft(),planes,x,similarityFunction);
        }

        else {
            return query(from.getRight(),planes,x,similarityFunction);
        }

    }


    /**
     * Compute the distance between 2 vectors
     * given a function name. Valid function names:
     * euclidean: euclidean distance
     * cosinedistance: cosine distance
     * cosine similarity: cosine similarity
     * manhattan: manhattan distance
     * jaccard: jaccard distance
     * hamming: hamming distance
     * @param function the function to use (default euclidean distance)
     * @param x the first vector
     * @param y the second vector
     * @return the distance between the 2 vectors given the inputs
     */
    public static INDArray computeDistanceMulti(String function,INDArray x,INDArray y,INDArray result) {
        Accumulation op = (Accumulation) getOp(function, x, y, result);
        Nd4j.getExecutioner().exec(op,-1);
        return op.z();
    }

    /**

    /**
     * Compute the distance between 2 vectors
     * given a function name. Valid function names:
     * euclidean: euclidean distance
     * cosinedistance: cosine distance
     * cosine similarity: cosine similarity
     * manhattan: manhattan distance
     * jaccard: jaccard distance
     * hamming: hamming distance
     * @param function the function to use (default euclidean distance)
     * @param x the first vector
     * @param y the second vector
     * @return the distance between the 2 vectors given the inputs
     */
    public static double computeDistance(String function,INDArray x,INDArray y,INDArray result) {
        Accumulation op = (Accumulation) getOp(function, x, y, result);
        Nd4j.getExecutioner().exec(op);
        return op.z().getDouble(0);
    }

    /**
     * Compute the distance between 2 vectors
     * given a function name. Valid function names:
     * euclidean: euclidean distance
     * cosinedistance: cosine distance
     * cosine similarity: cosine similarity
     * manhattan: manhattan distance
     * jaccard: jaccard distance
     * hamming: hamming distance
     * @param function the function to use (default euclidean distance)
     * @param x the first vector
     * @param y the second vector
     * @return the distance between the 2 vectors given the inputs
     */
    public static double computeDistance(String function,INDArray x,INDArray y) {
        return computeDistance(function,x,y,Nd4j.scalar(0.0));
    }

    /**
     * Initialize the tree given the input parameters
     * @param tree the tree to initialize
     * @param from the starting node
     * @param planes the hyper planes to use (vector space for similarity)
     * @param X the input data
     * @param maxSize the max number of indices on a given leaf node
     * @param depth the current depth of the tree
     * @param similarityFunction the similarity function to use
     */
    public static void buildTree(RPTree tree,
                                 RPNode from,
                                 RPHyperPlanes planes,
                                 INDArray X,
                                 int maxSize,
                                 int depth,
                                 String similarityFunction) {
        if(from.getIndices().size() <= maxSize) {
            //slimNode
            slimNode(from);
            return;
        }


        List<Double> distances = new ArrayList<>();
        RPNode left = new RPNode(tree,depth + 1);
        RPNode right = new RPNode(tree,depth + 1);

        if(planes.getWholeHyperPlane() == null || depth >= planes.getWholeHyperPlane().rows()) {
            planes.addRandomHyperPlane();
        }


        INDArray hyperPlane = planes.getHyperPlaneAt(depth);



        for(int i = 0; i < from.getIndices().size(); i++) {
            double cosineSim = computeDistance(similarityFunction,hyperPlane,X.slice(from.getIndices().get(i)));
            distances.add(cosineSim);
        }

        Collections.sort(distances);
        from.setMedian(distances.get(distances.size() / 2));


        for(int i = 0; i < from.getIndices().size(); i++) {
            double cosineSim = computeDistance(similarityFunction,hyperPlane,X.slice(from.getIndices().get(i)));
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


        buildTree(tree,left,planes,X,maxSize,depth + 1,similarityFunction);
        buildTree(tree,right,planes,X,maxSize,depth + 1,similarityFunction);

    }


    /**
     * Scan for leaves accumulating
     * the nodes in the passed in list
     * @param nodes the nodes so far
     * @param scan the tree to scan
     */
    public static void scanForLeaves(List<RPNode> nodes,RPTree scan) {
        scanForLeaves(nodes,scan.getRoot());
    }

    /**
     * Scan for leaves accumulating
     * the nodes in the passed in list
     * @param nodes the nodes so far
     */
    public static void scanForLeaves(List<RPNode> nodes,RPNode current) {
        if(current.getLeft() == null && current.getRight() == null)
            nodes.add(current);
        if(current.getLeft() != null)
            scanForLeaves(nodes,current.getLeft());
        if(current.getRight() != null)
            scanForLeaves(nodes,current.getRight());
    }


    /**
     * Prune indices from the given node
     * when it's a leaf
     * @param node the node to prune
     */
    public static void slimNode(RPNode node) {
        if(node.getRight() != null && node.getLeft() != null) {
            node.getIndices().clear();
        }

    }


}
