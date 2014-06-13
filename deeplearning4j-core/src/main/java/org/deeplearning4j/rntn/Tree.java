package org.deeplearning4j.rntn;

import org.jblas.DoubleMatrix;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Tree for a recursive neural tensor network
 * based on Socher et al's work.
 */
public class Tree implements Serializable {

    private DoubleMatrix vector;
    private DoubleMatrix prediction;
    private List<Tree> children;
    private double error;
    private Tree parent;
    private String value;
    private String label;
    private int goldLabel;


    /**
     * Returns all of the labels for this node and all of its children (recursively)
     * @return all of the labels of this node and its children recursively
     */
    public List<String> yield() {
        return yield(new ArrayList<String>());

    }

    /**
     * Returns the list of labels for this node and
     * all of its children recursively
     * @param labels the labels to add to
     * @return the list of labels for this node and
     * all of its children recursively
     */
    private List<String> yield(List<String> labels) {
        labels.add(label);
        for(Tree t : children) {
            labels.addAll(t.yield());
        }
        return labels;
    }


    public void setGoldLabel(int goldLabel) {
        this.goldLabel = goldLabel;
    }

    public int goldLabel() {
        return goldLabel;
    }

    public void setLabel(String label) {
        this.label = label;
    }

    public String label() {
        return label;
    }


    public String value() {
        return value;
    }


    public void setValue(String value) {
        this.value = value;

    }

    public Tree(Tree parent) {
        this.parent = parent;
        children = new ArrayList<>();
    }

    public Tree() {
        children = new ArrayList<>();
    }

    /**
     * Returns whether the node has any children or not
     * @return whether the node has any children or not
     */
    public boolean isLeaf() {
        return children.isEmpty();
    }

    public List<Tree> children() {
        return children;
    }

    /**
     * Node has one child that is a leaf
     * @return whether the node has one child and the child is a leaf
     */
    public boolean isPreTerminal() {
        return children.size() == 1 && children.get(0).isLeaf();
    }


    public Tree firstChild() {
        return children.isEmpty() ? null : children.get(0);
    }

    public Tree lastChild() {
        return children.isEmpty() ? null : children.get(children.size() - 1);
    }
    /**
     * Finds the depth of the tree.  The depth is defined as the length
     * of the longest path from this node to a leaf node.  Leaf nodes
     * have depth zero.  POS tags have depth 1. Phrasal nodes have
     * depth &gt;= 2.
     *
     * @return the depth
     */
    public int depth() {
        if (isLeaf()) {
            return 0;
        }
        int maxDepth = 0;
        List<Tree> kids = children();
        for (Tree kid : kids) {
            int curDepth = kid.depth();
            if (curDepth > maxDepth) {
                maxDepth = curDepth;
            }
        }
        return maxDepth + 1;
    }

    /**
     * Returns the distance between this node
     * and the specified subnode
     * @param node the node to get the distance from
     * @return the distance between the 2 nodes
     */
    public int depth(Tree node) {
        Tree p = node.parent(this);
        if (this == node) { return 0; }
        if (p == null) { return -1; }
        int depth = 1;
        while (this != p) {
            p = p.parent(this);
            depth++;
        }
        return depth;
    }

    /**
     * Returns the parent of the passed in tree via traversal
     * @param root the root node
     * @return the tree to traverse
     */
    public Tree parent(Tree root) {
        List<Tree> kids = root.children();
        return traverse(root, kids, this);
    }


    //traverses the tree by recursion
    private static Tree traverse(Tree parent, List<Tree> kids, Tree node) {
        for (Tree kid : kids) {
            if (kid == node) {
                return parent;
            }

            Tree ret = node.parent(kid);
            if (ret != null) {
                return ret;
            }
        }
        return null;
    }

    /**
     * Returns the
     * @param height
     * @param root
     * @return
     */
    public Tree ancestor(int height, Tree root) {
        if (height < 0) {
            throw new IllegalArgumentException("ancestor: height cannot be negative");
        }
        if (height == 0) {
            return this;
        }
        Tree par = parent(root);
        if (par == null) {
            return null;
        }
        return par.ancestor(height - 1, root);
    }


    /**
     * Returns the total prediction error for this
     * tree and its children
     * @return the total error for this tree and its children
     */
    public double errorSum() {
        if (isLeaf()) {
            return 0.0;
        } else if (isPreTerminal()) {
            return error();
        } else {
            double error = 0.0;
            for (Tree child : children()) {
                error += child.errorSum();
            }
            return error() + error;
        }
    }


    /**
     * Gets the leaves of the tree.  All leaves nodes are returned as a list
     * ordered by the natural left to right order of the tree.  Null values,
     * if any, are inserted into the list like any other value.
     *
     * @return a <code>List</code> of the leaves.
     */
    public <T extends Tree> List<T> getLeaves() {
        return getLeaves(new ArrayList<T>());
    }

    /**
     * Gets the leaves of the tree.
     *
     * @param list The list in which the leaves of the tree will be
     *             placed. Normally, this will be empty when the routine is called,
     *             but if not, the new yield is added to the end of the list.
     * @return a <code>List</code> of the leaves.
     */
    @SuppressWarnings("unchecked")
    public <T extends Tree> List<T> getLeaves(List<T> list) {
        if (isLeaf()) {
            list.add((T)this);
        } else {
            for (Tree kid : children()) {
                kid.getLeaves(list);
            }
        }
        return list;
    }

    @Override
    public Tree clone() {
        Tree ret = new Tree();
        ret.setChildren(children());
        ret.setError(error);
        ret.setValue(value);
        ret.setVector(vector);
        return ret;
    }


    /**
     * Returns the prediction error for this node
     * @return the prediction error for this node
     */
    public double error() {

        return error;

    }


    public void setError(double error) {
        this.error = error;
    }





    public void setParent(Tree parent) {
        this.parent = parent;
    }

    public Tree parent() {
        return parent;
    }

    public DoubleMatrix vector() {
        return vector;
    }

    public void setVector(DoubleMatrix vector) {
        this.vector = vector;
    }

    public DoubleMatrix prediction() {
        return prediction;
    }

    public void setPrediction(DoubleMatrix prediction) {
        this.prediction = prediction;
    }


    public void setChildren(List<Tree> children) {
        this.children = children;
    }


}
