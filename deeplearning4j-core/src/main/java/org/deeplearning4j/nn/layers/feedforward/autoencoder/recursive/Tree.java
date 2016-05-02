/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.nn.layers.feedforward.autoencoder.recursive;


import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Tree for a recursive neural tensor network
 * based on Socher et al's work.
 */
public class Tree implements Serializable {

    private INDArray vector;
    private INDArray prediction;
    private List<Tree> children;
    private double error;
    private Tree parent;
    private String headWord;
    private String value;
    private String label;
    private String type;
    private int goldLabel;
    private List<String> tokens;
    private List<String> tags;
    private String parse;
    private int begin,end;

    /**
     * Clone constructor (all but the children)
     * @param tree the tree to clone
     */
    public Tree(Tree tree) {
        setError(tree.error);
        setValue(tree.value);
        setVector(tree.vector);
        setParse(tree.parse);
        setLabel(tree.label);
        setGoldLabel(tree.goldLabel);
        setPrediction(tree.prediction);
        setTags(tree.tags);
        setBegin(tree.begin);
        setEnd(tree.end);
        setType(tree.type);
    }

    public Tree(Tree parent,List<String> tokens) {
        this.parent = parent;
        this.tokens = tokens;
        children = new ArrayList<>();
    }

    public Tree(List<String> tokens) {
        children = new ArrayList<>();
        this.tokens = tokens;
    }

    /**
     * The type of node; mainly extra meta data
     * @return
     */
    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

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
        for(Tree t : children()) {
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


    /**
     * Returns whether the node has any children or not
     * @return whether the node has any children or not
     */
    public boolean isLeaf() {
        return children == null || children.isEmpty();
    }

    public List<Tree> children() {
        if(children == null)
            children = new ArrayList<>();

        return children;
    }

    /**
     * Node has one child that is a leaf
     * @return whether the node has one child and the child is a leaf
     */
    public boolean isPreTerminal() {
        if(children == null && label != null && !label.equals("TOP"))
            children = new ArrayList<>();
        if(children != null && children.size() == 1) {
            Tree child = children.get(0);
            return child != null && child.isLeaf();
        }
        return false;
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
     * @param node the node to getFromOrigin the distance from
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
     * Returns the ancestor of the given tree
     * @param height
     * @param root
     * @return {@link Tree}
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
        Tree ret = new Tree(this);
        ret.connect(new ArrayList<>(children()));
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

    public List<String> getTokens() {
        return tokens;
    }

    public void setTokens(List<String> tokens) {
        this.tokens = tokens;
    }

    public void setParent(Tree parent) {
        this.parent = parent;
    }

    public Tree parent() {
        return parent;
    }

    public INDArray vector() {
        return vector;
    }

    public void setVector(INDArray vector) {
        this.vector = vector;
    }

    public INDArray prediction() {
        return prediction;
    }

    public void setPrediction(INDArray prediction) {
        this.prediction = prediction;
    }

    public List<String> tags() {
        return tags;
    }

    public void setTags(List<String> tags) {
        this.tags = tags;
    }

    public List<Tree> getChildren() {
        return children;
    }

    public String getHeadWord() {
        return headWord;
    }

    public void setHeadWord(String headWord) {
        this.headWord = headWord;
    }

    public void setParse(String parse) {
        this.parse = parse;
    }

    /**
     * Connects the given trees
     * and sets the parents of the children
     * @param children  the children to connect with
     */
    public void connect(List<Tree> children) {
        this.children = children;
        for(Tree t : children)
            t.setParent(this);
    }

    public int getBegin() {
        return begin;
    }

    public void setBegin(int begin) {
        this.begin = begin;
    }

    public int getEnd() {
        return end;
    }

    public void setEnd(int end) {
        this.end = end;
    }

    @Override
    public String toString() {
        return "Tree{" +
                "vector=" + vector +
                ", prediction=" + prediction +
                ", error=" + error +
                ", parent=" + parent +
                ", headWord='" + headWord + '\'' +
                ", value='" + value + '\'' +
                ", label='" + label + '\'' +
                ", type='" + type + '\'' +
                ", goldLabel=" + goldLabel +
                ", tokens=" + tokens +
                ", tags=" + tags +
                ", parse='" + parse + '\'' +
                ", begin=" + begin +
                ", end=" + end +
                '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Tree)) return false;

        Tree tree = (Tree) o;

        if (begin != tree.begin) return false;
        if (end != tree.end) return false;
        if (Double.compare(tree.error, error) != 0) return false;
        if (goldLabel != tree.goldLabel) return false;
        if (headWord != null ? !headWord.equals(tree.headWord) : tree.headWord != null) return false;
        if (label != null ? !label.equals(tree.label) : tree.label != null) return false;
        if (parse != null ? !parse.equals(tree.parse) : tree.parse != null) return false;
        if (prediction != null ? !prediction.equals(tree.prediction) : tree.prediction != null) return false;
        if (tags != null ? !tags.equals(tree.tags) : tree.tags != null) return false;
        if (tokens != null ? !tokens.equals(tree.tokens) : tree.tokens != null) return false;
        if (type != null ? !type.equals(tree.type) : tree.type != null) return false;
        if (value != null ? !value.equals(tree.value) : tree.value != null) return false;
        return !(vector != null ? !vector.equals(tree.vector) : tree.vector != null);

    }

    @Override
    public int hashCode() {
        int result;
        long temp;
        result = vector != null ? vector.hashCode() : 0;
        result = 31 * result + (prediction != null ? prediction.hashCode() : 0);
        temp = Double.doubleToLongBits(error);
        result = 31 * result + (int) (temp ^ (temp >>> 32));
        result = 31 * result + (headWord != null ? headWord.hashCode() : 0);
        result = 31 * result + (value != null ? value.hashCode() : 0);
        result = 31 * result + (label != null ? label.hashCode() : 0);
        result = 31 * result + (type != null ? type.hashCode() : 0);
        result = 31 * result + goldLabel;
        result = 31 * result + (tokens != null ? tokens.hashCode() : 0);
        result = 31 * result + (tags != null ? tags.hashCode() : 0);
        result = 31 * result + (parse != null ? parse.hashCode() : 0);
        result = 31 * result + begin;
        result = 31 * result + end;
        return result;
    }
}