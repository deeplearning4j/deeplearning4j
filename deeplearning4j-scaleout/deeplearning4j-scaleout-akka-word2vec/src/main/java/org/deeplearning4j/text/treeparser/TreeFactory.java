package org.deeplearning4j.text.treeparser;


import org.apache.uima.fit.util.FSCollectionFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.tcas.Annotation;
import org.cleartk.syntax.constituent.type.TreebankNode;
import org.cleartk.token.type.Token;
import org.deeplearning4j.rntn.Tree;

import java.util.ArrayList;
import java.util.List;

/**
 *
 * Static util class handling the conversion of
 * treebank nodes to Trees useful for recursive neural tensor networks
 *
 * @author Adam Gibson
 */
public class TreeFactory {


    /**
     * Builds a tree recursively
     * adding the children as necessary
     * @param node the node to build the tree based on
     * @return the compiled tree with all of its children
     * and childrens' children recursively
     * @throws Exception
     */
    public static Tree buildTree(TreebankNode node) throws Exception {
        if(node.getLeaf())
            return toTree(node);
        else {
            List<TreebankNode> preChildren = children(node);
            List<Tree> children = new ArrayList<>();
            Tree t = toTree(node);
            for(int i = 0; i < preChildren.size(); i++) {
                children.add(buildTree(preChildren.get(i)));
            }

            t.connect(children);
            return t;

        }




    }

    /**
     * Converts a treebank node to a tree
     * @param node the node to convert
     * @return the tree with the same tokens and type as
     * the given tree bank node
     * @throws Exception
     */
    public static Tree toTree(TreebankNode node) throws Exception {
        List<String> tokens = tokens(node);
        Tree ret = new Tree(tokens);
        ret.setValue(node.getNodeValue());
        ret.setType(node.getNodeType());
        return ret;
    }




    private static List<TreebankNode> children(TreebankNode node) {
        return new ArrayList<>(FSCollectionFactory.create(node.getChildren(),TreebankNode.class));
    }

    private static List<String> tokens(Annotation ann) throws Exception {
        List<String> ret = new ArrayList<>();
        for(Token t : JCasUtil.select(ann.getCAS().getJCas(),Token.class)) {
            ret.add(t.getCoveredText());
        }
        return ret;
    }

}
