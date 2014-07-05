package org.deeplearning4j.text.treeparser;

import org.cleartk.syntax.constituent.type.TreebankNode;
import org.deeplearning4j.rntn.Tree;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

/**
 * Basic Tree parser tests
 * @author  Adam Gibson
 */
public class TreeParserTest {
    private static Logger log = LoggerFactory.getLogger(TreeParserTest.class);
    private TreeParser parser;
    @Before
    public void init() throws Exception {
        parser = new TreeParser();
    }


    @Test
    public void testNumTrees() throws Exception {
        List<Tree> trees = parser.getTrees("This is one sentence. This is another sentence.");
        assertEquals(2,trees.size());

    }


    @Test
    public void testHierarchy() throws Exception {
        List<Tree> trees = parser.getTrees("This is one sentence. This is another sentence.");
        List<TreebankNode> treebankTrees = parser.getTreebankTrees("This is one sentence. This is another sentence.");
        assertEquals(treebankTrees.size(),trees.size());

        for(int i = 0; i < treebankTrees.size(); i++) {
            Tree t = trees.get(i);
            TreebankNode t2 = treebankTrees.get(i);
            assertEquals(t.children().size(),t2.getChildren().size());
        }

    }

}
