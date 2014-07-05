package org.deeplearning4j.text.treeparser;


import static org.junit.Assert.*;

import org.deeplearning4j.rntn.Tree;
import org.deeplearning4j.text.treeparser.transformer.TreeTransformer;
import org.junit.Before;
import org.junit.Test;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 7/1/14.
 */
public class TreeTransformerTests {

    private static Logger log = LoggerFactory.getLogger(TreeTransformerTests.class);
    private TreeParser parser;
    @Before
    public void init() throws Exception {
        parser = new TreeParser();
    }





    @Test
    public void testBinarize() throws Exception {
        List<Tree> trees = parser.getTrees("This is one sentence. This is another sentence.");
        TreeTransformer t = new BinarizeTreeTransformer();
        TreeTransformer cnf = new CollapseUnaries();
        for(Tree t2 : trees) {
            t2 = t.transform(t2);
            assertChildSize(t2);
            t2 = cnf.transform(t2);


        }
    }





    private void assertPreTerminalOrUnary(Tree tree) {
        for(Tree child : tree.children()) {
            assertPreTerminalOrUnary(child);
        }

        assertEquals(true,tree.children().size() == 1 || tree.isPreTerminal() || tree.isLeaf());


    }

    private void assertChildSize(Tree tree) {
        for(Tree child : tree.children()) {
            assertChildSize(child);
        }

        assertEquals(true,tree.isLeaf() || tree.isPreTerminal() || tree.children().size() <= 2);


    }


}
