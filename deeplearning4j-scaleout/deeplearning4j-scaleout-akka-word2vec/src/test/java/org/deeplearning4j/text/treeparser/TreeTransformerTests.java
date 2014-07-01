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
        for(Tree t2 : trees) {
            t2 = t.transform(t2);
           for(Tree child : t2.children()) {
              assertEquals(true,child.isLeaf() || child.isPreTerminal() || child.children().size() <= 2);
           }

        }
    }

}
