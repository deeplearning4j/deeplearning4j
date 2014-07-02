package org.deeplearning4j.example.text;

import org.cleartk.syntax.constituent.type.TopTreebankNode;
import org.deeplearning4j.rntn.Tree;
import org.deeplearning4j.text.treeparser.TreeParser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

/**
 * Created by agibsonccc on 6/30/14.
 */
public class ParserExample {

    private static Logger log = LoggerFactory.getLogger(ParserExample.class);

    public static void main(String[] args) throws Exception {

        TreeParser parser = new TreeParser();
        List<Tree> node = parser.getTrees("This is a tree.");

    }


}
