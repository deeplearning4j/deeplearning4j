package org.deeplearning4j.example.text;

import org.cleartk.syntax.constituent.type.TopTreebankNode;
import org.deeplearning4j.text.treeparser.TreeParser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by agibsonccc on 6/30/14.
 */
public class ParserExample {

    private static Logger log = LoggerFactory.getLogger(ParserExample.class);

    public static void main(String[] args) throws Exception {

        TreeParser parser = new TreeParser();
        TopTreebankNode node = parser.getTree("This is a tree.");
        log.info(node.getTreebankParse());

    }


}
