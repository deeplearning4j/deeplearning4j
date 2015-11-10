package org.deeplearning4j.graph.models.deepwalk;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 10/11/2015.
 */
public class TestGraphHuffman {

    @Test
    public void testGraphHuffman(){
        //Simple test case from Weiss - Data Structires and Algorithm Analysis in Java 3ed pg436
        //Huffman code is non-unique, but length of code for each node is same for all Huffman codes

        GraphHuffman gh = new GraphHuffman(7);

        int[] vertexDegrees = {10, 15, 12, 3, 4, 13, 1};

        gh.buildTree(vertexDegrees);

//        for( int i=0; i<7; i++ ) System.out.println(i + "\t" + gh.getCodeLengthForVertex(i));

        int[] expectedLengths = {3,2,2,5,4,2,5};
        for( int i=0; i<vertexDegrees.length; i++ ){
            assertEquals(expectedLengths[i], gh.getCodeLengthForVertex(i));
        }
    }

}
