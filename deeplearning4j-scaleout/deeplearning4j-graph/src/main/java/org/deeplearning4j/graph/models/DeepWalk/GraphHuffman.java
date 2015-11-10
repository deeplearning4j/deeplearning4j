package org.deeplearning4j.graph.models.deepwalk;

import lombok.AllArgsConstructor;

import java.util.PriorityQueue;

/**A Huffman tree specifically for graphs.
 * Vertices in graph are indexed by an integer, 0 to nVertices-1
 */
public class GraphHuffman {

    private static final int MAX_CODE_LENGTH = 64;
    private final long[] codes;
    private final byte[] codeLength;


    public GraphHuffman(int nVertices){
        codes = new long[nVertices];
        codeLength = new byte[nVertices];
    }

    public void buildTree(int[] nodeDegree){


        PriorityQueue<Node> pq = new PriorityQueue<>();
        for( int i=0; i<nodeDegree.length; i++ ) pq.add(new Node(i,nodeDegree[i],null,null));

        while(pq.size() > 1){
            Node left = pq.remove();
            Node right = pq.remove();
            Node newNode = new Node(-1,left.count+right.count,left,right);
            pq.add(newNode);
        }

        //Eventually: only one node left -> full tree
        Node tree = pq.remove();

        //Now: convert tree into binary codes.
        //How? traverse tree. Preorder traversal
        traverse(tree,0L,(byte)0);
    }

    @AllArgsConstructor
    private static class Node implements Comparable<Node>{
        private final int vertexIdx;
        private final long count;
        private Node left;
        private Node right;

        @Override
        public int compareTo(Node o) {
            return Long.compare(count,o.count);
        }
    }

    private void traverse(Node node, long codeSoFar, byte codeLengthSoFar ){
        if(codeLengthSoFar >= 64) throw new RuntimeException("Cannot generate code: code length exceeds 64 bits");
        if(node.left == null && node.right == null){
            //Leaf node
            codes[node.vertexIdx] = codeSoFar;
            codeLength[node.vertexIdx] = codeLengthSoFar;
            return;
        }

        byte b = 1;

        long codeLeft = setBit(codeSoFar, codeLengthSoFar+1, false);
        traverse(node.left,codeLeft,(byte)(codeLengthSoFar+1));

        long codeRight = setBit(codeSoFar, codeLengthSoFar + 1, true);
        traverse(node.right,codeRight,(byte)(codeLengthSoFar+1));
    }

    private long setBit(long in, int bitNum, boolean value){
        if(value) return (in |= 1L << bitNum);  //Bit mask |: 00010000
        else return (in &= ~(1 << bitNum));     //Bit mask &: 11101111
    }

    public long getCodeForVertex(int vertexNum){
        return codes[vertexNum];
    }

    public byte getCodeLengthForVertex(int vertexNum){
        return codeLength[vertexNum];
    }

//    public void buildTree(int[] nodeDegree){
//        //Undirected graph: probability of visiting vertex v during random walk
//        // is degree(v)/2m, where m is number of edges in graph
//        // thus, degree(v) is proportional to probability
//
////        PriorityQueue<?> queue = new PriorityQueue<>();
//
//
//
//        int[] count = new int[nodeDegree.length * 2 + 1];
//        int[] binary = new int[nodeDegree.length * 2 + 1];
//        int[] code = new int[MAX_CODE_LENGTH];
//        int[] point = new int[MAX_CODE_LENGTH];
//        int[] parentNode = new int[nodeDegree.length * 2 + 1];
//        int a = 0;
//
//        while (a < nodeDegree.length) {
//            count[a] = nodeDegree[a];
//            a++;
//        }
//
//        a = nodeDegree.length;
//
//        while(a < nodeDegree.length * 2) {
//            count[a] = Integer.MAX_VALUE;
//            a++;
//        }
//
//        int pos1 = nodeDegree.length - 1;
//        int pos2 = nodeDegree.length;
//
//        int min1i;
//        int min2i;
//
//        a = 0;
//        // Following algorithm constructs the Huffman tree by adding one node at a time
//        for (a = 0; a < nodeDegree.length - 1; a++) {
//            // First, find two smallest nodes 'min1, min2'
//            if (pos1 >= 0) {
//                if (count[pos1] < count[pos2]) {
//                    min1i = pos1;
//                    pos1--;
//                } else {
//                    min1i = pos2;
//                    pos2++;
//                }
//            } else {
//                min1i = pos2;
//                pos2++;
//            }
//            if (pos1 >= 0) {
//                if (count[pos1] < count[pos2]) {
//                    min2i = pos1;
//                    pos1--;
//                } else {
//                    min2i = pos2;
//                    pos2++;
//                }
//            } else {
//                min2i = pos2;
//                pos2++;
//            }
//
//            count[nodeDegree.length + a] = count[min1i] + count[min2i];
//            parentNode[min1i] = nodeDegree.length + a;
//            parentNode[min2i] = nodeDegree.length + a;
//            binary[min2i] = 1;
//        }
//        // Now assign binary code to each vocabulary word
//        int i ;
//        int b;
//        // Now assign binary code to each vocabulary word
//        for (a = 0; a < nodeDegree.length; a++) {
//            b = a;
//            i = 0;
//            do {
//                code[i] = binary[b];
//                point[i] = b;
//                i++;
//                b = parentNode[b];
//
//            } while(b != nodeDegree.length * 2 - 2 && i < 39);
//
//
//            codeLength[a] = (byte)i;
////            words.get(a).setCodeLength(i);
////            words.get(a).getPoints().add(nodeDegree.length - 2);
//
//            codes[a] = 0L;
//            for (b = 0; b < i; b++) {
//                codes[a] = setBit(codes[a],i-b-1,(code[b]==1));
////                words.get(a).getCodes().set(i - b - 1,code[b]);
////                words.get(a).getPoints().set(i - b,point[b] - nodeDegree.length);
//            }
//        }
//
//    }



}
