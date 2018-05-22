/*-*
 * Copyright © 2010-2015 Atilika Inc. and contributors (see CONTRIBUTORS.md)
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License.  A copy of the
 * License is distributed with this work in the LICENSE.md file.  You may
 * also obtain a copy of the License from
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.atilika.kuromoji.trie;

import com.atilika.kuromoji.trie.PatriciaTrie.KeyMapper;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintWriter;

/**
 * Utility class to format a {@link PatriciaTrie} to dot format for debugging, inspection, etc.
 *
 * @param <V> value type
 *
 * See @see <a href="http://graphviz.org/">Graphviz</a>
 */
public class PatriciaTrieFormatter<V> {

    private final static String FONT_NAME = "Helvetica";

    /**
     * Constructor
     */
    public PatriciaTrieFormatter() {}

    /**
     * Format trie
     *
     * @param trie  trie to format
     * @return formatted trie, not null
     */
    public String format(PatriciaTrie<V> trie) {
        return format(trie, true);
    }

    /**
     * Format trie
     *
     * @param trie  trie to format
     * @param formatBitString  true if the bits for this key should be included in the node
     * @return formatted trie, not null
     */
    public String format(PatriciaTrie<V> trie, boolean formatBitString) {
        StringBuilder builder = new StringBuilder();
        builder.append(formatHeader());
        builder.append(formatNode(trie.getRoot().getLeft(), -1, trie.getKeyMapper(), formatBitString));
        builder.append(formatTrailer());
        return builder.toString();
    }

    /**
     * Format trie and write to file
     *
     * @param trie  trie to format
     * @param file  file to write to
     * @throws FileNotFoundException if the file exists but is a directory rather than a regular file,
     * does not exist but cannot be created, or cannot be opened for any other reason
     */
    public void format(PatriciaTrie<V> trie, File file) throws FileNotFoundException {
        format(trie, file, false);
    }

    /**
     * Format trie and write to file
     *
     * @param trie  trie to format
     * @param file  file to write to
     * @param formatBitString  true if the bits for this key should be included in the node
     * @throws FileNotFoundException if the file exists but is a directory rather than a regular file,
     * does not exist but cannot be created, or cannot be opened for any other reason
     */
    public void format(PatriciaTrie<V> trie, File file, boolean formatBitString) throws FileNotFoundException {
        PrintWriter writer = new PrintWriter(new FileOutputStream(file));
        writer.println(format(trie, formatBitString));
        writer.close();
    }

    /**
     * Format header
     *
     * @return formatted header, not null
     */
    private String formatHeader() {
        StringBuilder builder = new StringBuilder();
        builder.append("digraph patricia {\n");
        //      builder.append("graph [ fontsize=30 labelloc=\"t\" label=\"\" splines=true overlap=false ];\n");
        //      builder.append("# A2 paper size\n");
        //      builder.append("size = \"34.4,16.5\";\n");
        //      builder.append("# try to fill paper\n");
        //      builder.append("ratio = fill;\n");
        //      builder.append("edge [ fontname=\"" + FONT_NAME + "\" fontcolor=\"red\" color=\"#606060\" ]\n");
        builder.append("nodesep=1.5;");
        builder.append("node [ style=\"filled\" fillcolor=\"#e8e8f0\" shape=\"Mrecord\" fontname=\"" + FONT_NAME
                        + "\" ]\n");
        //      builder.append("edge [ fontname=\"" + FONT_NAME + "\" fontcolor=\"red\" color=\"#606060\" ]\n");
        //      builder.append("node [ shape=\"circle\" ]\n");
        return builder.toString();
    }

    /**
     * Format trailer
     *
     * @return formatted trailer
     */
    private String formatTrailer() {
        return "}";
    }

    /**
     * Formats nodes
     *
     * @param node  node to format
     * @param bit  bit for this node
     * @param keyMapper  keymapper to map keys to bits
     * @param formatBitString  true if the bits for this key should be included in the node
     * @return formatted node, not null
     */
    private String formatNode(PatriciaTrie.PatriciaNode<V> node, int bit, KeyMapper<String> keyMapper,
                    boolean formatBitString) {
        if (node.getBit() <= bit) {
            return "";
        } else {
            StringBuffer buffer = new StringBuffer();
            buffer.append("\"");
            buffer.append(getNodeId(node));
            buffer.append("\"");
            buffer.append(" [ ");
            buffer.append("label=");
            buffer.append(formatNodeLabel(node, keyMapper, formatBitString));
            buffer.append(" ]");
            buffer.append("\n");

            buffer.append(formatPointer(node, node.getLeft(), "l", "sw"));
            buffer.append(formatPointer(node, node.getRight(), "r", "se"));

            buffer.append(formatNode(node.getLeft(), node.getBit(), keyMapper, formatBitString));
            buffer.append(formatNode(node.getRight(), node.getBit(), keyMapper, formatBitString));

            return buffer.toString();
        }
    }

    /**
     * Formats a link between two nodes
     *
     * @param from  from node
     * @param to  to node
     * @param label  label for this link
     * @param tailport  tail port to use when formatting (dot-specific, "sw" or "se)
     * @return formatted link, not null
     */
    private String formatPointer(PatriciaTrie.PatriciaNode<V> from, PatriciaTrie.PatriciaNode<V> to, String label,
                    String tailport) {
        StringBuilder builder = new StringBuilder();
        builder.append(getNodeId(from));
        builder.append(" -> ");
        builder.append(getNodeId(to));
        builder.append(" [ ");
        builder.append("label=\"");
        builder.append(label);
        builder.append(" \"");
        builder.append("tailport=\"");
        builder.append(tailport);
        builder.append(" \"");
        builder.append("fontcolor=\"#666666\" ");
        builder.append(" ]");
        builder.append("\n");
        return builder.toString();
    }

    /**
     * Format node label
     *
     * @param node  node to format
     * @param keyMapper  keymapper to map keys to bits
     * @param formatBitString  true if the bits for this key should be included in the node
     * @return formatted formatted node, not null
     */
    private String formatNodeLabel(PatriciaTrie.PatriciaNode<V> node, KeyMapper<String> keyMapper,
                    boolean formatBitString) {
        StringBuilder builder = new StringBuilder();
        builder.append("<<table border=\"0\" cellborder=\"0\">");
        // Key
        builder.append("<tr><td>");
        builder.append("key: <font color=\"#00a000\">");
        builder.append(getNodeLabel(node));
        builder.append("</font> </td></tr>");

        // Critical bit
        builder.append("<tr><td>");
        builder.append("bit: <font color=\"blue\">");
        builder.append(node.getBit());
        builder.append("</font> </td></tr>");

        // Bit string
        if (formatBitString) {
            builder.append("<tr><td>");
            builder.append("bitString: <font color=\"blue\">");
            String bitString = keyMapper.toBitString(node.getKey());
            int c = node.getBit() + node.getBit() / 4;
            builder.append(bitString.substring(0, c));
            builder.append("<font color=\"red\">");
            builder.append(bitString.charAt(c));
            builder.append("</font>");
            builder.append(bitString.substring(c + 1));
            builder.append("</font> </td></tr>");
        }

        // Value
        builder.append("<tr><td>");
        builder.append("value: <font color=\"#00a0a0\">");
        builder.append(node.getValue());
        builder.append("</font> </td></tr>");

        builder.append("</table>>");
        return builder.toString();
    }

    /**
     * Get node label
     *
     * @param node
     * @return label, not null
     */
    private String getNodeLabel(PatriciaTrie.PatriciaNode<V> node) {
        return node.getKey();
    }

    /**
     * Get node id used to distinguish nodes internally
     *
     * @param node
     * @return node id, not null
     */
    private String getNodeId(PatriciaTrie.PatriciaNode<V> node) {
        if (node == null) {
            return "null";
        } else {
            return node.getKey();
        }
    }
}
