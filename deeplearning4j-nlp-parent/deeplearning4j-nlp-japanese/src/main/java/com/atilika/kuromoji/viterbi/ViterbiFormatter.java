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
package com.atilika.kuromoji.viterbi;

import com.atilika.kuromoji.dict.ConnectionCosts;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ViterbiFormatter {

    private final static String BOS_LABEL = "BOS";
    private final static String EOS_LABEL = "EOS";
    private final static String FONT_NAME = "Helvetica";

    private ConnectionCosts costs;
    private Map<String, ViterbiNode> nodeMap;
    private Map<String, String> bestPathMap;

    private boolean foundBOS;

    public ViterbiFormatter(ConnectionCosts costs) {
        this.costs = costs;
        this.nodeMap = new HashMap<>();
        this.bestPathMap = new HashMap<>();
    }

    public String format(ViterbiLattice lattice) {
        return format(lattice, null);
    }

    public String format(ViterbiLattice lattice, List<ViterbiNode> bestPath) {

        initBestPathMap(bestPath);

        StringBuilder builder = new StringBuilder();
        builder.append(formatHeader());
        builder.append(formatNodes(lattice));
        builder.append(formatTrailer());
        return builder.toString();

    }

    private void initBestPathMap(List<ViterbiNode> bestPath) {
        this.bestPathMap.clear();

        if (bestPath == null) {
            return;
        }
        for (int i = 0; i < bestPath.size() - 1; i++) {
            ViterbiNode from = bestPath.get(i);
            ViterbiNode to = bestPath.get(i + 1);

            String fromId = getNodeId(from);
            String toId = getNodeId(to);

            assert this.bestPathMap.containsKey(fromId) == false;
            assert this.bestPathMap.containsValue(toId) == false;
            this.bestPathMap.put(fromId, toId);
        }
    }

    private String formatNodes(ViterbiLattice lattice) {
        ViterbiNode[][] startsArray = lattice.getStartIndexArr();
        ViterbiNode[][] endsArray = lattice.getEndIndexArr();
        this.nodeMap.clear();
        this.foundBOS = false;

        StringBuilder builder = new StringBuilder();
        for (int i = 1; i < endsArray.length; i++) {
            if (endsArray[i] == null || startsArray[i] == null) {
                continue;
            }
            for (int j = 0; j < endsArray[i].length; j++) {
                ViterbiNode from = endsArray[i][j];
                if (from == null) {
                    continue;
                }
                builder.append(formatNodeIfNew(from));
                for (int k = 0; k < startsArray[i].length; k++) {
                    ViterbiNode to = startsArray[i][k];
                    if (to == null) {
                        break;
                    }
                    builder.append(formatNodeIfNew(to));
                    builder.append(formatEdge(from, to));
                }
            }
        }
        return builder.toString();
    }

    private String formatNodeIfNew(ViterbiNode node) {
        String nodeId = getNodeId(node);
        if (!this.nodeMap.containsKey(nodeId)) {
            this.nodeMap.put(nodeId, node);
            return formatNode(node);
        } else {
            return "";
        }
    }

    private String formatHeader() {
        StringBuilder builder = new StringBuilder();
        builder.append("digraph viterbi {\n");
        builder.append("graph [ fontsize=30 labelloc=\"t\" label=\"\" splines=true overlap=false rankdir = \"LR\" ];\n");
        builder.append("# A2 paper size\n");
        builder.append("size = \"34.4,16.5\";\n");
        builder.append("# try to fill paper\n");
        builder.append("ratio = fill;\n");
        builder.append("edge [ fontname=\"" + FONT_NAME + "\" fontcolor=\"red\" color=\"#606060\" ]\n");
        builder.append("node [ style=\"filled\" fillcolor=\"#e8e8f0\" shape=\"Mrecord\" fontname=\"" + FONT_NAME
                        + "\" ]\n");

        return builder.toString();
    }

    private String formatTrailer() {
        return "}";
    }


    private String formatEdge(ViterbiNode from, ViterbiNode to) {
        if (this.bestPathMap.containsKey(getNodeId(from))
                        && this.bestPathMap.get(getNodeId(from)).equals(getNodeId(to))) {
            return formatEdge(from, to, "color=\"#40e050\" fontcolor=\"#40a050\" penwidth=3 fontsize=20 ");

        } else {
            return formatEdge(from, to, "");
        }
    }


    private String formatEdge(ViterbiNode from, ViterbiNode to, String attributes) {
        StringBuilder builder = new StringBuilder();
        builder.append(getNodeId(from));
        builder.append(" -> ");
        builder.append(getNodeId(to));
        builder.append(" [ ");
        builder.append("label=\"");
        builder.append(getCost(from, to));
        builder.append("\"");
        builder.append(" ");
        builder.append(attributes);
        builder.append(" ");
        builder.append(" ]");
        builder.append("\n");
        return builder.toString();
    }

    private String formatNode(ViterbiNode node) {
        StringBuilder builder = new StringBuilder();
        builder.append("\"");
        builder.append(getNodeId(node));
        builder.append("\"");
        builder.append(" [ ");
        builder.append("label=");
        builder.append(formatNodeLabel(node));
        if (node.getType() == ViterbiNode.Type.USER) {
            builder.append(" fillcolor=\"#e8f8e8\"");
        } else if (node.getType() == ViterbiNode.Type.UNKNOWN) {
            builder.append(" fillcolor=\"#f8e8f8\"");
        } else if (node.getType() == ViterbiNode.Type.INSERTED) {
            builder.append(" fillcolor=\"#ffe8e8\"");
        }
        builder.append(" ]");
        return builder.toString();
    }

    private String formatNodeLabel(ViterbiNode node) {
        StringBuilder builder = new StringBuilder();
        builder.append("<<table border=\"0\" cellborder=\"0\">");
        builder.append("<tr><td>");
        builder.append(getNodeLabel(node));
        builder.append("</td></tr>");
        builder.append("<tr><td>");
        builder.append("<font color=\"blue\">");
        builder.append(node.getWordCost());
        builder.append("</font>");
        builder.append("</td></tr>");
        builder.append("</table>>");
        return builder.toString();
    }

    private String getNodeId(ViterbiNode node) {
        return String.valueOf(node.hashCode());
    }

    private String getNodeLabel(ViterbiNode node) {
        if (node.getType() == ViterbiNode.Type.KNOWN && node.getWordId() == 0) {
            if (this.foundBOS) {
                return EOS_LABEL;
            } else {
                this.foundBOS = true;
                return BOS_LABEL;
            }
        } else {
            return node.getSurface();
        }
    }

    private int getCost(ViterbiNode from, ViterbiNode to) {
        return this.costs.get(from.getLeftId(), to.getRightId());
    }
}
