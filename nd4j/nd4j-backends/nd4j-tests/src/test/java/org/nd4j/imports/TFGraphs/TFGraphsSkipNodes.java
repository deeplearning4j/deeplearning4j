package org.nd4j.imports.TFGraphs;

import java.util.*;

/**
 * Created by susaneraly on 2/20/18.
 */
public class TFGraphsSkipNodes {

    private static final Map<String, List<String>> SKIP_NODE_MAP = Collections.unmodifiableMap(
            new HashMap<String, List<String>>() {{
                //Note that we are testing equality with keep_prob of 1.0
                //The following are all dependent on rng seed and will fail. All other nodes pass.
                put("deep_mnist",
                        new ArrayList<>(Arrays.asList("dropout/dropout/random_uniform/RandomUniform",
                                "dropout/dropout/random_uniform/mul",
                                "dropout/dropout/random_uniform",
                                "dropout/dropout/add")));
            }});

    public static boolean skipNode(String modelName, String varName) {

        if (!SKIP_NODE_MAP.keySet().contains(modelName)) {
            return false;
        } else {
            for (String some_node : SKIP_NODE_MAP.get(modelName)) {
                if (some_node.equals(varName)) {
                    return true;
                }
            }
            return false;
        }

    }
}
