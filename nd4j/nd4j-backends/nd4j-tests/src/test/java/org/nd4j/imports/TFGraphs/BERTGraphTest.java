package org.nd4j.imports.TFGraphs;

import lombok.extern.slf4j.Slf4j;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.autodiff.samediff.transform.*;
import org.nd4j.base.Preconditions;
import org.nd4j.graph.ui.LogFileWriter;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.imports.tensorflow.TFImportOverride;
import org.nd4j.imports.tensorflow.TFOpImportFilter;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.resources.Downloader;
import org.nd4j.util.ArchiveUtils;

import java.io.File;
import java.net.URL;
import java.util.*;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

@Slf4j
public class BERTGraphTest {

    @Test
    public void testBert() throws Exception {

        String url = "https://deeplearning4jblob.blob.core.windows.net/testresources/bert_mrpc_frozen_v1.zip";
        File saveDir = new File(TFGraphTestZooModels.getBaseModelDir(), ".nd4jtests/bert_mrpc_frozen_v1");
        saveDir.mkdirs();

        File localFile = new File(saveDir, "bert_mrpc_frozen_v1.zip");
        String md5 = "7cef8bbe62e701212472f77a0361f443";

        if(localFile.exists() && !Downloader.checkMD5OfFile(md5, localFile)) {
            log.info("Deleting local file: does not match MD5. {}", localFile.getAbsolutePath());
            localFile.delete();
        }

        if (!localFile.exists()) {
            log.info("Starting resource download from: {} to {}", url, localFile.getAbsolutePath());
            Downloader.download("BERT MRPC", new URL(url), localFile, md5, 3);
        }

        //Extract
        File f = new File(saveDir, "bert_mrpc_frozen.pb");
        if(!f.exists() || !Downloader.checkMD5OfFile("93d82bca887625632578df37ea3d3ca5", f)){
            if(f.exists()) {
                f.delete();
            }
            ArchiveUtils.zipExtractSingleFile(localFile, f, "bert_mrpc_frozen.pb");
        }

        /*
        Important node: This BERT model uses a FIXED (hardcoded) minibatch size, not dynamic as most models use
         */
        int minibatchSize = 4;

        /*
         * Define: Op import overrides. This is used to skip the IteratorGetNext node and instead crate some placeholders
         */
        Map<String, TFImportOverride> m = new HashMap<>();
        m.put("IteratorGetNext", (inputs, controlDepInputs, nodeDef, initWith, attributesForNode, graph) -> {
            //Return 3 placeholders called "IteratorGetNext:0", "IteratorGetNext:1", "IteratorGetNext:3" instead of the training iterator
            return Arrays.asList(
                    initWith.placeHolder("IteratorGetNext", DataType.INT, minibatchSize, 128),
                    initWith.placeHolder("IteratorGetNext:1", DataType.INT, minibatchSize, 128),
                    initWith.placeHolder("IteratorGetNext:4", DataType.INT, minibatchSize, 128)
            );
        });

        //Skip the "IteratorV2" op - we don't want or need this
        TFOpImportFilter filter = (nodeDef, initWith, attributesForNode, graph) -> { return "IteratorV2".equals(nodeDef.getName()); };

        SameDiff sd = TFGraphMapper.getInstance().importGraph(f, m, filter);

        /*
        Modify the network to remove hard-coded dropout operations for inference.
        This is a little ugly as Tensorflow/BERT's dropout is implemented as a set of discrete operations - random, mul, div, floor, etc.
        We need to select all instances of this subgraph, and then remove them from the graph entirely.

        Note that in general there are two ways to define subgraphs (larger than 1 operation) for use in GraphTransformUtil
        (a) withInputSubgraph - the input must match this predicate, AND it is added to the subgraph (i.e., matched and is selected to be part of the subgraph)
        (b) withInputMatching - the input must match this predicate, BUT it is NOT added to the subgraph (i.e., must match only)

        In effect, this predicate will match the set of directly connected operations with the following structure:
        (.../dropout/div, .../dropout/Floor) -> (.../dropout/mul)
        (.../dropout/add) -> (.../dropout/Floor)
        (.../dropout/random_uniform) -> (.../dropout/add)
        (.../dropout/random_uniform/mul) -> (.../dropout/random_uniform)
        (.../dropout/random_uniform/RandomUniform, .../dropout/random_uniform/sub) -> (.../dropout/random_uniform/mul)

        Then, for all subgraphs that match this predicate, we will process them (in this case, simply replace the entire subgraph by passing the input to the output)

        How do you work out the appropriate subgraph to replace?
        The simplest approach is to visualize the graph - either in TensorBoard or using SameDiff UI.
        See writeBertUI() in this file, then open DL4J UI and go to localhost:9000/samediff
        */
        SubGraphPredicate p = SubGraphPredicate.withRoot(OpPredicate.nameMatches(".*/dropout/mul"))     //.../dropout/mul is the output variable, post dropout
                .withInputCount(2)
                .withInputSubgraph(0, SubGraphPredicate.withRoot(OpPredicate.nameMatches(".*/dropout/div")))        //.../dropout/div is the first input. "withInputS
                .withInputSubgraph(1, SubGraphPredicate.withRoot(OpPredicate.nameMatches(".*/dropout/Floor"))
                        .withInputSubgraph(0, SubGraphPredicate.withRoot(OpPredicate.nameMatches(".*/dropout/add"))
                                .withInputSubgraph(1, SubGraphPredicate.withRoot(OpPredicate.nameMatches(".*/dropout/random_uniform"))
                                        .withInputSubgraph(0, SubGraphPredicate.withRoot(OpPredicate.nameMatches(".*/dropout/random_uniform/mul"))
                                                .withInputSubgraph(0, SubGraphPredicate.withRoot(OpPredicate.nameMatches(".*/dropout/random_uniform/RandomUniform")))
                                                .withInputSubgraph(1, SubGraphPredicate.withRoot(OpPredicate.nameMatches(".*/dropout/random_uniform/sub")))

                                        )
                                )
                        )
                );

        List<SubGraph> subGraphs = GraphTransformUtil.getSubgraphsMatching(sd, p);
        int subGraphCount = subGraphs.size();
        assertTrue("Subgraph count: " + subGraphCount, subGraphCount > 0);


        /*
        Create the subgraph processor.
        The subgraph processor is applied to each subgraph - i.e., it defines what we should replace it with.
        It's a 2-step process:
        (1) The SubGraphProcessor is applied to define the replacement subgraph (add any new operations, and define the new outputs, etc).
            In this case, we aren't adding any new ops - so we'll just pass the "real" input (pre dropout activations) to the output.
            Note that the number of returned outputs must match the existing number of outputs (1 in this case).
            Immediately after SubgraphProcessor.processSubgraph returns, both the existing subgraph (to be replaced) and new subgraph (just added)
            exist in parallel.
        (2) The existing subgraph is then removed from the graph, leaving only the new subgraph (as defined in processSubgraph method)
            in its place.

         Note that the order of the outputs you return matters!
         If the original outputs are [A,B,C] and you return output variables [X,Y,Z], then anywhere "A" was used as input
         will now use "X"; similarly Y replaces B, and Z replaces C.
         */
        sd = GraphTransformUtil.replaceSubgraphsMatching(sd, p, new SubGraphProcessor() {
            @Override
            public List<SDVariable> processSubgraph(SameDiff sd, SubGraph subGraph) {
                List<SDVariable> inputs = subGraph.inputs();    //Get inputs to the subgraph
                //Find pre-dropout input variable:
                SDVariable newOut = null;
                for(SDVariable v : inputs){
                    if(v.getVarName().endsWith("/BiasAdd") || v.getVarName().endsWith("/Softmax") || v.getVarName().endsWith("/add_1") || v.getVarName().endsWith("/Tanh")){
                        newOut = v;
                        break;
                    }
                }

                if(newOut != null){
                    //Pass this input variable as the new output
                    return Collections.singletonList(newOut);
                }

                throw new RuntimeException("No pre-dropout input variable found");
            }
        });

        //Small test / sanity check for asFlatPrint():
        sd.asFlatPrint();


        /*
        Output during inference:
        INFO:tensorflow:*** Example ***
        INFO:tensorflow:guid: test-1
        INFO:tensorflow:tokens: [CLS] the broader standard & poor ' s 500 index < . sp ##x > was 0 . 46 points lower , or 0 . 05 percent , at 99 ##7 . 02 . [SEP] the technology - laced nas ##da ##q composite index . ix ##ic was up 7 . 42 points , or 0 . 45 percent , at 1 , 65 ##3 . 44 . [SEP]
        INFO:tensorflow:input_ids: 101 1996 12368 3115 1004 3532 1005 1055 3156 5950 1026 1012 11867 2595 1028 2001 1014 1012 4805 2685 2896 1010 2030 1014 1012 5709 3867 1010 2012 5585 2581 1012 6185 1012 102 1996 2974 1011 17958 17235 2850 4160 12490 5950 1012 11814 2594 2001 2039 1021 1012 4413 2685 1010 2030 1014 1012 3429 3867 1010 2012 1015 1010 3515 2509 1012 4008 1012 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        INFO:tensorflow:input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        INFO:tensorflow:segment_ids: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        INFO:tensorflow:label: 0 (id = 0)
        INFO:tensorflow:*** Example ***
        INFO:tensorflow:guid: test-2
        INFO:tensorflow:tokens: [CLS] shares in ba were down 1 . 5 percent at 168 pen ##ce by 142 ##0 gm ##t , off a low of 164 ##p , in a slightly stronger overall london market . [SEP] shares in ba were down three percent at 165 - 1 / 4 pen ##ce by 09 ##33 gm ##t , off a low of 164 pen ##ce , in a stronger market . [SEP]
        INFO:tensorflow:input_ids: 101 6661 1999 8670 2020 2091 1015 1012 1019 3867 2012 16923 7279 3401 2011 16087 2692 13938 2102 1010 2125 1037 2659 1997 17943 2361 1010 1999 1037 3621 6428 3452 2414 3006 1012 102 6661 1999 8670 2020 2091 2093 3867 2012 13913 1011 1015 1013 1018 7279 3401 2011 5641 22394 13938 2102 1010 2125 1037 2659 1997 17943 7279 3401 1010 1999 1037 6428 3006 1012 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        INFO:tensorflow:input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        INFO:tensorflow:segment_ids: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        INFO:tensorflow:label: 0 (id = 0)
        INFO:tensorflow:*** Example ***
        INFO:tensorflow:guid: test-3
        INFO:tensorflow:tokens: [CLS] last year , com ##cast signed 1 . 5 million new digital cable subscribers . [SEP] com ##cast has about 21 . 3 million cable subscribers , many in the largest u . s . cities . [SEP]
        INFO:tensorflow:input_ids: 101 2197 2095 1010 4012 10526 2772 1015 1012 1019 2454 2047 3617 5830 17073 1012 102 4012 10526 2038 2055 2538 1012 1017 2454 5830 17073 1010 2116 1999 1996 2922 1057 1012 1055 1012 3655 1012 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        INFO:tensorflow:input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        INFO:tensorflow:segment_ids: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        INFO:tensorflow:label: 0 (id = 0)
        INFO:tensorflow:*** Example ***
        INFO:tensorflow:guid: test-4
        INFO:tensorflow:tokens: [CLS] revenue rose 3 . 9 percent , to $ 1 . 63 billion from $ 1 . 57 billion . [SEP] the mclean , virginia - based company said newspaper revenue increased 5 percent to $ 1 . 46 billion . [SEP]
        INFO:tensorflow:input_ids: 101 6599 3123 1017 1012 1023 3867 1010 2000 1002 1015 1012 6191 4551 2013 1002 1015 1012 5401 4551 1012 102 1996 17602 1010 3448 1011 2241 2194 2056 3780 6599 3445 1019 3867 2000 1002 1015 1012 4805 4551 1012 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        INFO:tensorflow:input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        INFO:tensorflow:segment_ids: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        INFO:tensorflow:label: 0 (id = 0)
         */
        INDArray ex1Idxs = Nd4j.createFromArray(101,1996,12368,3115,1004,3532,1005,1055,3156,5950,1026,1012,11867,2595,1028,2001,1014,1012,4805,2685,2896,1010,2030,1014,1012,5709,3867,1010,2012,5585,2581,1012,6185,1012,102,1996,2974,1011,17958,17235,2850,4160,12490,5950,1012,11814,2594,2001,2039,1021,1012,4413,2685,1010,2030,1014,1012,3429,3867,1010,2012,1015,1010,3515,2509,1012,4008,1012,102,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
        INDArray ex1Mask = Nd4j.createFromArray(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
        INDArray ex1SegmentId = Nd4j.createFromArray(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);

        INDArray ex2Idxs = Nd4j.createFromArray(101,6661,1999,8670,2020,2091,1015,1012,1019,3867,2012,16923,7279,3401,2011,16087,2692,13938,2102,1010,2125,1037,2659,1997,17943,2361,1010,1999,1037,3621,6428,3452,2414,3006,1012,102,6661,1999,8670,2020,2091,2093,3867,2012,13913,1011,1015,1013,1018,7279,3401,2011,5641,22394,13938,2102,1010,2125,1037,2659,1997,17943,7279,3401,1010,1999,1037,6428,3006,1012,102,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
        INDArray ex2Mask = Nd4j.createFromArray(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
        INDArray ex2SegmentId = Nd4j.createFromArray(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);

        INDArray ex3Idxs = Nd4j.createFromArray(101,2197,2095,1010,4012,10526,2772,1015,1012,1019,2454,2047,3617,5830,17073,1012,102,4012,10526,2038,2055,2538,1012,1017,2454,5830,17073,1010,2116,1999,1996,2922,1057,1012,1055,1012,3655,1012,102,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
        INDArray ex3Mask = Nd4j.createFromArray(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
        INDArray ex3SegmentId = Nd4j.createFromArray(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);

        INDArray ex4Idxs = Nd4j.createFromArray(101,6599,3123,1017,1012,1023,3867,1010,2000,1002,1015,1012,6191,4551,2013,1002,1015,1012,5401,4551,1012,102,1996,17602,1010,3448,1011,2241,2194,2056,3780,6599,3445,1019,3867,2000,1002,1015,1012,4805,4551,1012,102,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
        INDArray ex4Mask = Nd4j.createFromArray(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
        INDArray ex4SegmentId = Nd4j.createFromArray(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);

        INDArray idxs = Nd4j.vstack(ex1Idxs, ex2Idxs, ex3Idxs, ex4Idxs);
        INDArray mask = Nd4j.vstack(ex1Mask, ex2Mask, ex3Mask, ex4Mask);
        INDArray segmentIdxs = Nd4j.vstack(ex1SegmentId, ex2SegmentId, ex3SegmentId, ex4SegmentId);

        Map<String, INDArray> placeholderValues = new HashMap<>();
        placeholderValues.put("IteratorGetNext", idxs);
        placeholderValues.put("IteratorGetNext:1", mask);
        placeholderValues.put("IteratorGetNext:4", segmentIdxs);

        Map<String, INDArray> out = sd.exec(placeholderValues, "loss/Softmax");
        INDArray softmax = out.get("loss/Softmax");
//        System.out.println("OUTPUT - Softmax");
//        System.out.println(softmax);
//        System.out.println(Arrays.toString(softmax.data().asFloat()));

        INDArray exp0 = Nd4j.createFromArray(0.99860954f, 0.0013904407f);
        INDArray exp1 = Nd4j.createFromArray(0.0005442508f, 0.99945575f);
        INDArray exp2 = Nd4j.createFromArray(0.9987967f, 0.0012033002f);
        INDArray exp3 = Nd4j.createFromArray(0.97409827f, 0.025901746f);

        assertEquals(exp0, softmax.getRow(0));
        assertEquals(exp1, softmax.getRow(1));
        assertEquals(exp2, softmax.getRow(2));
        assertEquals(exp3, softmax.getRow(3));
    }

    @Test @Ignore   //AB ignored 08/04/2019 until fixed
    public void testBertTraining() throws Exception {
        String url = "https://deeplearning4jblob.blob.core.windows.net/testresources/bert_mrpc_frozen_v1.zip";
        File saveDir = new File(TFGraphTestZooModels.getBaseModelDir(), ".nd4jtests/bert_mrpc_frozen_v1");
        saveDir.mkdirs();

        File localFile = new File(saveDir, "bert_mrpc_frozen_v1.zip");
        String md5 = "7cef8bbe62e701212472f77a0361f443";

        if(localFile.exists() && !Downloader.checkMD5OfFile(md5, localFile)) {
            log.info("Deleting local file: does not match MD5. {}", localFile.getAbsolutePath());
            localFile.delete();
        }

        if (!localFile.exists()) {
            log.info("Starting resource download from: {} to {}", url, localFile.getAbsolutePath());
            Downloader.download("BERT MRPC", new URL(url), localFile, md5, 3);
        }

        //Extract
        File f = new File(saveDir, "bert_mrpc_frozen.pb");
        if(!f.exists() || !Downloader.checkMD5OfFile("93d82bca887625632578df37ea3d3ca5", f)){
            if(f.exists()) {
                f.delete();
            }
            ArchiveUtils.zipExtractSingleFile(localFile, f, "bert_mrpc_frozen.pb");
        }

        /*
        Important node: This BERT model uses a FIXED (hardcoded) minibatch size, not dynamic as most models use
         */
        int minibatchSize = 4;

        /*
         * Define: Op import overrides. This is used to skip the IteratorGetNext node and instead crate some placeholders
         */
        Map<String, TFImportOverride> m = new HashMap<>();
        m.put("IteratorGetNext", (inputs, controlDepInputs, nodeDef, initWith, attributesForNode, graph) -> {
            //Return 3 placeholders called "IteratorGetNext:0", "IteratorGetNext:1", "IteratorGetNext:3" instead of the training iterator
            return Arrays.asList(
                    initWith.placeHolder("IteratorGetNext", DataType.INT, minibatchSize, 128),
                    initWith.placeHolder("IteratorGetNext:1", DataType.INT, minibatchSize, 128),
                    initWith.placeHolder("IteratorGetNext:4", DataType.INT, minibatchSize, 128)
            );
        });

        //Skip the "IteratorV2" op - we don't want or need this
        TFOpImportFilter filter = (nodeDef, initWith, attributesForNode, graph) -> { return "IteratorV2".equals(nodeDef.getName()); };

        SameDiff sd = TFGraphMapper.getInstance().importGraph(f, m, filter);

        //For training, convert weights and biases from constants to variables:
        for(SDVariable v : sd.variables()){
            if(v.isConstant() && v.dataType().isFPType()){
                log.info("Converting to variable: {} - shape {}", v.getVarName(), v.shape());
                v.convertToVariable();
            }
        }

        System.out.println("INPUTS: " + sd.inputs());
        System.out.println("OUTPUTS: " + sd.outputs());

        //For training, we'll need to add a label placeholder for one-hot labels:
        SDVariable label = sd.placeHolder("label", DataType.FLOAT, 4, 2);
        SDVariable softmax = sd.getVariable("loss/Softmax");
        sd.loss().logLoss("loss", label, softmax);
        assertEquals(Collections.singletonList("loss"), sd.getLossVariables());

        //Peform simple overfitting test - same input, but inverted labels

        INDArray ex1Idxs = Nd4j.createFromArray(101,1996,12368,3115,1004,3532,1005,1055,3156,5950,1026,1012,11867,2595,1028,2001,1014,1012,4805,2685,2896,1010,2030,1014,1012,5709,3867,1010,2012,5585,2581,1012,6185,1012,102,1996,2974,1011,17958,17235,2850,4160,12490,5950,1012,11814,2594,2001,2039,1021,1012,4413,2685,1010,2030,1014,1012,3429,3867,1010,2012,1015,1010,3515,2509,1012,4008,1012,102,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
        INDArray ex1Mask = Nd4j.createFromArray(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
        INDArray ex1SegmentId = Nd4j.createFromArray(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);

        INDArray ex2Idxs = Nd4j.createFromArray(101,6661,1999,8670,2020,2091,1015,1012,1019,3867,2012,16923,7279,3401,2011,16087,2692,13938,2102,1010,2125,1037,2659,1997,17943,2361,1010,1999,1037,3621,6428,3452,2414,3006,1012,102,6661,1999,8670,2020,2091,2093,3867,2012,13913,1011,1015,1013,1018,7279,3401,2011,5641,22394,13938,2102,1010,2125,1037,2659,1997,17943,7279,3401,1010,1999,1037,6428,3006,1012,102,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
        INDArray ex2Mask = Nd4j.createFromArray(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
        INDArray ex2SegmentId = Nd4j.createFromArray(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);

        INDArray ex3Idxs = Nd4j.createFromArray(101,2197,2095,1010,4012,10526,2772,1015,1012,1019,2454,2047,3617,5830,17073,1012,102,4012,10526,2038,2055,2538,1012,1017,2454,5830,17073,1010,2116,1999,1996,2922,1057,1012,1055,1012,3655,1012,102,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
        INDArray ex3Mask = Nd4j.createFromArray(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
        INDArray ex3SegmentId = Nd4j.createFromArray(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);

        INDArray ex4Idxs = Nd4j.createFromArray(101,6599,3123,1017,1012,1023,3867,1010,2000,1002,1015,1012,6191,4551,2013,1002,1015,1012,5401,4551,1012,102,1996,17602,1010,3448,1011,2241,2194,2056,3780,6599,3445,1019,3867,2000,1002,1015,1012,4805,4551,1012,102,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
        INDArray ex4Mask = Nd4j.createFromArray(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
        INDArray ex4SegmentId = Nd4j.createFromArray(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);

        INDArray idxs = Nd4j.vstack(ex1Idxs, ex2Idxs, ex3Idxs, ex4Idxs);
        INDArray mask = Nd4j.vstack(ex1Mask, ex2Mask, ex3Mask, ex4Mask);
        INDArray segmentIdxs = Nd4j.vstack(ex1SegmentId, ex2SegmentId, ex3SegmentId, ex4SegmentId);
        INDArray labelArr = Nd4j.createFromArray(new float[][]{
                {1, 0},
                {0, 1},
                {1, 0},
                {1, 0}});

        TrainingConfig c = TrainingConfig.builder()
                .updater(new Adam(2e-5))
                .l2(1e-5)
                .dataSetFeatureMapping("IteratorGetNext", "IteratorGetNext:1", "IteratorGetNext:4")
                .dataSetLabelMapping("label")
                .build();
        sd.setTrainingConfig(c);

        MultiDataSet mds = new org.nd4j.linalg.dataset.MultiDataSet(new INDArray[]{idxs, mask, segmentIdxs}, new INDArray[]{labelArr});

        Map<String, INDArray> placeholderValues = new HashMap<>();
        placeholderValues.put("IteratorGetNext", idxs);
        placeholderValues.put("IteratorGetNext:1", mask);
        placeholderValues.put("IteratorGetNext:4", segmentIdxs);
        placeholderValues.put("label", labelArr);

        INDArray lossArr = sd.exec(placeholderValues, "loss").get("loss");
        assertTrue(lossArr.isScalar());
        double scoreBefore = lossArr.getDouble(0);
        for( int i=0; i<100; i++ ){
            sd.fit(mds);
        }

        lossArr = sd.exec(placeholderValues, "loss").get("loss");
        assertTrue(lossArr.isScalar());
        double scoreAfter = lossArr.getDouble(0);

        System.out.println("Score Before: " + scoreBefore);
        System.out.println("Score After: " + scoreAfter);
    }

    @Test @Ignore
    public void writeBertUI() throws Exception {
        //Test used to generate graph for visualization to work out appropriate subgraph structure to replace
        File f = new File("C:/Temp/TF_Graphs/mrpc_output/frozen/bert_mrpc_frozen.pb");
        int minibatchSize = 4;

        Map<String, TFImportOverride> m = new HashMap<>();
        m.put("IteratorGetNext", (inputs, controlDepInputs, nodeDef, initWith, attributesForNode, graph) -> {
            //Return 3 placeholders called "IteratorGetNext:0", "IteratorGetNext:1", "IteratorGetNext:3" instead of the training iterator
            return Arrays.asList(
                    initWith.placeHolder("IteratorGetNext", DataType.INT, minibatchSize, 128),
                    initWith.placeHolder("IteratorGetNext:1", DataType.INT, minibatchSize, 128),
                    initWith.placeHolder("IteratorGetNext:4", DataType.INT, minibatchSize, 128)
            );
        });

        //Skip the "IteratorV2" op - we don't want or need this
        TFOpImportFilter filter = (nodeDef, initWith, attributesForNode, graph) -> {
            return "IteratorV2".equals(nodeDef.getName());
        };

        SameDiff sd = TFGraphMapper.getInstance().importGraph(f, m, filter);

        LogFileWriter w = new LogFileWriter(new File("C:/Temp/BERT_UI.bin"));
        long bytesWritten = w.writeGraphStructure(sd);
        long bytesWritten2 = w.writeFinishStaticMarker();
    }

}
