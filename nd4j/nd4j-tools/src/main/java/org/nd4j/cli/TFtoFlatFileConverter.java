package org.nd4j.cli;

import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.transform.*;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.imports.tensorflow.TFImportOverride;
import org.nd4j.imports.tensorflow.TFOpImportFilter;
import org.nd4j.linalg.api.buffer.DataType;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * Created by Yves Quemener on 12/7/18.
 */
public class TFtoFlatFileConverter {

    public static void convert(String inFile, String outFile) throws IOException, org.nd4j.linalg.exception.ND4JIllegalStateException {
        SameDiff tg = TFGraphMapper.getInstance().importGraph(new File(inFile));
        tg.asFlatFile(new File(outFile));
    }

    public static void convertBERT(String inFile, String outFile) throws IOException, org.nd4j.linalg.exception.ND4JIllegalStateException  {
        //
        // Working around some issues in the BERT model's execution. See file:
        // nd4j/nd4j-backends/nd4j-tests/src/test/java/org/nd4j/imports/TFGraphs/BERTGraphTest.java
        // for details.

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
        TFOpImportFilter filter = (nodeDef, initWith, attributesForNode, graph) -> { return "IteratorV2".equals(nodeDef.getName()); };


        SameDiff sd = TFGraphMapper.getInstance().importGraph(new File(inFile), m, filter);


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


        System.out.println("Exporting file "+outFile);
        sd.asFlatFile(new File(outFile));
    }


    @Test @Ignore
    public void convertFile(){
        // Change these variables to convert the file you want
        String inputFilename = "/home/user/dl4j/models/densenet.pb";
        String outputFilename = "/home/user/dl4j/models/densenet.fb";
        // BERT Models require a specific pre-processing
        Boolean isBertModel = false;

        try {
            if(isBertModel){
                convertBERT(inputFilename, outputFilename);
            }
            else {
                convert(inputFilename, outputFilename);
            }
        }
        catch (IOException e){
            Assert.fail(e.toString());
        }
        catch (org.nd4j.linalg.exception.ND4JIllegalStateException e){
            Assert.fail(e.toString());
        }
    }

    public static void main(String [] args) throws IOException {
        if(args.length<2){
            System.err.println("Usage:\n" +
                    "mvn exec:java -Dexec.mainClass=\"org.nd4j.cli.TFtoFlatFileConverter\" -Dexec.args=\"<input_file.pb> <output_file.fb>\"\n");
        }
        else{
            convert(args[0], args[1]);
        }
    }

}
