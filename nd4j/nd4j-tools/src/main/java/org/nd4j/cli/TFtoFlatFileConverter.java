package org.nd4j.cli;

import lombok.val;
import org.nd4j.autodiff.execution.NativeGraphExecutioner;
import org.nd4j.autodiff.execution.conf.ExecutionMode;
import org.nd4j.autodiff.execution.conf.ExecutorConfiguration;
import org.nd4j.autodiff.execution.conf.OutputMode;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.transform.*;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.imports.tensorflow.TFImportOverride;
import org.nd4j.imports.tensorflow.TFOpImportFilter;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * Created by Yves Quemener on 12/7/18.
 */
public class TFtoFlatFileConverter {

    public static void convert(String inFile, String outFile) throws IOException {

        SameDiff tg = TFGraphMapper.getInstance().importGraph(new File(inFile));
        tg.asFlatFile(new File(outFile));
    }

    public static void convertBERT(String inFile, String outFile) throws IOException {

        /*
         * Define: Op import overrides for the BERT model. This is used to skip the IteratorGetNext node and instead crate some placeholders
         */
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

    public static void main(String [] args) throws IOException {
        if(args.length<2){
            System.err.println("Usage:\n" +
                    "mvn exec:java -Dexec.mainClass=\"org.nd4j.cli.TFtoFlatFileConverter\" -Dexec.args=\"<input_file.pb> <output_file.fb>\"\n");
            //mobileNetFlatBufferSanity();
            //mobileNetFlatBufferSanityLibNd4j();
            //testDenseNet();
            //testMobileNet();
            //testMobileNetOutput();
            //testTfFiles();
        }
        else{
            convert(args[0], args[1]);
            //convertBERT(args[0], args[1]);
        }
    }

/*    public static void testTfFiles() throws IOException {
        String inDir = "//home/yves/dl4j/models/";
        String outDir = "/home/yves/tmp/fbmodels/";
        String[] filenames = {"mobilenet_v1_0.5_128_frozen.pb",
                "mobilenet_v2_1.0_224_frozen.pb",
                "nasnet_mobile.pb",
                "resnetv2_imagenet_frozen_graph.pb",
                "squeezenet.pb",
                "densenet.pb"};

        for(int i=0;i<filenames.length;i++){
            System.out.println("Testing model "+filenames[i]);
            String outFilename = filenames[i].substring(0, filenames[i].length()-3) + ".fb";
            convert(inDir+filenames[i], outDir+outFilename);
        }
    }

    public static void testMobileNet() throws IOException {
        //convert("/home/yves/tmp/mobilenet_v1_0.5_128_frozen.pb", "/home/yves/tmp/mobilenet.fb");
    }

    public static void testDenseNet() throws IOException {
        //convert("/home/yves/tmp/densenet/densenet.pb", "/home/yves/tmp/densenet.fb");
    }

    public static void testMobileNetOutput(){
        SameDiff tg = TFGraphMapper.getInstance().importGraph(new File("/home/yves/dl4j/models/mobilenet_v1_0.5_128_frozen.pb"));
        INDArray array = Nd4j.ones(1,128,128,3);
        tg.associateArrayWithVariable(array, "input");
        //INDArray result = tg.execAndEndResult();
        INDArray result = tg.execSingle(Collections.singletonMap("input",array), tg.outputs().get(0));

        System.out.println("Result = "+result.toString());
        System.out.println("Result ind max = "+result.argMax().toString());
        System.out.println("Result arg max = "+result.get(result.argMax()).toString());
    }

    public static void mobileNetFlatBufferSanity() throws IOException {
        SameDiff tg = SameDiff.fromFlatFile(new File("/home/yves/dl4j/models/flatBufferModels/master_version/mobilenet_v1_0.5_128_frozen.fb"));
        INDArray array = Nd4j.zeros(1,128,128,3);
        tg.associateArrayWithVariable(array, "input");

        INDArray result = tg.execSingle(Collections.singletonMap("input",array), tg.outputs().get(0));


        System.out.println("Result = "+result.toString());
        System.out.println("Result ind max = "+result.argMax().toString());
        System.out.println("Result arg max = "+result.get(result.argMax()).toString());
    }

    public static void mobileNetFlatBufferSanityLibNd4j() throws IOException {
        //SameDiff tg = SameDiff.fromFlatFile(new File("/home/yves/dl4j/models/flatBufferModels/master_version/mobilenet_v1_0.5_128_frozen.fb"));
        //SameDiff tg = SameDiff.fromFlatFile(new File("/home/yves/tmp/fbmodels/mobilenet_v1_0.5_128_frozen.fb"));
        SameDiff tg = TFGraphMapper.getInstance().importGraph(new File("/home/yves/dl4j/models/mobilenet_v1_0.5_128_frozen.pb"));
        INDArray array = Nd4j.zeros(1,128,128,3);
        tg.associateArrayWithVariable(array, "input");

        //INDArray result = tg.execSingle(Collections.singletonMap("input",array), tg.outputs().get(0));

        val executioner = new NativeGraphExecutioner();
        ExecutorConfiguration configuration = ExecutorConfiguration.builder()
                .executionMode(ExecutionMode.SEQUENTIAL)
                .profilingMode(OpExecutioner.ProfilingMode.DISABLED)
                .gatherTimings(true)
                .outputMode(OutputMode.VARIABLE_SPACE)
                .build();


        executioner.executeGraph(tg, configuration);
        INDArray result = tg.getVariable("MobilenetV1/Predictions/Reshape_1").getArr();

        System.out.println("Result = "+result.toString());
        System.out.println("Result ind max = "+result.argMax().toString());
        System.out.println("Result arg max = "+result.get(result.argMax()).toString());
    }

*/
}
