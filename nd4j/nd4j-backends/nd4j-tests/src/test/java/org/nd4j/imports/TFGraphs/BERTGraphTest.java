package org.nd4j.imports.TFGraphs;

import org.junit.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.graph.ui.LogFileWriter;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.imports.tensorflow.TFImportOverride;
import org.nd4j.imports.tensorflow.TFOpImportFilter;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.File;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;

public class BERTGraphTest {

    @Test
    public void writeBertUI() throws Exception {
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

    @Test
    public void testBert(){
        /*
        Important node: BERT model uses a FIXED (hardcoded) minibatch size, not dynamic as most models use
         */
//        File f = new File("C:\\Temp\\TF_Graphs\\mrpc_output\\BERT_uncased_L-12_H-768_A-12_mrpc_frozen.pb");
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
        TFOpImportFilter filter = (nodeDef, initWith, attributesForNode, graph) -> { return "IteratorV2".equals(nodeDef.getName()); };

        SameDiff sd = TFGraphMapper.getInstance().importGraph(f, m, filter);

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

        INDArray idxs = Nd4j.create(DataType.INT, minibatchSize, 128);
        idxs.getRow(0).assign(ex1Idxs);
        idxs.getRow(1).assign(ex2Idxs);
        idxs.getRow(2).assign(ex3Idxs);
        idxs.getRow(3).assign(ex4Idxs);

        INDArray mask = Nd4j.create(DataType.INT, minibatchSize, 128);
        mask.getRow(0).assign(ex1Mask);
        mask.getRow(1).assign(ex2Mask);
        mask.getRow(2).assign(ex3Mask);
        mask.getRow(3).assign(ex4Mask);

        INDArray segmentIdxs = Nd4j.create(DataType.INT, minibatchSize, 128);
        segmentIdxs.getRow(0).assign(ex1SegmentId);
        segmentIdxs.getRow(1).assign(ex2SegmentId);
        segmentIdxs.getRow(2).assign(ex3SegmentId);
        segmentIdxs.getRow(3).assign(ex4SegmentId);

        Map<String, INDArray> placeholderValues = new HashMap<>();
        placeholderValues.put("IteratorGetNext", idxs);
        placeholderValues.put("IteratorGetNext:1", mask);
        placeholderValues.put("IteratorGetNext:4", segmentIdxs);

        Map<String, INDArray> out = sd.exec(placeholderValues, "loss/Softmax");
        INDArray softmax = out.get("loss/Softmax");
        System.out.println("OUTPUT - Softmax");
        System.out.println(softmax);

        INDArray exp0 = Nd4j.createFromArray(0.99860954f, 0.0013904407f).reshape(1, 2);
        INDArray exp1 = Nd4j.createFromArray(0.0005442508f, 0.99945575f).reshape(1, 2);
        INDArray exp2 = Nd4j.createFromArray(0.9987967f, 0.0012033002f).reshape(1, 2);
        INDArray exp3 = Nd4j.createFromArray(0.97409827f, 0.025901746f).reshape(1, 2);

        assertEquals(exp0, softmax.getRow(0));
        assertEquals(exp1, softmax.getRow(1));
        assertEquals(exp2, softmax.getRow(2));
        assertEquals(exp3, softmax.getRow(3));
    }

}
