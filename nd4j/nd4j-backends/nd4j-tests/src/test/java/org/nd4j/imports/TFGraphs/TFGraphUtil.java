package org.nd4j.imports.TFGraphs;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.apache.commons.io.FileUtils;
import org.nd4j.common.base.Preconditions;
import org.nd4j.common.resources.Resources;
import org.nd4j.common.tests.ResourceUtils;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.*;

@Slf4j
public class TFGraphUtil {

    private TFGraphUtil(){ }

    public static Map<String,TestCase> getTestCases(String baseDir) throws Exception {

        long start = System.currentTimeMillis();
//        String baseDir = "tf_graphs/examples/";
        List<String> l = ResourceUtils.listClassPathFiles(baseDir, true, false);
        long end = System.currentTimeMillis();

        Set<String> listAsSet = new HashSet<>(l);


        Set<String> modelSet = new HashSet<>();
        List<String> modelFileNames = new ArrayList<>();


        Map<String, TestCase> map = new HashMap<>();

        long start2 = System.currentTimeMillis();
        for(String s : l){
            String sub = s.substring(baseDir.length()+1);
//            int idx = sub.indexOf('/');
//            if(idx > 0){
//                String model
//            }

            int idx = sub.lastIndexOf('/');


            if(idx > 0) {
                String name = sub.substring(0, idx);
                String modelDir = baseDir + sub.substring(0,idx+1);
                String expModel = modelDir + TFGraphTestAllSameDiff.MODEL_FILENAME;
//                while(!Resources.exists(expModel) && idx > 0){
                while(!listAsSet.contains(expModel) && idx > 0){
                    //Due to a mixing of directories and variable names - we
                    //For example we might have "X/frozen_model.pb"
                    //And then also "X/something/or/other.csv
                    //When this occurs - we should look up the path to determine which part is the model name
                    // and which part is the variable name
                    idx = sub.lastIndexOf('/', idx);
                    if(idx < 0){
                        System.out.println("***** BAD TEST DIRECTORY: " + s + " ******");
                        continue;
                    }

                    sub = sub.substring(0, idx);
                    expModel = baseDir + "/" + sub + "/" + TFGraphTestAllSameDiff.MODEL_FILENAME;
                }


                modelSet.add(name);

                TestCase tc = map.get(name);
                if(tc == null){
                    tc = new TestCase(name, null, null, null);
                    map.put(name, tc);
                }

                if(s.endsWith("prediction.csv")){
                    if(tc.outputs == null)
                        tc.outputs = new HashMap<>();
                    String varName = s.substring(modelDir.length()).replaceAll("____", "/");
//                    String varName = sub.substring(idx+1).replaceAll("____", "/");
                    varName = varName.substring(0, varName.length() - "prediction.csv".length() - 1);
                    tc.outputs.put(varName, s);
                } else if(s.endsWith("placeholder.shape")){
                    if(tc.inputs == null)
                        tc.inputs = new HashMap<>();
//                    String varName = sub.substring(idx+1).replaceAll("____", "/");
                    String varName = s.substring(modelDir.length()).replaceAll("____", "/");
                    varName = varName.substring(0, varName.length() - "placeholder.shape".length() - 1);
                    tc.inputs.put(varName, s);
                } else if(s.endsWith("/dtypes")){
                    File f = Resources.asFile(s);
                    List<String> lines = FileUtils.readLines(f, StandardCharsets.UTF_8);
                    tc.datatypes = new HashMap<>();
                    for(String line : lines){
                        String[] split = line.split(" ");
                        Preconditions.checkState(split.length == 2, "Expected 2 entries in dtypes file, got %s", split.length);
                        String key = split[0].replaceAll("____", "/");
                        DataType value = ArrayOptionsHelper.dataType(split[1]);

                        // adding zero output duplicate (if it doesn't exist)
                        if (key.endsWith(".0")) {
                            val nkey = key.replaceAll("\\.0$","");
                            if (!tc.datatypes.containsKey(nkey)) {
                                tc.datatypes.put(nkey, value);
                            }
                        } else if (key.endsWith(":0")) {
                            val nkey = key.replaceAll(":0$","");
                            if (!tc.datatypes.containsKey(nkey)) {
                                tc.datatypes.put(nkey, value);
                            }
                        }

                        tc.datatypes.put(line, null);
                    }
                }
//                System.out.println(sub);
            }
        }
        long end2 = System.currentTimeMillis();

        System.out.println("List duration: " + (end-start));
        System.out.println("Process duration: " + (end2-start2));
        return map;
    }

    private static long parseLong(String line){
        line = line.trim();       //Handle whitespace
        if(line.matches("-?\\d+\\.0+")){
            //Annoyingly, some integer data is stored with redundant/unnecessary zeros - like "-7.0000000"
            return Long.parseLong(line.substring(0, line.indexOf('.')));
        } else {
            return Long.parseLong(line);
        }
    }

    private static double parseDouble(String line){
        line = line.trim();   //Handle whitespace - some lines are like "      -inf"
        if("nan".equalsIgnoreCase(line)){
            return Double.NaN;
        } else if("inf".equalsIgnoreCase(line)) {
            return Double.POSITIVE_INFINITY;
        } else if("-inf".equalsIgnoreCase(line)){
            return Double.NEGATIVE_INFINITY;
        } else {
            return Double.parseDouble(line);
        }
    }

    private static boolean parseBoolean(String line){
        line = line.trim();
        if(line.matches("1(\\.0*)?")){          //Booleans are ocassionally represented like 1.000000 or 0.000000
            return true;
        } else if(line.matches("0(\\.0*)?")){
            return false;
        }
        return Boolean.parseBoolean(line);
    }

    public static INDArray loadCsv(String path, TestCase tc) throws IOException {

        DataType type = tc.datatypes.get(path);

        String shapeFile = path.substring(0, path.length()-4) + ".shape";
        List<String> shapeLines = FileUtils.readLines(Resources.asFile(shapeFile), StandardCharsets.UTF_8);
        List<String> filteredShape = new ArrayList<>(shapeLines.size());
        for(String s : shapeLines){
            String trimmed = s.trim();
            if(!trimmed.isEmpty()){
                filteredShape.add(trimmed);
            }
        }

        if(type == null){
            log.warn("DATATYPE NOT AVAILABLE FOR: {} - {}", tc.modelName, path);
            //Soon: this will be an exception
            type = DataType.FLOAT;
        }

        INDArray varValue = null;
        if(filteredShape.size() == 0){
            //Scalar
            String content = FileUtils.readFileToString(Resources.asFile(path), StandardCharsets.UTF_8);    //IOUtils.toString(resources.get(i).getSecond().getInputStream(), StandardCharsets.UTF_8);
            switch (type){
                case DOUBLE:
                case FLOAT:
                case HALF:
                case BFLOAT16:
                    varValue = Nd4j.scalar(type, parseDouble(content));
                    break;
                case LONG:
                case INT:
                case SHORT:
                case UBYTE:
                case BYTE:
                case UINT16:
                case UINT32:
                case UINT64:
                    varValue = Nd4j.scalar(type, parseLong(content));
                    break;
                case BOOL:
                    varValue = Nd4j.scalar(parseBoolean(content));
                    break;
                case UTF8:
                    varValue = Nd4j.scalar(content);
                    break;
                case COMPRESSED:
                case UNKNOWN:
                default:
                    throw new UnsupportedOperationException("Unknown / not implemented datatype: " + type);
            }
        } else {
            int[] varShape = new int[filteredShape.size()];
            for( int j=0; j<filteredShape.size(); j++ ){
                varShape[j] = Integer.parseInt(filteredShape.get(j));
            }

            try {
//                String content;
//                Pair<Resource,Resource> p = resources.get(i);
//                boolean isRef = p.getSecond().isFile() && !p.getSecond().exists();
//
//                InputStream stream;
//                if(isRef){
//                    //Slight hack for loading strumpf reference files
//                    File r = new StrumpfResolver().localCacheRoot();
//                    String path = p.getSecond().getFile() + StrumpfResolver.REF;
//                    File f = ResourceFile.fromFile(path).localFile(r);
//                    stream = new BufferedInputStream(new FileInputStream(f));
//                } else {
//                    stream = new BufferedInputStream(resources.get(i).getSecond().getInputStream());
//                }
//
//                try(InputStream is = stream){
//                    content = String.join("\n", IOUtils.readLines(is, StandardCharsets.UTF_8));
//                }
                String content = FileUtils.readFileToString(Resources.asFile(path), StandardCharsets.UTF_8);

                if (content.isEmpty()) {
                    //Should be zeros in shape
                    boolean foundZero = false;
                    for( int s : varShape){
                        foundZero |= (s == 0);
                    }
                    if(foundZero){
                        varValue = Nd4j.create(type, ArrayUtil.toLongArray(varShape));
                    } else {
                        throw new IllegalStateException("Empty data but non-empty shape: " + shapeFile);
                    }
                } else {
                    if(varShape.length == 1 && varShape[0] == 0)        //Annoyingly, some scalars have shape [0] instead of []
                        varShape = new int[0];

                    String[] cLines = content.split("\n");
                    switch (type){
                        case DOUBLE:
                        case FLOAT:
                        case HALF:
                        case BFLOAT16:
                            double[] dArr = new double[cLines.length];
                            int x=0;
                            while(x < dArr.length){
                                dArr[x] = parseDouble(cLines[x]);
                                x++;
                            }
                            varValue = Nd4j.createFromArray(dArr).castTo(type).reshape('c', varShape);
                            break;
                        case LONG:
                        case INT:
                        case SHORT:
                        case UBYTE:
                        case BYTE:
                        case UINT16:
                        case UINT32:
                        case UINT64:
                            long[] lArr = new long[cLines.length];
                            int y=0;
                            while(y < lArr.length){
                                lArr[y] = parseLong(cLines[y]);
                                y++;
                            }
                            varValue = Nd4j.createFromArray(lArr).castTo(type).reshape('c', varShape);
                            break;
                        case BOOL:
                            boolean[] bArr = new boolean[cLines.length];
                            int z=0;
                            while(z < bArr.length){
                                bArr[z] = parseBoolean(cLines[z]);
                                z++;
                            }
                            varValue = Nd4j.createFromArray(bArr).reshape('c', varShape);
                            break;
                        case UTF8:
                            varValue = Nd4j.create(cLines).reshape('c', varShape);
                            break;
                        case COMPRESSED:
                        case UNKNOWN:
                        default:
                            throw new UnsupportedOperationException("Unknown / not implemented datatype: " + type);
                    }
                }
            } catch (NumberFormatException e) {
                log.warn("Error parsing number", e);
//                continue;
            }
        }

        return varValue;
    }

}
