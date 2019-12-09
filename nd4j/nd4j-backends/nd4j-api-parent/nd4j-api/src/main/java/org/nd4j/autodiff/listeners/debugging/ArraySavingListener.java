package org.nd4j.autodiff.listeners.debugging;

import lombok.NonNull;
import org.nd4j.autodiff.listeners.At;
import org.nd4j.autodiff.listeners.BaseListener;
import org.nd4j.autodiff.listeners.Operation;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ArraySavingListener extends BaseListener {

    protected final File dir;
    protected int count = 0;

    public ArraySavingListener(@NonNull File dir){

        if(!dir.exists()){
            dir.mkdir();
        }

        if(dir.listFiles() != null && dir.listFiles().length > 0){
            throw new IllegalStateException("Directory is not empty: " + dir.getAbsolutePath());
        }

        this.dir = dir;
    }

    @Override
    public boolean isActive(Operation operation) {
        return true;
    }


    @Override
    public void opExecution(SameDiff sd, At at, MultiDataSet batch, SameDiffOp op, INDArray[] outputs) {
        List<String> outNames = op.getOutputsOfOp();
        for(int i=0; i<outputs.length; i++ ){
            String filename = (count++) + "_" + outNames.get(i).replaceAll("/", "__") + ".bin";
            File outFile = new File(dir, filename);

            INDArray arr = outputs[i];
            try {
                Nd4j.saveBinary(arr, outFile);
                System.out.println(outFile.getAbsolutePath());
            } catch (IOException e){
                throw new RuntimeException(e);
            }
        }
    }

    public static void compare(File dir1, File dir2, double eps) throws Exception {
        File[] files1 = dir1.listFiles();
        File[] files2 = dir2.listFiles();
        Preconditions.checkNotNull(files1, "No files in directory 1: %s", dir1);
        Preconditions.checkNotNull(files2, "No files in directory 2: %s", dir2);
        Preconditions.checkState(files1.length == files2.length, "Different number of files: %s vs %s", files1.length, files2.length);

        Map<String,File> m1 = toMap(files1);
        Map<String,File> m2 = toMap(files2);

        for(File f : files1){
            String name = f.getName();
            String varName = name.substring(name.indexOf('_') + 1, name.length()-4); //Strip "x_" and ".bin"
            File f2 = m2.get(varName);

            INDArray arr1 = Nd4j.readBinary(f);
            INDArray arr2 = Nd4j.readBinary(f2);

            //TODO String arrays won't work here!
            boolean eq = arr1.equalsWithEps(arr2, eps);
            if(eq){
                System.out.println("Equals: " + varName.replaceAll("__", "/"));
            } else {
                INDArray sub = arr1.sub(arr2);
                INDArray diff = Nd4j.math.abs(sub);
                double maxDiff = diff.maxNumber().doubleValue();
                System.out.println("FAILS: " + varName.replaceAll("__", "/") + " - max difference = " + maxDiff);
                System.out.println("\t" + f.getAbsolutePath());
                System.out.println("\t" + f2.getAbsolutePath());
                sub.close();
                diff.close();;
            }
            arr1.close();
            arr2.close();
        }
    }

    private static Map<String,File> toMap(File[] files){
        Map<String,File> ret = new HashMap<>();
        for(File f : files) {
            String  name = f.getName();
            String varName = name.substring(name.indexOf('_') + 1, name.length() - 4); //Strip "x_" and ".bin"
            ret.put(varName, f);
        }
        return ret;
    }
}
