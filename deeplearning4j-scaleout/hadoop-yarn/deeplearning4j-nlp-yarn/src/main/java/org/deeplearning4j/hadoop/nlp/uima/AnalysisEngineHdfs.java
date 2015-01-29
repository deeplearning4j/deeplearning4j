package org.deeplearning4j.hadoop.nlp.uima;

import org.apache.uima.analysis_engine.AnalysisEngineDescription;
import static org.apache.uima.fit.factory.AnalysisEngineFactory.*;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.uima.analysis_engine.AnalysisEngine;

import java.io.File;
import java.io.OutputStream;

/**
 * Create an analysis engine from an hdfs path
 * @author Adam Gibson
 */
public class AnalysisEngineHdfs {
   private AnalysisEngineHdfs() {}

    /**
     * Write the given analysis engine to the specified path
     * @param fs the file system to write to
     * @param to the path to write
     * @param toWrite the analysis engine to write
     * @throws Exception
     */
    public static void writeAnalysisEngineDescriptor(FileSystem fs,Path to,AnalysisEngineDescription toWrite) throws Exception {
        OutputStream create = fs.create(to,true);
        toWrite.toXML(create);
        create.flush();
        create.close();


    }

    /**
     * Reads the configuration from the specified location
     * @param from the file system to read from
     * @param to the analysis engine descriptor
     * @param extraArgs any extra arguments to specify
     * @return the created analysis engine
     * @throws Exception if one occurs
     */
    public static AnalysisEngine readConfFrom(FileSystem from,Path to,Object...extraArgs) throws Exception {
        File local = new File(to.getName());
        from.copyToLocalFile(false,to,new Path(local.getPath()),true);
        AnalysisEngineDescription desc = createEngineDescriptionFromPath(local.getAbsolutePath(), extraArgs);
        AnalysisEngine ret = createEngine(desc);
        local.delete();
        return ret;

    }

}
