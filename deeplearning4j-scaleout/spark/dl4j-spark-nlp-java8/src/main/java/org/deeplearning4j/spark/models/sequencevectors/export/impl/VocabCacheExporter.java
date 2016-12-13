package org.deeplearning4j.spark.models.sequencevectors.export.impl;

import lombok.Getter;
import lombok.NonNull;
import org.apache.spark.api.java.JavaRDD;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.spark.models.sequencevectors.export.ExportContainer;

import java.io.File;

/**
 * This model exporter is suitable for debug/testing only.
 *
 * PLEASE NOTE: Never use this exporter in real distributed environment!
 *
 * @author raver119@gmail.com
 */
public class VocabCacheExporter extends HdfsModelExporter<VocabWord> {

    @Getter protected VocabCache<VocabWord> vocabCache;
    @Getter protected WeightLookupTable<VocabWord> lookupTable;

    public VocabCacheExporter() {
        try {
            File tempFile = File.createTempFile("temp", "file");
            tempFile.deleteOnExit();

            this.path = tempFile.getAbsolutePath();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public VocabCacheExporter(@NonNull String path) {
        super(path);
    }


    @Override
    public void export(JavaRDD<ExportContainer<VocabWord>> rdd) {
        super.export(rdd);

        // now we just load it back
        Word2Vec word2Vec = WordVectorSerializer.readWord2VecModel(path);

        // this is bad & dirty, but we don't really need anything else for testing
        vocabCache = word2Vec.getVocab();
        lookupTable = word2Vec.getLookupTable();
    }
}
