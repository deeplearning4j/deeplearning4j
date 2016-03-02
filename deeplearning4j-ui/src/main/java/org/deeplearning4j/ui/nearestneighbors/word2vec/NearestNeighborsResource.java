/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.ui.nearestneighbors.word2vec;

import io.dropwizard.views.View;
import org.apache.commons.collections.map.HashedMap;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.clustering.sptree.DataPoint;
import org.deeplearning4j.clustering.vptree.VPTree;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.reader.impl.BasicModelUtils;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.ui.storage.SessionStorage;
import org.deeplearning4j.ui.storage.def.ObjectType;
import org.deeplearning4j.ui.uploads.FileResource;
import org.deeplearning4j.util.SerializationUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.ws.rs.GET;
import javax.ws.rs.POST;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import java.io.File;
import java.util.*;

/**
 * Nearest neighbors
 *
 * @author Adam Gibson
 */
@Path("/word2vec")
public class NearestNeighborsResource extends FileResource {
    private SessionStorage storage = SessionStorage.getInstance();

    private static final Logger logger = LoggerFactory.getLogger(NearestNeighborsResource.class);
     /**
     * The file path for uploads
     *y
     * @param filePath the file path for uploads
     */
    public NearestNeighborsResource(String filePath) {
        super(filePath);
    }

    public NearestNeighborsResource() {

    }

    @POST
    @Path("/vocab")
    @Produces(MediaType.APPLICATION_JSON)
    public Response getVocab() {
        WordVectors vectors = (WordVectors) storage.getObject("UploadedFile", ObjectType.WORD2VEC);
        if (vectors == null) return Response.noContent().build();

        List<String> words = new ArrayList<>();
        VocabCache<VocabWord> vocabCache = (VocabCache<VocabWord> ) vectors.vocab();
        for(VocabWord word : vocabCache.vocabWords())
            words.add(word.getWord());
        return Response.ok((new ArrayList<>(words))).build();
    }

    @POST
    @Produces(MediaType.APPLICATION_JSON)
    @Path("/words")
    public Response getWords(NearestNeighborsQuery query) {
        WordVectors vectors = (WordVectors) storage.getObject("UploadedFile", ObjectType.WORD2VEC);
        if (vectors == null) return Response.noContent().build();

        Collection<String> nearestNeighors = vectors.wordsNearest(query.getWord(),query.getNumWords());
        Map<String,Double> map = new LinkedHashMap<>();
        for(String s : nearestNeighors) {
            double sim = vectors.similarity(query.getWord(), s);
            map.put(s, sim);
        }
        return Response.ok(map).build();
    }


    @Override
    public void handleUpload(File path) {
        logger.info("handleUpload: " + path.getAbsolutePath());
        WordVectors vectors;
        try {
            if(path.getAbsolutePath().endsWith(".ser"))
                 vectors = SerializationUtils.readObject(path);
            else {
                vectors = WordVectorSerializer.fromPair(WordVectorSerializer.loadTxt(path));
            }
            vectors.setModelUtils(new BasicModelUtils());

            storage.putObject("UploadedFile", ObjectType.WORD2VEC, vectors);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
