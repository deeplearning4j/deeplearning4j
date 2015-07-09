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

package org.deeplearning4j.ui.nearestneighbors;

import io.dropwizard.views.View;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.URL;
import java.util.*;

import javax.ws.rs.GET;
import javax.ws.rs.POST;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.clustering.sptree.DataPoint;
import org.deeplearning4j.clustering.vptree.VPTree;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.ui.api.UrlResource;
import org.deeplearning4j.ui.uploads.FileResource;
import org.deeplearning4j.util.SerializationUtils;

/**
 * Nearest neighbors
 *
 * @author Adam Gibson
 */
@Path("/nearestneighbors")
public class NearestNeighborsResource extends FileResource {
    private VPTree tree;
    private List<VocabWord> words;
    private Map<Integer,VocabWord> theVocab;
    private VocabCache vocab;
    private WordVectors wordVectors;
    private File localFile;

    /**
     * The file path for uploads
     *y
     * @param filePath the file path for uploads
     */
    public NearestNeighborsResource(String filePath) {
        super(filePath);
    }

    @GET
    public View get() {
        return new NearestNeighborsView();
    }

    @POST
    @Path("/update")
    @Produces(MediaType.APPLICATION_JSON)
    public Response updateFilePath(UrlResource resource) {
        if(!resource.getUrl().startsWith("http")) {
            this.localFile = new File(".",resource.getUrl());
            handleUpload(localFile);
        }
        else {
            File dl = new File(filePath,UUID.randomUUID().toString());
            try {
                FileUtils.copyURLToFile(new URL(resource.getUrl()), dl);
            } catch (Exception e) {
                e.printStackTrace();
            }

            handleUpload(dl);

        }

        return Response.ok(Collections.singletonMap("message","Uploaded file")).build();
    }

    @POST
    @Path("/vocab")
    @Produces(MediaType.APPLICATION_JSON)
    public Response getVocab() {
        List<String> words = new ArrayList<>();

        if(wordVectors != null) {
            words.addAll(wordVectors.vocab().words());
        }
        else {
            for(VocabWord word : this.words) {
                words.add(word.getWord());
            }
        }

        return Response.ok((new ArrayList<>(words))).build();
    }

    @POST
    @Produces(MediaType.APPLICATION_JSON)
    @Path("/words")
    public Response getWords(NearestNeighborsQuery query) {
        Map<String,Double> map = new HashMap<>();

        if(wordVectors != null) {
            Collection<String> words = wordVectors.wordsNearest(query.getWord(),query.getNumWords());
            for(String word : words) {
                map.put(word,wordVectors.similarity(query.getWord(),word));
            }
        }
        else {
            List<DataPoint> results = new ArrayList<>();
            List<Double> distances = new ArrayList<>();
            tree.search(tree.getItems().get(vocab.indexOf(query.getWord())),query.getNumWords(),results,distances);
            for(int i = 0; i < results.size(); i++) {
                map.put(theVocab.get(results.get(i).getIndex()).getWord(),distances.get(i));
            }
        }


        return Response.ok(map).build();
    }


    @Override
    public void handleUpload(File path) {
        try {
            if(path.getAbsolutePath().endsWith(".ser")) {
                WordVectors vectors = SerializationUtils.readObject(path);
                InMemoryLookupTable table = (InMemoryLookupTable) vectors.lookupTable();
                tree = new VPTree(table.getSyn0(),"dot",true);
                words = new ArrayList<>(vectors.vocab().vocabWords());
                theVocab = new HashMap<>();

                for(VocabWord word : words) {
                    theVocab.put(word.getIndex(),word);
                }
                this.vocab = vectors.vocab();


            }
            else if(path.getAbsolutePath().contains("Google")) {
                WordVectors vectors = WordVectorSerializer.loadGoogleModel(path, true);
                this.wordVectors = vectors;
            }

            else {
                Pair<InMemoryLookupTable, VocabCache> vocab = WordVectorSerializer.loadTxt(path);
                this.wordVectors = WordVectorSerializer.fromPair(vocab);

            }


        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
