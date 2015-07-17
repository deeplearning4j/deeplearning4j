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

package org.deeplearning4j.ui.tsne;

import io.dropwizard.views.View;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.ui.uploads.FileResource;

import javax.ws.rs.GET;
import javax.ws.rs.POST;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Tsne renders
 *
 * @author Adam Gibson
 */
@Path("/tsne")
@Produces(MediaType.TEXT_HTML)
public class TsneResource extends FileResource {
    private List<VocabWord> words;
    /**
     * The file path for uploads
     *
     * @param filePath the file path for uploads
     */
    public TsneResource(String filePath) {
        super(filePath);
    }

    @GET
    public View get() {
        return new TsneView();
    }

    @POST
    @Path("/vocab")
    @Produces(MediaType.APPLICATION_JSON)
    public Response getVocab() {
        List<String> words = new ArrayList<>();
        for(VocabWord word : this.words)
            words.add(word.getWord());
        return Response.ok((new ArrayList<>(words))).build();
    }

    @Override
    public void handleUpload(File path) {

    }
}
