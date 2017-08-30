/*
 *  * Copyright 2017 Skymind, Inc.
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
 */

package org.datavec.image.recordreader.objdetect.impl;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.NonNull;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.apache.commons.io.IOUtils;
import org.datavec.image.recordreader.objdetect.ImageObject;
import org.datavec.image.recordreader.objdetect.ImageObjectLabelProvider;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.shade.jackson.annotation.JsonProperty;
import org.nd4j.shade.jackson.core.JsonParser;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.databind.*;
import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;
import org.nd4j.shade.jackson.dataformat.xml.XmlFactory;
import org.nd4j.shade.jackson.dataformat.xml.annotation.JacksonXmlElementWrapper;
import org.nd4j.shade.jackson.dataformat.xml.annotation.JacksonXmlProperty;
import org.nd4j.shade.jackson.dataformat.xml.annotation.JacksonXmlRootElement;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * Label provider, for use with {@link org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader}, for use
 * with the VOC2007 to 2012 datasets.<br>
 * <a href="http://host.robots.ox.ac.uk/pascal/VOC/voc2007/">http://host.robots.ox.ac.uk/pascal/VOC/voc2007/</a><br>
 * <a href="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/">http://host.robots.ox.ac.uk/pascal/VOC/voc2012/</a>
 *
 * @author Alex Black
 */
public class VocLabelProvider implements ImageObjectLabelProvider {

    private String baseDirectory;
    private String annotationsDir;

    private ThreadLocal<ObjectMapper> mapper = new ThreadLocal<>();

    public VocLabelProvider(@NonNull String baseDirectory){
        this.baseDirectory = baseDirectory;
        this.annotationsDir = FilenameUtils.concat(baseDirectory, "Annotations");

        if(!new File(annotationsDir).exists()){
            throw new IllegalStateException("Annotations directory does not exist. Annotation files should be " +
                    "present at baseDirectory/Annotations/nnnnnn.xml. Expected location: " + annotationsDir);
        }
    }

    @Override
    public List<ImageObject> getImageObjectsForPath(String path) {
        int idx = path.lastIndexOf('/');
        idx = Math.max(idx, path.lastIndexOf('\\'));

        String filename = path.substring(idx+1, path.length()-4);   //-4: ".jpg"
        String xmlPath = FilenameUtils.concat(annotationsDir, filename + ".xml");
        File xmlFile = new File(xmlPath);
        if(!xmlFile.exists()){
            throw new IllegalStateException("Could not find XML file for image " + path + "; expected at " + xmlPath);
        }

        String xmlContent;
        try{
            xmlContent = FileUtils.readFileToString(xmlFile);
        } catch (IOException e){
            throw new RuntimeException(e);
        }

        //Parse XML, extract labels
        ObjectMapper m = mapper.get();
        if(m == null){
//            XmlFactory f = new XmlFactory();

            m = new ObjectMapper(new XmlFactory());
            m.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
            m.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
            mapper.set(m);
        }

//        JsonNode node;
//        try {
//            node = m.readTree(xmlContent);
//        } catch (IOException e){
//            //Should never happen
//            throw new RuntimeException(e);
//        }
//
//        Iterator<String> iter = node.fieldNames();
//        while(iter.hasNext()){
//            System.out.println(iter.next());
//        }
//
//        System.out.println("----------");
//        Iterator<Map.Entry<String,JsonNode>> iter2 = node.fields();
//        while(iter2.hasNext()){
//            Map.Entry<String,JsonNode> me = iter2.next();
//            System.out.println(me.getKey() + "\t" + me.getValue().getNodeType());
//        }
//        System.out.println(node.size());

        return null;
    }

    @Override
    public List<ImageObject> getImageObjectsForPath(URI uri) {
        return getImageObjectsForPath(uri.toString());
    }

}
