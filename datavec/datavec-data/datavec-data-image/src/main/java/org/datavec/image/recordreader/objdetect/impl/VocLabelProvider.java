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

import lombok.NonNull;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.datavec.image.recordreader.objdetect.ImageObject;
import org.datavec.image.recordreader.objdetect.ImageObjectLabelProvider;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.List;

/**
 * Label provider for object detection, for use with {@link org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader}.
 * This label provider reads the datasets from the the PASCAL Visual Object Classes - VOC2007 to VOC2012 datasets.<br>
 * The VOC datasets contain 20 classes and (for VOC2012) 17,125 images.<br>
 * <a href="http://host.robots.ox.ac.uk/pascal/VOC/voc2007/">http://host.robots.ox.ac.uk/pascal/VOC/voc2007/</a><br>
 * <a href="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/">http://host.robots.ox.ac.uk/pascal/VOC/voc2012/</a>
 * <br>
 * <br>
 * How to use:<br>
 * 1. Download and extract VOC dataset<br>
 * 2. Set baseDirectory to (for example) VOC2012 directory (should contain JPEGImages and Annotations directories)<br>
 *
 *
 * @author Alex Black
 */
public class VocLabelProvider implements ImageObjectLabelProvider {

    private static final String OBJECT_START_TAG = "<object>";
    private static final String OBJECT_END_TAG = "</object>";
    private static final String NAME_TAG = "<name>";
    private static final String XMIN_TAG = "<xmin>";
    private static final String YMIN_TAG = "<ymin>";
    private static final String XMAX_TAG = "<xmax>";
    private static final String YMAX_TAG = "<ymax>";

    private String annotationsDir;

    public VocLabelProvider(@NonNull String baseDirectory){
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

        //Normally we'd use Jackson to parse XML, but Jackson has real trouble with multiple XML elements with
        //  the same name. However, the structure is simple and we can parse it manually (even though it's not
        // the most elegant thing to do :)
        String[] lines = xmlContent.split("\n");

        List<ImageObject> out = new ArrayList<>();
        for( int i=0; i<lines.length; i++ ){
            if(!lines[i].contains(OBJECT_START_TAG)){
                continue;
            }
            String name = null;
            int xmin = Integer.MIN_VALUE;
            int ymin = Integer.MIN_VALUE;
            int xmax = Integer.MIN_VALUE;
            int ymax = Integer.MIN_VALUE;
            while(!lines[i].contains(OBJECT_END_TAG)){
                if(name == null && lines[i].contains(NAME_TAG)){
                    int idxStartName = lines[i].indexOf('>') + 1;
                    int idxEndName = lines[i].lastIndexOf('<');
                    name = lines[i].substring(idxStartName, idxEndName);
                    i++;
                    continue;
                }
                if(xmin == Integer.MIN_VALUE && lines[i].contains(XMIN_TAG)){
                    xmin = extractAndParse(lines[i]);
                    i++;
                    continue;
                }
                if(ymin == Integer.MIN_VALUE && lines[i].contains(YMIN_TAG)){
                    ymin = extractAndParse(lines[i]);
                    i++;
                    continue;
                }
                if(xmax == Integer.MIN_VALUE && lines[i].contains(XMAX_TAG)){
                    xmax = extractAndParse(lines[i]);
                    i++;
                    continue;
                }
                if(ymax == Integer.MIN_VALUE && lines[i].contains(YMAX_TAG)){
                    ymax = extractAndParse(lines[i]);
                    i++;
                    continue;
                }

                i++;
            }

            if(name == null){
                throw new IllegalStateException("Invalid object format: no name tag found for object in file " + xmlPath);
            }
            if(xmin == Integer.MIN_VALUE || ymin == Integer.MIN_VALUE || xmax == Integer.MIN_VALUE || ymax == Integer.MIN_VALUE){
                throw new IllegalStateException("Invalid object format: did not find all of xmin/ymin/xmax/ymax tags in " + xmlPath);
            }

            out.add(new ImageObject(xmin, ymin, xmax, ymax, name));
        }

        return out;
    }

    private int extractAndParse(String line){
        int idxStartName = line.indexOf('>') + 1;
        int idxEndName = line.lastIndexOf('<');
        String substring = line.substring(idxStartName, idxEndName);
        return Integer.parseInt(substring);
    }

    @Override
    public List<ImageObject> getImageObjectsForPath(URI uri) {
        return getImageObjectsForPath(uri.toString());
    }

}
