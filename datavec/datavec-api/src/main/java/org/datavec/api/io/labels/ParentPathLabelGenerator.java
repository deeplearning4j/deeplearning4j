/*-
 *  * Copyright 2016 Skymind, Inc.
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
package org.datavec.api.io.labels;

import org.apache.commons.io.FilenameUtils;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;

import java.io.File;
import java.net.URI;

/**
 * Returns as label the base name of the parent file of the path (the directory).
 *
 * @author saudet
 */
public class ParentPathLabelGenerator implements PathLabelGenerator {

    public ParentPathLabelGenerator() {}

    @Override
    public Writable getLabelForPath(String path) {
        // Label is in the directory
        String dirName = FilenameUtils.getName(new File(path).getParent());
        return new Text(dirName);
    }

    @Override
    public Writable getLabelForPath(URI uri) {
        return getLabelForPath(new File(uri).toString());
    }

    @Override
    public boolean inferLabelClasses() {
        return true;
    }
}
