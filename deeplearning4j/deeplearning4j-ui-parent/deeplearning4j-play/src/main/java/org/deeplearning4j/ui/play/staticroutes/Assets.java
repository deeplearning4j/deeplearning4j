/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.ui.play.staticroutes;

import org.nd4j.shade.guava.net.HttpHeaders;
import lombok.AllArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FilenameUtils;
import org.nd4j.linalg.io.ClassPathResource;
import play.mvc.Result;
import play.mvc.StaticFileMimeTypes;

import java.io.InputStream;
import java.util.Optional;

import static play.mvc.Results.ok;

/**
 * Simple function for serving assets. Assets are assumed to be in the specified root directory
 *
 * @author Alex Black
 */
@AllArgsConstructor
@Slf4j
public class Assets {

    public static Result assetRequest(String assetsRootDirectory, String s) {

        String fullPath;
        if(s.startsWith("webjars/")){
            fullPath = "META-INF/resources/" + s;
        } else {
             fullPath = assetsRootDirectory + s;
        }

        InputStream inputStream;
        try {
            inputStream = new ClassPathResource(fullPath).getInputStream();
        } catch (Throwable t) {
            log.warn("Could not find requested UI asset: {}", s, t);
            return ok();
        }

        String fileName = FilenameUtils.getName(fullPath);

        Optional<String> contentType = StaticFileMimeTypes.fileMimeTypes().forFileName(fileName);
        String ct;
        ct = contentType.orElse("application/octet-stream");

        return ok(inputStream)
                .withHeader(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=\"" + fileName + "\"")
                .as(ct);
    }
}
