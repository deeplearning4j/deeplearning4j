/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.nd4j.imports.tensorflow;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.compress.archivers.ArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.apache.commons.io.input.CloseShieldInputStream;
import org.nd4j.base.Preconditions;
import org.nd4j.imports.converters.DifferentialFunctionClassHolder;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.util.ArchiveUtils;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.io.*;
import java.util.*;
import java.util.zip.GZIPInputStream;
import java.util.zip.ZipFile;

/**
 * A simple utility that analyzes TensorFlow graphs and reports details about the models:<br>
 * - The path of the model file(s)<br>
 * - The path of the model(s) that can't be imported due to missing ops<br>
 * - The path of model files that couldn't be read for some reason (corrupt file?)<br>
 * - The total number of ops in all graphs<br>
 * - The number of unique ops in all graphs<br>
 * - The (unique) names of all ops encountered in all graphs<br>
 * - The (unique) names of all ops that were encountered, and can be imported, in all graphs<br>
 * - The (unique) names of all ops that were encountered, and can NOT be imported (lacking import mapping)<br>
 *<br>
 * Note that an op is considered to be importable if has an import mapping specified for that op name in SameDiff.<br>
 * This alone does not guarantee that the op can be imported successfully.
 *
 * @author Alex Black
 */
@Slf4j
public class TensorFlowImportValidator {

    /**
     * Recursively scan the specified directory for .pb files, and evaluate which operations/graphs can/can't be imported
     * @param directory Directory to scan
     * @return Status for TensorFlow import for all models in
     * @throws IOException
     */
    public static TFImportStatus checkAllModelsForImport(@NonNull File directory) throws IOException {
        return checkModelForImport(directory, false);
    }

    public static TFImportStatus checkAllModelsForImport(@NonNull File directory, boolean includeArchives) throws IOException {

        List<String> fileExts = new ArrayList<>();
        fileExts.add("pb");
        if (includeArchives) {
            fileExts.addAll(Arrays.asList("zip", "tar.gz", "gzip", "tgz", "gz", "7z", "tar.bz2", "tar.gz2", "tar.lz", "tar.lzma", "tg", "tar"));
        }

        return checkAllModelsForImport(directory, fileExts.toArray(new String[fileExts.size()]));
    }

    public static TFImportStatus checkAllModelsForImport(File directory, String[] fileExtensions) throws IOException {
        Preconditions.checkState(directory.isDirectory(), "Specified directory %s is not actually a directory", directory);


        Collection<File> files = FileUtils.listFiles(directory, fileExtensions, true);
        Preconditions.checkState(!files.isEmpty(), "No model files found in directory %s", directory);

        TFImportStatus status = null;
        for(File f : files){
            if(isArchiveFile(f)){
                String p = f.getAbsolutePath();
                log.info("Checking archive file for .pb files: " + p);

                String ext = FilenameUtils.getExtension(p).toLowerCase();
                switch (ext){
                    case "zip":
                        List<String> filesInZip;
                        try {
                            filesInZip = ArchiveUtils.zipListFiles(f);
                        } catch (Throwable t){
                            log.warn("Unable to read from file, skipping: {}", f.getAbsolutePath(), t);
                            continue;
                        }
                        for(String s : filesInZip){
                            if(s.endsWith(".pb")){
                                try (ZipFile zf = new ZipFile(f); InputStream is = zf.getInputStream(zf.getEntry(s))){
                                    String p2 = p + "/" + s;
                                    log.info("Found possible frozen model (.pb) file in zip archive: {}", p2);
                                    TFImportStatus currStatus = checkModelForImport(p2,  is, false);
                                    if(currStatus.getCantImportModelPaths() != null && !currStatus.getCantImportModelPaths().isEmpty()){
                                        log.info("Unable to load - not a frozen model .pb file: {}", p2);
                                    } else {
                                        log.info("Found frozen model .pb file in archive: {}", p2);
                                    }
                                    status = (status == null ? currStatus : status.merge(currStatus));
                                }
                            }
                        }
                        break;
                    case "tar":
                    case "tar.gz":
                    case "tar.bz2":
                    case "tgz":
                    case "gz":
                    case "bz2":
                        if(p.endsWith(".tar.gz") || p.endsWith(".tgz") || p.endsWith(".tar") || p.endsWith(".tar.bz2")) {
                            boolean isTar = p.endsWith(".tar");
                            List<String> filesInTarGz;
                            try {
                                filesInTarGz = isTar ? ArchiveUtils.tarListFiles(f) : ArchiveUtils.tarGzListFiles(f);
                            } catch (Throwable t){
                                log.warn("Unable to read from file, skipping: {}", f.getAbsolutePath(), t);
                                continue;
                            }
                            for (String s : filesInTarGz) {
                                if (s.endsWith(".pb")) {
                                    TarArchiveInputStream is;
                                    if(p.endsWith(".tar")){
                                        is = new TarArchiveInputStream(new BufferedInputStream(new FileInputStream(f)));
                                    } else if(p.endsWith(".tar.gz") || p.endsWith(".tgz")){
                                        is = new TarArchiveInputStream(new GZIPInputStream(new BufferedInputStream(new FileInputStream(f))));
                                    } else if(p.endsWith(".tar.bz2")){
                                        is = new TarArchiveInputStream(new BZip2CompressorInputStream(new BufferedInputStream(new FileInputStream(f))));
                                    } else {
                                        throw new RuntimeException("Can't parse file type: " + s);
                                    }

                                    try {
                                        String p2 = p + "/" + s;
                                        log.info("Found possible frozen model (.pb) file in {} archive: {}", ext, p2);

                                        ArchiveEntry entry;
                                        boolean found = false;
                                        while((entry = is.getNextTarEntry()) != null){
                                            String name = entry.getName();
                                            if(s.equals(name)){
                                                //Found entry we want...
                                                TFImportStatus currStatus = checkModelForImport(p2, new CloseShieldInputStream(is), false);
                                                if(currStatus.getCantImportModelPaths() != null && !currStatus.getCantImportModelPaths().isEmpty()){
                                                    log.info("Unable to load - not a frozen model .pb file: {}", p2);
                                                } else {
                                                    log.info("Found frozen model .pb file in archive: {}", p2);
                                                }
                                                status = (status == null ? currStatus : status.merge(currStatus));
                                                found = true;
                                            }
                                        }
                                        Preconditions.checkState(found, "Could not find expected tar entry in file: " + p2);
                                    } finally {
                                        is.close();
                                    }
                                }
                            }
                            break;
                        }
                        //Fall through for .gz - FilenameUtils.getExtension("x.tar.gz") returns "gz" :/
                    case "gzip":
                        //Assume single file...
                        try(InputStream is = new GZIPInputStream(new BufferedInputStream(new FileInputStream(f)))){
                            try {
                                TFImportStatus currStatus = checkModelForImport(f.getAbsolutePath(), is, false);
                                status = (status == null ? currStatus : status.merge(currStatus));
                            } catch (Throwable t){
                                log.warn("Unable to read from file, skipping: {}", f.getAbsolutePath(), t);
                                continue;
                            }
                        }
                        break;
                    default:
                        throw new UnsupportedOperationException("Archive type not yet implemented: " + f.getAbsolutePath());
                }
            } else {
                log.info("Checking model file: " + f.getAbsolutePath());
                TFImportStatus currStatus = checkModelForImport(f);
                status = (status == null ? currStatus : status.merge(currStatus));
            }

            System.out.println("DONE FILE: " + f.getAbsolutePath() + " - totalOps = " + (status == null ? 0 : status.getOpNames().size())
                    + " - supported ops: " + (status == null ? 0 : status.getImportSupportedOpNames().size())
                    + " - unsupported ops: " + (status == null ? 0 : status.getUnsupportedOpNames().size())
            );
        }
        return status;
    }

    public static boolean isArchiveFile(File f){
        return !f.getPath().endsWith(".pb");
    }

    /**
     * See {@link #checkModelForImport(File)}. Defaults to exceptionOnRead = false
     */
    public static TFImportStatus checkModelForImport(@NonNull File file) throws IOException {
        return checkModelForImport(file, false);
    }

    /**
     * Check whether the TensorFlow frozen model (protobuf format) can be imported into SameDiff or not
     * @param file            Protobuf file
     * @param exceptionOnRead If true, and the file can't be read, throw an exception. If false, return an "empty" TFImportStatus
     * @return Status for importing the file
     * @throws IOException If error
     */
    public static TFImportStatus checkModelForImport(@NonNull File file, boolean exceptionOnRead) throws IOException {
        try (InputStream is = new FileInputStream(file)) {
            return checkModelForImport(file.getAbsolutePath(), is, exceptionOnRead);
        }
    }

    public static TFImportStatus checkModelForImport(String path, InputStream is, boolean exceptionOnRead) throws IOException {
        TFGraphMapper m = TFGraphMapper.getInstance();

        try {
            int opCount = 0;
            Set<String> opNames = new HashSet<>();

            try(InputStream bis = new BufferedInputStream(is)) {
                GraphDef graphDef = m.parseGraphFrom(bis);
                List<NodeDef> nodes = m.getNodeList(graphDef);

                if(nodes.isEmpty()){
                    throw new IllegalStateException("Error loading model for import - loaded graph def has no nodes (empty/corrupt file?): " + path);
                }

                for (NodeDef nd : nodes) {
                    if (m.isVariableNode(nd) || m.isPlaceHolderNode(nd))
                        continue;

                    String op = nd.getOp();
                    opNames.add(op);
                    opCount++;
                }
            }

            Set<String> importSupportedOpNames = new HashSet<>();
            Set<String> unsupportedOpNames = new HashSet<>();
            Map<String,Set<String>> unsupportedOpModel = new HashMap<>();

            for (String s : opNames) {
                if (DifferentialFunctionClassHolder.getInstance().getOpWithTensorflowName(s) != null) {
                    importSupportedOpNames.add(s);
                } else {
                    unsupportedOpNames.add(s);
                    if(unsupportedOpModel.containsKey(s)) {
                        continue;
                    } else {
                        Set<String> l = new HashSet<>();
                        l.add(path);
                        unsupportedOpModel.put(s, l);
                    }

                }
            }




            return new TFImportStatus(
                    Collections.singletonList(path),
                    unsupportedOpNames.size() > 0 ? Collections.singletonList(path) : Collections.<String>emptyList(),
                    Collections.<String>emptyList(),
                    opCount,
                    opNames.size(),
                    opNames,
                    importSupportedOpNames,
                    unsupportedOpNames,
                    unsupportedOpModel);
        } catch (Throwable t){
            if(exceptionOnRead) {
                throw new IOException("Error reading model from path " + path + " - not a TensorFlow frozen model in ProtoBuf format?", t);
            }
            log.warn("Failed to import model from: " + path + " - not a TensorFlow frozen model in ProtoBuf format?", t);
            return new TFImportStatus(
                    Collections.<String>emptyList(),
                    Collections.<String>emptyList(),
                    Collections.singletonList(path),
                    0,
                    0,
                    Collections.<String>emptySet(),
                    Collections.<String>emptySet(),
                    Collections.<String>emptySet(),
                    Collections.<String, Set<String>>emptyMap());
        }
    }
}
