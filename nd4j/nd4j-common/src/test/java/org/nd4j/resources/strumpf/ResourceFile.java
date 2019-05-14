package org.nd4j.resources.strumpf;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.apache.commons.io.FileUtils;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.databind.DeserializationFeature;
import org.nd4j.shade.jackson.databind.MapperFeature;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.SerializationFeature;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Map;

@AllArgsConstructor
@NoArgsConstructor
@Data
@JsonIgnoreProperties("filePath")
public class ResourceFile {

    public static final ObjectMapper MAPPER = newMapper();

    //Note: Field naming to match Strumpf JSON format
    protected int current_version;
    protected Map<String,String> v1;

    //Not in JSON:
    protected String filePath;

    public static ResourceFile fromFile(String path){
        return fromFile(new File(path));
    }

    public static ResourceFile fromFile(File file){
        String s;
        try {
            s = FileUtils.readFileToString(file, StandardCharsets.UTF_8);
            ResourceFile rf = MAPPER.readValue(s, ResourceFile.class);
            rf.setFilePath(file.getPath());
            return rf;
        } catch (IOException e){
            throw new RuntimeException(e);
        }
    }

    public String relativePath(){
        String hashKey = null;
        for(String key : v1.keySet()){
            if(key.endsWith("_hash")){
                hashKey = key;
                break;
            }
        }
        if(hashKey == null){
            throw new IllegalStateException("Could not find <filename>_hash in resource reference file: " + filePath);
        }

        String relativePath = hashKey.substring(0, hashKey.length()-5); //-5 to remove "_hash" suffix
        return relativePath.replaceAll("\\\\", "/");
    }

    public boolean localFileExistsAndValid(File cacheRootDir){

        File file = getLocalFile(cacheRootDir);
        if(!file.exists()){
            return false;
        }

        //File exists... but is it valid?

    }

    /**
     * Get the local file - or where it *would* be if it has been downloaded. If it does not exist, it will not be downloaded here
     * @return
     */
    protected File getLocalFile(File cacheRootDir){
        String relativePath = relativePath();

        //For resolving local files with different versions, we want paths like:
        // ".../dir/filename.txt__v1/filename.txt"
        // ".../dir/filename.txt__v2/filename.txt"
        //This is to support multiple versions of files simultaneously... for example, different projects needing different
        // versions, or supporting old versions of resource files etc

        int lastSlash = Math.max(relativePath.lastIndexOf('/'), relativePath.lastIndexOf('\\'));
        String filename;
        if(lastSlash < 0){
            filename = relativePath;
        } else {
            filename = relativePath.substring(lastSlash+1);
        }

        File parentDir = new File(cacheRootDir, relativePath + "__v" + current_version);
        File file = new File(parentDir, filename);
        return file;
    }

    /**
     * Get the local file - downloading and caching if required
     * @return
     */
    public File localFile( File cacheRootDir ){
        if(localFileExistsAndValid(cacheRootDir)){
            return getLocalFile(cacheRootDir);
        }

        
    }


    public static final ObjectMapper newMapper(){
        ObjectMapper ret = new ObjectMapper();
        ret.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        ret.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
        ret.configure(MapperFeature.SORT_PROPERTIES_ALPHABETICALLY, true);
        ret.enable(SerializationFeature.INDENT_OUTPUT);
        return ret;
    }
}
