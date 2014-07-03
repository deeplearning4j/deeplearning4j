package au.com.bytecode.opencsv.bean;
import java.util.HashMap;
import java.util.Map;

/**
 Copyright 2007,2010 Kyle Miller.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

public class HeaderColumnNameTranslateMappingStrategy<T> extends HeaderColumnNameMappingStrategy<T> {
    private Map<String, String> columnMapping = new HashMap<String, String>();
    protected String getColumnName(int col) {
        return col < header.length ? columnMapping.get(header[col].toUpperCase()) : null;
    }
    public Map<String, String> getColumnMapping() {
        return columnMapping;
    }
    public void setColumnMapping(Map<String, String> columnMapping) {
       for (String key : columnMapping.keySet()) {
          this.columnMapping.put(key.toUpperCase(), columnMapping.get(key));
       }
    }
}
