/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.python4j;


import java.io.Closeable;
import java.util.HashSet;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;

public class PythonContextManager {

    private static Set<String> contexts = new HashSet<>();
    private static AtomicBoolean init = new AtomicBoolean(false);
    private static String currentContext;
    private static final String MAIN_CONTEXT = "main";
    private static final String COLLAPSED_KEY = "__collapsed__";

    static {
        init();
    }


    public static class Context implements Closeable{
        private final String name;
        private  final String previous;
        private final boolean temp;
        public Context(){
            name = "temp_" + UUID.randomUUID().toString().replace("-", "_");
            temp = true;
            previous = getCurrentContext();
            setContext(name);
        }
        public Context(String name){
           this.name = name;
           temp = false;
            previous = getCurrentContext();
            setContext(name);
        }

        @Override
        public void close(){
            setContext(previous);
            if (temp) deleteContext(name);
        }
    }

    private static void init() {
        if (init.get()) return;
        new PythonExecutioner();
        init.set(true);
        currentContext = MAIN_CONTEXT;
        contexts.add(currentContext);
    }


    /**
     * Adds a new context.
     * @param contextName
     */
    public static void addContext(String contextName) {
        if (!validateContextName(contextName)) {
            throw new PythonException("Invalid context name: " + contextName);
        }
        contexts.add(contextName);
    }

    /**
     * Returns true if context exists, else false.
     * @param contextName
     * @return
     */
    public static boolean hasContext(String contextName) {
        return contexts.contains(contextName);
    }

    private static boolean validateContextName(String s) {
        for (int i=0; i<s.length(); i++){
            char c = s.toLowerCase().charAt(i);
            if (i == 0){
                if (c >= '0' && c <= '9'){
                    return false;
                }
            }
            if (!(c=='_' || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9'))){
                return false;
            }
        }
        return true;
    }

    private static String getContextPrefix(String contextName) {
        return COLLAPSED_KEY + contextName + "__";
    }

    private static String getCollapsedVarNameForContext(String varName, String contextName) {
        return getContextPrefix(contextName) + varName;
    }

    private static String expandCollapsedVarName(String varName, String contextName) {
        String prefix = COLLAPSED_KEY + contextName + "__";
        return varName.substring(prefix.length());

    }

    private static void collapseContext(String contextName) {
        try (PythonGC _ = PythonGC.watch()) {
            PythonObject globals = Python.globals();
            PythonObject pop = globals.attr("pop");
            PythonObject keysF = globals.attr("keys");
            PythonObject keys = keysF.call();
            PythonObject keysList = Python.list(keys);
            int numKeys = Python.len(keysList).toInt();
            for (int i = 0; i < numKeys; i++) {
                PythonObject key = keysList.get(i);
                String keyStr = key.toString();
                if (!((keyStr.startsWith("__") && keyStr.endsWith("__")) || keyStr.startsWith("__collapsed_"))) {
                    String collapsedKey = getCollapsedVarNameForContext(keyStr, contextName);
                    PythonObject val = pop.call(key);

                    PythonObject pyNewKey = new PythonObject(collapsedKey);
                    globals.set(pyNewKey, val);
                }
            }
        } catch (Exception pe) {
            throw new RuntimeException(pe);
        }
    }

    private static void expandContext(String contextName) {
        try (PythonGC _ = PythonGC.watch()) {
            String prefix = getContextPrefix(contextName);
            PythonObject globals = Python.globals();
            PythonObject pop = globals.attr("pop");
            PythonObject keysF = globals.attr("keys");

            PythonObject keys = keysF.call();

            PythonObject keysList = Python.list(keys);
            try (PythonGC __ = PythonGC.pause()) {
                int numKeys = Python.len(keysList).toInt();

                for (int i = 0; i < numKeys; i++) {
                    PythonObject key = keysList.get(i);
                    String keyStr = key.toString();
                    if (keyStr.startsWith(prefix)) {
                        String expandedKey = expandCollapsedVarName(keyStr, contextName);
                        PythonObject val = pop.call(key);
                        PythonObject newKey = new PythonObject(expandedKey);
                        globals.set(newKey, val);
                    }
                }
            }
        }
    }


    /**
     * Activates the specified context
     * @param contextName
     */
    public static void setContext(String contextName) {
        if (contextName.equals(currentContext)) {
            return;
        }
        if (!hasContext(contextName)) {
            addContext(contextName);
        }


        collapseContext(currentContext);

        expandContext(contextName);
        currentContext = contextName;

    }

    /**
     * Activates the main context
     */
    public static void setMainContext() {
        setContext(MAIN_CONTEXT);

    }

    /**
     * Returns the current context's name.
     * @return
     */
    public static String getCurrentContext() {
        return currentContext;
    }

    /**
     * Resets the current context.
     */
    public static void reset() {
        String tempContext = "___temp__context___";
        String currContext = currentContext;
        setContext(tempContext);
        deleteContext(currContext);
        setContext(currContext);
        deleteContext(tempContext);
    }

    /**
     * Deletes the specified context.
     * @param contextName
     */
    public static void deleteContext(String contextName) {
        if (contextName.equals(currentContext)) {
            throw new PythonException("Cannot delete current context!");
        }
        if (!contexts.contains(contextName)) {
            return;
        }
        String prefix = getContextPrefix(contextName);
        PythonObject globals = Python.globals();
        PythonObject keysList = Python.list(globals.attr("keys").call());
        int numKeys = Python.len(keysList).toInt();
        for (int i = 0; i < numKeys; i++) {
            PythonObject key = keysList.get(i);
            String keyStr = key.toString();
            if (keyStr.startsWith(prefix)) {
                globals.attr("__delitem__").call(key);
            }
        }
        contexts.remove(contextName);
    }

    /**
     * Deletes all contexts except the main context.
     */
    public static void deleteNonMainContexts() {
        setContext(MAIN_CONTEXT); // will never fail
        for (String c : contexts.toArray(new String[0])) {
            if (!c.equals(MAIN_CONTEXT)) {
                deleteContext(c); // will never fail
            }
        }

    }

    /**
     * Returns the names of all contexts.
     * @return
     */
    public String[] getContexts() {
        return contexts.toArray(new String[0]);
    }

}
