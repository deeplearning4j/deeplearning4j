/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// Created for robust data type validation and error reporting
//

#ifndef ND4J_DATATYPE_VALIDATION_H
#define ND4J_DATATYPE_VALIDATION_H

#include <array/DataType.h>
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <system/common.h>

namespace sd {
    
    /**
     * Comprehensive data type validation and error reporting system
     * 
     * This class provides robust validation for data types, distinguishing between:
     * - Invalid data type IDs (not in enum)
     * - Valid but uncompiled data types (selective builds)
     * - Runtime vs compile-time mismatches
     */
    class SD_LIB_EXPORT DataTypeValidation {
    private:
        static std::unordered_set<DataType> compiledTypes_;
        static std::unordered_map<DataType, std::string> typeNames_;
        static std::unordered_map<int, DataType> enumMapping_;
        static bool initialized_;
        
        static void initializeIfNeeded();
        static void populateTypeNames();
        static void populateCompiledTypes();
        
    public:
        /**
         * Initialize the validation system (called automatically)
         */
        static void initialize();
        
        /**
         * Check if a data type enum value is valid
         * @param dataType The data type to check
         * @return true if the data type exists in the enum
         */
        static bool isValidDataType(DataType dataType);
        
        /**
         * Check if a numeric data type ID is valid
         * @param dataTypeId The numeric data type ID to check
         * @return true if the ID maps to a valid enum value
         */
        static bool isValidDataType(int dataTypeId);
        
        /**
         * Check if a data type is compiled into this build
         * @param dataType The data type to check
         * @return true if the type is available in this binary
         */
        static bool isCompiledDataType(DataType dataType);
        

        
        /**
         * Get a descriptive error message for numeric data type ID
         * @param dataTypeId The problematic data type ID
         * @param context Context string (function name, etc.)
         * @return Detailed error message with suggestions
         */
        static std::string getDataTypeErrorMessage(int dataTypeId, const char* context);
        
        /**
         * Get list of available data types in this build
         * @return Vector of compiled data types
         */
        static std::vector<DataType> getAvailableDataTypes();
        
        /**
         * Get human-readable name for data type
         * @param dataType The data type
         * @return String name (e.g., "FLOAT32", "INT64")
         */
        static const char* getDataTypeName(DataType dataType);
        
        /**
         * Get build configuration information
         * @return String describing the build configuration and available types
         */
        static std::string getBuildInfo();
        
        /**
         * Register a compiled data type (used internally by build system)
         * @param type The data type to register
         * @param name Human-readable name
         */
        static void registerCompiledType(DataType type, const std::string& name);
    };
    
    /**
     * RAII helper for data type validation in critical paths
     * 
     * Usage:
     *   DataTypeValidator validator(myDataType, "MyFunction::myMethod()");
     *   // Will throw if data type is invalid or not compiled
     */
    class SD_LIB_EXPORT DataTypeValidator {
    private:
        DataType dataType_;
        std::string context_;
        
    public:
        /**
         * Constructor that validates the data type
         * @param dt Data type to validate
         * @param context Context for error messages
         * @throws std::exception if validation fails
         */
        DataTypeValidator(DataType dt, const std::string& context) 
            : dataType_(dt), context_(context) {
            validateOrThrow();
        }
        
        /**
         * Perform validation and throw if invalid
         * @throws std::exception with detailed error message
         */
        void validateOrThrow();
        
        /**
         * Get the validated data type
         * @return The data type (guaranteed to be valid and compiled)
         */
        DataType getDataType() const { return dataType_; }
    };
}

#endif // ND4J_DATATYPE_VALIDATION_H
