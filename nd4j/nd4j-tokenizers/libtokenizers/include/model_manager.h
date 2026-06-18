#pragma once

#include <string>
#include <vector>
#include <memory>

namespace tokenizers {

class TokenizerWrapper;

/**
 * @brief Manager for loading and caching tokenizer models
 */
class ModelManager {
public:
    enum class ModelFormat {
        HuggingFaceJSON,
        SentencePiece,
        Custom
    };
    
    /**
     * @brief Load a tokenizer from file
     * @param model_path Path to the model file
     * @param format Model format (auto-detected if not specified)
     * @return Unique pointer to TokenizerWrapper
     */
    static std::unique_ptr<TokenizerWrapper> load_model(
        const std::string& model_path,
        ModelFormat format = ModelFormat::HuggingFaceJSON
    );
    
    /**
     * @brief Load a tokenizer from embedded resources
     * @param model_name Name of the embedded model
     * @return Unique pointer to TokenizerWrapper
     */
    static std::unique_ptr<TokenizerWrapper> load_embedded_model(
        const std::string& model_name
    );
    
    /**
     * @brief Load a tokenizer from memory blob
     * @param model_data Binary model data
     * @param format Model format
     * @return Unique pointer to TokenizerWrapper
     */
    static std::unique_ptr<TokenizerWrapper> load_from_memory(
        const std::vector<uint8_t>& model_data,
        ModelFormat format = ModelFormat::HuggingFaceJSON
    );
    
    /**
     * @brief Load a tokenizer from JSON string
     * @param json_config JSON configuration string
     * @return Unique pointer to TokenizerWrapper
     */
    static std::unique_ptr<TokenizerWrapper> load_from_json(
        const std::string& json_config
    );
    
    /**
     * @brief Check if a model file exists and is valid
     * @param model_path Path to check
     * @return true if valid, false otherwise
     */
    static bool is_valid_model_file(const std::string& model_path);
    
    /**
     * @brief Get list of available embedded models
     * @return Vector of model names
     */
    static std::vector<std::string> get_embedded_models();

private:
    ModelManager() = default;
};

} // namespace tokenizers
