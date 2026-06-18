#include "model_manager.h"
#include "tokenizer_wrapper.h"
#include <fstream>
#include <sstream>
#include <algorithm>

namespace tokenizers {

std::unique_ptr<TokenizerWrapper> ModelManager::load_model(
    const std::string& model_path,
    ModelFormat format) {
    
    if (!is_valid_model_file(model_path)) {
        throw TokenizerException("Invalid model file: " + model_path);
    }
    
    try {
        return std::make_unique<TokenizerWrapper>(model_path);
    } catch (const std::exception& e) {
        throw TokenizerException("Failed to load model from " + model_path + ": " + e.what());
    }
}

std::unique_ptr<TokenizerWrapper> ModelManager::load_embedded_model(
    const std::string& model_name) {
    
    // This would typically load from embedded resources
    // For now, we'll just throw an exception indicating it's not implemented
    throw TokenizerException("Embedded models not yet implemented: " + model_name);
}

std::unique_ptr<TokenizerWrapper> ModelManager::load_from_memory(
    const std::vector<uint8_t>& model_data,
    ModelFormat format) {
    
    if (model_data.empty()) {
        throw TokenizerException("Empty model data provided");
    }
    
    // Convert binary data to string (assuming it's JSON)
    std::string json_config(model_data.begin(), model_data.end());
    
    try {
        return std::make_unique<TokenizerWrapper>(TokenizerWrapper::from_json(json_config));
    } catch (const std::exception& e) {
        throw TokenizerException("Failed to load model from memory: " + std::string(e.what()));
    }
}

std::unique_ptr<TokenizerWrapper> ModelManager::load_from_json(
    const std::string& json_config) {
    
    if (json_config.empty()) {
        throw TokenizerException("Empty JSON configuration provided");
    }
    
    try {
        return std::make_unique<TokenizerWrapper>(TokenizerWrapper::from_json(json_config));
    } catch (const std::exception& e) {
        throw TokenizerException("Failed to load model from JSON: " + std::string(e.what()));
    }
}

bool ModelManager::is_valid_model_file(const std::string& model_path) {
    std::ifstream file(model_path);
    if (!file.is_open()) {
        return false;
    }
    
    // Check if file is not empty
    file.seekg(0, std::ios::end);
    if (file.tellg() == 0) {
        return false;
    }
    
    // Reset to beginning
    file.seekg(0, std::ios::beg);
    
    // Basic JSON validation - check if it starts with '{'
    char first_char;
    file >> first_char;
    
    return first_char == '{';
}

std::vector<std::string> ModelManager::get_embedded_models() {
    // Return empty vector for now - would be populated with embedded model names
    return {};
}

} // namespace tokenizers
