#pragma once

#include "CppMLCTypes.h"

#include <cstdint>
#include <vector>

class CppMLCTensor;

class CppMLCTensorDescriptor {
public:
//    static auto ConvolutionWeightsDescriptor(uint32_t width, uint32_t height, uint32_t inputFeatureCount, uint32_t outputFeatureCount, eMLCDataType dataType) -> CppMLCTensorDescriptor;
//    static auto ConvolutionBiasesDescriptor(uint32_t featureChannelCount, eMLCDataType dataType) -> CppMLCTensorDescriptor;

    CppMLCTensorDescriptor(std::vector<uint32_t> const& shape, eMLCDataType dataType = eMLCDataType::Float32);
    CppMLCTensorDescriptor(std::vector<uint32_t> const& shape, std::vector<uint32_t> const& sequenceLengths, bool sortedSequences, eMLCDataType dataType);
    CppMLCTensorDescriptor(uint32_t width, uint32_t height, uint32_t featureChannels, uint32_t batchSize, eMLCDataType dataType = eMLCDataType::Float32);
    CppMLCTensorDescriptor(uint32_t width, uint32_t height, uint32_t featureChannels, uint32_t batchSize);
    ~CppMLCTensorDescriptor();

    auto getDataType() const -> eMLCDataType;
    auto getDimensionsCount() const -> uint32_t;
    auto getShape() const -> std::vector<uint32_t>;
    auto getStride() const -> std::vector<uint32_t>;
    auto getTensorAllocationSizeInBytes() const -> uint64_t;
    auto getSequenceLengths() const -> std::vector<uint32_t>;
    bool isSortedSequences() const;
    auto getBatchSizePerSequenceStep() const -> std::vector<uint32_t>;
    auto getMaxTensorDimensions() const -> uint64_t;

private:
    CppMLCTensorDescriptor(void* tensorDescriptor);

public:
    void* self{};

    friend CppMLCTensor;
};

std::ostream& operator<<(std::ostream& out, CppMLCTensorDescriptor const&);
