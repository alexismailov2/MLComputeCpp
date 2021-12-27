#ifndef MLC_EXAMPLE_CPPMLCTENSORDESCRIPTOR_H
#define MLC_EXAMPLE_CPPMLCTENSORDESCRIPTOR_H

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

    auto getDataType() -> eMLCDataType;
    auto getDimensionsCount() -> uint32_t;
    auto getShape() -> std::vector<uint32_t>;
    auto getStride() -> std::vector<uint32_t>;
    auto getTensorAllocationSizeInBytes() -> uint64_t;
    auto getSequenceLengths() -> std::vector<uint32_t>;
    bool isSortedSequences();
    auto getBatchSizePerSequenceStep() -> std::vector<uint32_t>;
    auto getMaxTensorDimensions() -> uint64_t;

private:
    CppMLCTensorDescriptor(void* tensorDescriptor);

public://
    void* self{};

    friend CppMLCTensor;
};

#endif //MLC_EXAMPLE_CPPMLCTENSORDESCRIPTOR_H
