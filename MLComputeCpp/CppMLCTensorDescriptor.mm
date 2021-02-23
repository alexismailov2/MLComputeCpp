#import "CppMLCTensorDescriptor.h"
#import "CppMLCTypesPrivate.h"

#import <MLCompute/MLCTensorDescriptor.h>

//auto
//CppMLCTensorDescriptor::ConvolutionWeightsDescriptor(uint32_t width,
//                                                     uint32_t height,
//                                                     uint32_t inputFeatureCount,
//                                                     uint32_t outputFeatureCount,
//                                                     eMLCDataType dataType) -> CppMLCTensorDescriptor
//{
//}
//
//auto
//CppMLCTensorDescriptor::ConvolutionBiasesDescriptor(uint32_t featureChannelCount,
//                                                    eMLCDataType dataType) -> CppMLCTensorDescriptor
//{
//}

CppMLCTensorDescriptor::CppMLCTensorDescriptor(std::vector<uint32_t> const& shape,
                                               eMLCDataType dataType)
    : self{[MLCTensorDescriptor descriptorWithShape:CppMLCTypesPrivate::toNSArray(shape)
                                           dataType:toNative(dataType)]}
{
}

CppMLCTensorDescriptor::CppMLCTensorDescriptor(std::vector<uint32_t> const& shape,
                                               std::vector<uint32_t> const& sequenceLengths,
                                               bool sortedSequences,
                                               eMLCDataType dataType)
    : self{[MLCTensorDescriptor descriptorWithShape:CppMLCTypesPrivate::toNSArray(shape)
                                    sequenceLengths:CppMLCTypesPrivate::toNSArray(sequenceLengths)
                                    sortedSequences:(sortedSequences) ? YES : NO
                                           dataType:toNative(dataType)]}
{
}

CppMLCTensorDescriptor::CppMLCTensorDescriptor(uint32_t width,
                                               uint32_t height,
                                               uint32_t featureChannels,
                                               uint32_t batchSize,
                                               eMLCDataType dataType)
    : self{[MLCTensorDescriptor descriptorWithWidth:(NSUInteger)width
                                             height:(NSUInteger)height
                                featureChannelCount:(NSUInteger)featureChannels
                                          batchSize:(NSUInteger)batchSize
                                           dataType:toNative(dataType)]}
{
}

CppMLCTensorDescriptor::CppMLCTensorDescriptor(uint32_t width,
                                               uint32_t height,
                                               uint32_t featureChannels,
                                               uint32_t batchSize)
    : self{[MLCTensorDescriptor descriptorWithWidth:(NSUInteger)width
                                             height:(NSUInteger)height
                                featureChannelCount:(NSUInteger)featureChannels
                                          batchSize:(NSUInteger)batchSize]}
{
}

CppMLCTensorDescriptor::~CppMLCTensorDescriptor() {
    [(id)self dealloc];
}

CppMLCTensorDescriptor::CppMLCTensorDescriptor(void* tensorDescriptor)
    : self{tensorDescriptor}
{
}

auto CppMLCTensorDescriptor::getDataType() -> eMLCDataType {
    return MLCDataTypeToCpp(((MLCTensorDescriptor*)self).dataType);
}

auto CppMLCTensorDescriptor::getDimensionsCount() -> uint32_t {
    return (uint32_t)(((MLCTensorDescriptor*)self).dimensionCount);
}

auto CppMLCTensorDescriptor::getShape() -> std::vector<uint32_t> {
    __block std::vector<uint32_t> vectorList;
    vectorList.reserve([((MLCTensorDescriptor*)self).shape count]);
    [((MLCTensorDescriptor*)self).shape enumerateObjectsUsingBlock:^(NSNumber* _Nonnull obj, NSUInteger idx, BOOL* _Nonnull stop) {
        vectorList.push_back((uint32_t)obj.unsignedIntegerValue);
    }];
    return vectorList;
}

auto CppMLCTensorDescriptor::getStride() -> std::vector<uint32_t> {
    __block std::vector<uint32_t> vectorList;
    vectorList.reserve([((MLCTensorDescriptor*)self).stride count]);
    [((MLCTensorDescriptor*)self).stride enumerateObjectsUsingBlock:^(NSNumber* _Nonnull obj, NSUInteger idx, BOOL* _Nonnull stop) {
        vectorList.push_back((uint32_t)obj.unsignedIntegerValue);
    }];
    return vectorList;
}

auto CppMLCTensorDescriptor::getTensorAllocationSizeInBytes() -> uint64_t {
    return (uint64_t)((MLCTensorDescriptor*)self).tensorAllocationSizeInBytes;
}

auto CppMLCTensorDescriptor::getSequenceLengths() -> std::vector<uint32_t> {
    __block std::vector<uint32_t> vectorList;
    vectorList.reserve([((MLCTensorDescriptor*)self).sequenceLengths count]);
    [((MLCTensorDescriptor*)self).sequenceLengths enumerateObjectsUsingBlock:^(NSNumber* _Nonnull obj, NSUInteger idx, BOOL* _Nonnull stop) {
        vectorList.push_back((uint32_t)obj.unsignedIntegerValue);
    }];
    return vectorList;
}

bool CppMLCTensorDescriptor::isSortedSequences() {
    return ((MLCTensorDescriptor*)self).sortedSequences == YES;
}

auto CppMLCTensorDescriptor::getBatchSizePerSequenceStep() -> std::vector<uint32_t> {
    __block std::vector<uint32_t> vectorList;
    vectorList.reserve([((MLCTensorDescriptor*)self).batchSizePerSequenceStep count]);
    [((MLCTensorDescriptor*)self).batchSizePerSequenceStep enumerateObjectsUsingBlock:^(NSNumber* _Nonnull obj, NSUInteger idx, BOOL* _Nonnull stop) {
        vectorList.push_back((uint32_t)obj.unsignedIntegerValue);
    }];
    return vectorList;
}

auto CppMLCTensorDescriptor::getMaxTensorDimensions() -> uint64_t {
    return (uint64_t)MLCTensorDescriptor.maxTensorDimensions;
}

