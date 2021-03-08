#import "CppMLCTensor.h"
#import "CppMLCTensorDescriptor.h"
#import "CppMLCTensorData.h"
#import "CppMLCTypesPrivate.h"

#import <Foundation/Foundation.h>
#import <MLCompute/MLCTensor.h>

auto CppMLCTensor::tensorId() -> uint64_t
{
    return (uint64_t)((MLCTensor*)self).tensorID;
}

auto CppMLCTensor::descriptor() -> CppMLCTensorDescriptor
{
    return CppMLCTensorDescriptor((void*)(((MLCTensor*)self).descriptor));
}

auto CppMLCTensor::data() -> std::vector<float>
{
    return CppMLCTypesPrivate::NSDataToVectorFloat(((MLCTensor*)self).data);
}

auto CppMLCTensor::label() -> std::string
{
    return std::string([((MLCTensor*)self).label UTF8String]);
}

void CppMLCTensor::label(std::string const& label)
{
    [((MLCTensor*)self).label initWithUTF8String:label.c_str()];
}

auto CppMLCTensor::device() -> CppMLCDevice {
    return CppMLCDevice{((MLCTensor*)self).device};
}

auto CppMLCTensor::optimizerData() -> std::vector<CppMLCTensorData>
{
    return CppMLCTypesPrivate::MLCTensorDataArrayToVector(((MLCTensor*)self).optimizerData);
}

auto CppMLCTensor::optimizerDeviceData() -> std::vector<CppMLCTensorOptimizerDeviceData>
{
    return CppMLCTypesPrivate::MLCTensorOptimizerDeviceDataToVector(((MLCTensor*)self).optimizerDeviceData);
}

CppMLCTensor::CppMLCTensor(const CppMLCTensorDescriptor &tensorDescriptor)
    : self{[MLCTensor tensorWithDescriptor:(MLCTensorDescriptor*)tensorDescriptor.self]}
{
}

CppMLCTensor::CppMLCTensor(CppMLCTensorDescriptor const& tensorDescriptor,
                           eMLCRandomInitializerType randomInitializerType)
    : self{[MLCTensor tensorWithDescriptor:(MLCTensorDescriptor*)tensorDescriptor.self
                     randomInitializerType:toNative(randomInitializerType)]}
{
}

CppMLCTensor::CppMLCTensor(std::vector<uint32_t> const& shape)
    : self{[MLCTensor tensorWithShape:CppMLCTypesPrivate::toNSArray(shape)]}
{
}

CppMLCTensor::CppMLCTensor(CppMLCTensorDescriptor const& tensorDescriptor, CppMLCTensorData const& data)
    : self{[MLCTensor tensorWithDescriptor:(MLCTensorDescriptor*)tensorDescriptor.self
                                      data:(MLCTensorData*)data.self]}
{
}

CppMLCTensor::CppMLCTensor(std::vector<uint32_t> const& shape, eMLCRandomInitializerType randomInitializerType)
    : self{[MLCTensor tensorWithShape:CppMLCTypesPrivate::toNSArray(shape)
                randomInitializerType:toNative(randomInitializerType)]}
{
}

CppMLCTensor::CppMLCTensor(const std::vector<uint32_t> &shape, eMLCDataType dataType)
    : self{[MLCTensor tensorWithShape:CppMLCTypesPrivate::toNSArray(shape)
                             dataType:toNative(dataType)]}
{
}

CppMLCTensor::CppMLCTensor(const std::vector<uint32_t> &shape, const CppMLCTensorData &data, eMLCDataType dataType)
    : self{[MLCTensor tensorWithShape:CppMLCTypesPrivate::toNSArray(shape)
                                 data:(MLCTensorData*)data.self
                             dataType:toNative(dataType)]}
{
}

CppMLCTensor::CppMLCTensor(const std::vector<uint32_t> &shape, uint32_t fillData, eMLCDataType dataType)
    : self{[MLCTensor tensorWithShape:CppMLCTypesPrivate::toNSArray(shape)
                         fillWithData:[NSNumber numberWithUnsignedInt:(NSUInteger)fillData]
                             dataType:toNative(dataType)]}
{
}

CppMLCTensor::CppMLCTensor(uint32_t width, uint32_t height, uint32_t featureChannelCount, uint32_t batchSize)
    : self{[MLCTensor tensorWithWidth:(NSUInteger)width
                               height:(NSUInteger)height
                  featureChannelCount:(NSUInteger)featureChannelCount
                            batchSize:(NSUInteger)batchSize]}
{
}

CppMLCTensor::CppMLCTensor(uint32_t width, uint32_t height, uint32_t featureChannelCount, uint32_t batchSize,
                           float fillData, eMLCDataType dataType)
    : self{[MLCTensor tensorWithWidth:(NSUInteger)width
                               height:(NSUInteger)height
                  featureChannelCount:(NSUInteger)featureChannelCount
                            batchSize:(NSUInteger)batchSize
                         fillWithData:fillData
                             dataType:toNative(dataType)]}
{
}

CppMLCTensor::CppMLCTensor(uint32_t width, uint32_t height, uint32_t featureChannelCount, uint32_t batchSize,
                           eMLCRandomInitializerType randomInitializerType)
    : self{[MLCTensor tensorWithWidth:(NSUInteger)width
                               height:(NSUInteger)height
                  featureChannelCount:(NSUInteger)featureChannelCount
                            batchSize:(NSUInteger)batchSize
                randomInitializerType:toNative(randomInitializerType)]}
{
}

CppMLCTensor::CppMLCTensor(uint32_t width, uint32_t height, uint32_t featureChannelCount, uint32_t batchSize,
                           const CppMLCTensorData &data)
    : self{[MLCTensor tensorWithWidth:(NSUInteger)width
                               height:(NSUInteger)height
                  featureChannelCount:(NSUInteger)featureChannelCount
                            batchSize:(NSUInteger)batchSize
                                 data:(MLCTensorData*)data.self]}
{
}

CppMLCTensor::CppMLCTensor(uint32_t width, uint32_t height, uint32_t featureChannelCount, uint32_t batchSize,
                           const CppMLCTensorData &data, eMLCDataType dataType)
    : self{[MLCTensor tensorWithWidth:(NSUInteger)width
                               height:(NSUInteger)height
                  featureChannelCount:(NSUInteger)featureChannelCount
                            batchSize:(NSUInteger)batchSize
                                 data:(MLCTensorData*)data.self
                             dataType:toNative(dataType)]}
{
}

auto CppMLCTensor::tensorWithSequenceLength(uint32_t sequenceLength,
                                            uint32_t featureChannelCount,
                                            uint32_t batchSize) -> CppMLCTensor
{
    return CppMLCTensor{[MLCTensor tensorWithSequenceLength:(NSUInteger)sequenceLength
                                        featureChannelCount:(NSUInteger)featureChannelCount
                                                  batchSize:(NSUInteger)batchSize]};
}

auto CppMLCTensor::tensorWithSequenceLengths(std::vector<uint32_t> const& sequenceLengths,
                                             bool sortedSequences,
                                             uint32_t featureChannelCount,
                                             uint32_t batchSize,
                                             CppMLCTensorData& data) -> CppMLCTensor
{
    return CppMLCTensor{[MLCTensor tensorWithSequenceLengths:CppMLCTypesPrivate::toNSArray(sequenceLengths)
                                             sortedSequences:sortedSequences ? YES : NO
                                         featureChannelCount:(NSUInteger)featureChannelCount
                                                   batchSize:(NSUInteger)batchSize
                                                        data:(MLCTensorData*)data.self]};
}

auto CppMLCTensor::tensorWithSequenceLength(uint32_t sequenceLength,
                                            uint32_t featureChannelCount,
                                            uint32_t batchSize,
                                            eMLCRandomInitializerType randomInitializerType) -> CppMLCTensor
{
    return CppMLCTensor{[MLCTensor tensorWithSequenceLength:(NSInteger)sequenceLength
                                        featureChannelCount:featureChannelCount
                                                  batchSize:(NSUInteger)batchSize
                                      randomInitializerType:toNative(randomInitializerType)]};
}

auto CppMLCTensor::tensorWithSequenceLength(uint32_t sequenceLength,
                                            uint32_t featureChannelCount,
                                            uint32_t batchSize,
                                            CppMLCTensorData &data) -> CppMLCTensor
{
    return CppMLCTensor{[MLCTensor tensorWithSequenceLength:(NSInteger)sequenceLength
                                        featureChannelCount:featureChannelCount
                                                  batchSize:(NSUInteger)batchSize
                                                       data:(MLCTensorData*)data.self]};
}

auto CppMLCTensor::tensorWithSequenceLengths(std::vector<uint32_t> const& sequenceLengths,
                                             bool sortedSequences,
                                             uint32_t featureChannelCount,
                                             uint32_t batchSize,
                                             eMLCRandomInitializerType randomInitializerType) -> CppMLCTensor
{
    return CppMLCTensor{[MLCTensor tensorWithSequenceLengths:CppMLCTypesPrivate::toNSArray(sequenceLengths)
                                             sortedSequences:sortedSequences ? YES : NO
                                         featureChannelCount:(NSUInteger)featureChannelCount
                                                   batchSize:(NSUInteger)batchSize
                                       randomInitializerType:toNative(randomInitializerType)]};
}

bool CppMLCTensor::hasValidNumerics()
{
    return ((MLCTensor*)self).hasValidNumerics == YES;
}

bool CppMLCTensor::synchronizeData()
{
    return [(MLCTensor*)self synchronizeData] == YES;
}

bool CppMLCTensor::synchronizeOptimizerData()
{
    return [(MLCTensor*)self synchronizeOptimizerData] == YES;
}

bool CppMLCTensor::copyDataFromDeviceMemoryToBytes(void* bytes, uint32_t length, bool synchronizeWithDevice)
{
    return [(MLCTensor*)self copyDataFromDeviceMemoryToBytes:bytes
                                                      length:(NSUInteger)length
                                       synchronizeWithDevice:synchronizeWithDevice ? YES : NO] == YES;
}

bool CppMLCTensor::bindAndWriteData(CppMLCTensorData const& data, CppMLCDevice const& toDevice)
{
    return [(MLCTensor*)self bindAndWriteData:(MLCTensorData*)data.self
                                       toDevice:(MLCDevice*)toDevice.self] == YES;
}

bool CppMLCTensor::bindOptimizerData(std::vector<CppMLCTensorData> const& data,
                                     std::vector<CppMLCTensorOptimizerDeviceData> const& deviceData)
{
    return [(MLCTensor*)self bindOptimizerData:CppMLCTypesPrivate::toNSArray(data)
                                    deviceData:CppMLCTypesPrivate::toNSArray(deviceData)] == YES;
}

CppMLCTensor::CppMLCTensor(void *self)
    : self{self}
{
}









