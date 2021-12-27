#pragma once

#include "CppMLCTensorDescriptor.h"
#include "CppMLCTensorData.h"
#include "CppMLCDevice.h"
#include "CppMLCTensorOptimizerDeviceData.h"

#include <string>

class CppMLCBatchNormalizationLayer;
class CppMLCTensorParameter;
class CppMLCTensorData;
class CppMLCConvolutionLayer;
class CppMLCEmbeddingLayer;
class CppMLCFullyConnectedLayer;
class CppMLCGraph;
class CppMLCTypesPrivate;
class CppMLCGroupNormalizationLayer;
class CppMLCLossLayer;
class CppMLCLayerNormalizationLayer;
class CppMLCTrainingGraph;

class CppMLCTensor
{
public:
    auto tensorId() const -> uint64_t;
    auto descriptor() const -> CppMLCTensorDescriptor;
    auto data() const -> std::vector<float>;
    auto label() const -> std::string;
    void label(std::string const& label) const;
    auto device() -> CppMLCDevice;
    auto optimizerData() const -> std::vector<CppMLCTensorData>;
    auto optimizerDeviceData() const -> std::vector<CppMLCTensorOptimizerDeviceData>;

    /*! @abstract   Create a MLCTensor object
        @discussion Create a tensor object without any data
        @return     A new MLCTensor object
     */
    CppMLCTensor(CppMLCTensorDescriptor const& tensorDescriptor);

//    /*! @abstract   Create a MLCTensor object
//        @discussion Create a tensor object initialized with a random initializer such as Glorot Uniform.
//        @param      tensorDescriptor              The tensor descriptor
//        @param      randomInitializerType   The random initializer type
//        @return     A new MLCTensor object
//     */
//     CppMLCTensor(CppMLCTensorDescriptor const& tensorDescriptor, eMLCRandomInitializerType randomInitializerType);

//    /*! @abstract   Create a MLCTensor object
//        @discussion Create a tensor object with a MLCTensorData object that specifies the tensor data buffer
//        @param      tensorDescriptor              The tensor descriptor
//        @param      fillData                      The scalar data to fill to tensor with
//        @return     A new MLCTensor object
//     */
//    CppMLCTensor(CppMLCTensorDescriptor const& tensorDescriptor, uint32_t fillData);

//    /*! @abstract   Create a MLCTensor object
//        @discussion Create a tensor object with a MLCTensorData object that specifies the tensor data buffer
//        @param      tensorDescriptor              The tensor descriptor
//        @param      data                                         The random initializer type
//        @return     A new MLCTensor object
//     */
//     CppMLCTensor(CppMLCTensorDescriptor const& tensorDescriptor, CppMLCTensorData const& data);

    /*! @abstract   Create a MLCTensor object
        @discussion Create a tensor object without any data.  The tensor data type is MLCDataTypeFloat32.
        @param      shape                           The tensor shape
        @return     A new MLCTensor object
     */
    CppMLCTensor(std::vector<uint32_t> const& shape);

    /*! @abstract   Create a MLCTensor object
        @discussion Create a tensor object initialized with a random initializer such as Glorot Uniform.
                    The tensor data type is MLCDataTypeFloat32
        @param      shape                                       The tensor shape
        @param      randomInitializerType   The random initializer type
        @return     A new MLCTensor object
     */
    CppMLCTensor(std::vector<uint32_t> const& shape, eMLCRandomInitializerType randomInitializerType);

    /*! @abstract   Create a MLCTensor object
        @discussion Create a tensor object without any data
        @param      shape                           The tensor shape
        @param      dataType                    The tensor data type
        @return     A new MLCTensor object
     */
    CppMLCTensor(std::vector<uint32_t> const& shape, eMLCDataType dataType);

    /*! @abstract   Create a MLCTensor object
        @discussion Create a tensor object with data
        @param      shape                           The tensor shape
        @param      data                             The tensor data
        @param      dataType                    The tensor data type
        @return     A new MLCTensor object
     */
    CppMLCTensor(std::vector<uint32_t> const& shape, CppMLCTensorData const& data, eMLCDataType dataType);

    /*! @abstract   Create a MLCTensor object
        @discussion Create a tensor object with data
        @param      shape                  The tensor shape
        @param      fillData               The scalar value to initialize the tensor data with
        @param      dataType               The tensor data type
        @return     A new MLCTensor object
     */
    CppMLCTensor(std::vector<uint32_t> const& shape, uint32_t fillData, eMLCDataType dataType);

    /*! @abstract   Create a MLCTensor  object
        @discussion Create a NCHW tensor object with tensor data type = MLCDataTypeFloat32
        @param      width                           The tensor width
        @param      height                         The tensor height
        @param      featureChannelCount     Number of feature channels
        @param      batchSize                  The tensor batch size
        @return     A new MLCTensor object
     */
    CppMLCTensor(uint32_t width, uint32_t height, uint32_t featureChannelCount, uint32_t batchSize);

    /*! @abstract   Create a MLCTensor  object
        @discussion Create a NCHW tensor object initialized with a scalar value
        @param      width                           The tensor width
        @param      height                         The tensor height
        @param      featureChannelCount     Number of feature channels
        @param      batchSize                  The tensor batch size
        @param      fillData           The scalar value to initialize the tensor data with
        @param      dataType                    The tensor data type
        @return     A new MLCTensorData object
     */
    CppMLCTensor(uint32_t width, uint32_t height, uint32_t featureChannelCount, uint32_t batchSize, float fillData, eMLCDataType dataType);

    /*! @abstract   Create a MLCTensor  object
        @discussion Create a NCHW tensor object initialized with a random initializer type.
                    The tensor data type is MLCDataTypeFloat32
        @param      width                                      The tensor width
        @param      height                                    The tensor height
        @param      featureChannelCount                Number of feature channels
        @param      batchSize                              The tensor batch size
        @param      randomInitializerType   The random initializer type
        @return     A new MLCTensor object
     */
    CppMLCTensor(uint32_t width, uint32_t height, uint32_t featureChannelCount, uint32_t batchSize, eMLCRandomInitializerType randomInitializerType);

    /*! @abstract   Create a MLCTensor  object
        @discussion Create a NCHW tensor object with a tensor data object
                    The tensor data type is MLCDataTypeFloat32.
        @param      width                           The tensor width
        @param      height                         The tensor height
        @param      featureChannelCount     Number of feature channels
        @param      batchSize                  The tensor batch size
        @param      data                             The tensor data
        @return     A new MLCTensor object
     */
    CppMLCTensor(uint32_t width, uint32_t height, uint32_t featureChannelCount, uint32_t batchSize, CppMLCTensorData const& data);

    /*! @abstract   Create a MLCTensor  object
        @discussion Create a NCHW tensor object with a tensor data object
                    The tensor data type is MLCDataTypeFloat32.
        @param      width                           The tensor width
        @param      height                         The tensor height
        @param      featureChannelCount     Number of feature channels
        @param      batchSize                  The tensor batch size
        @param      data                             The tensor data
        @param      dataType                    The tensor data type
        @return     A new MLCTensor object
     */
    CppMLCTensor(uint32_t width, uint32_t height, uint32_t featureChannelCount, uint32_t batchSize, CppMLCTensorData const& data, eMLCDataType dataType);

    /*! @abstract   Create a MLCTensor  object
        @discussion Create a tensor typically used by a recurrent layer
                    The tensor data type is MLCDataTypeFloat32.
        @param      sequenceLength       The length of sequences stored in the tensor
        @param      featureChannelCount     Number of feature channels
        @param      batchSize                  The tensor batch size
        @return     A new MLCTensor object
     */
    static
    auto tensorWithSequenceLength(uint32_t sequenceLength,
                                  uint32_t featureChannelCount,
                                  uint32_t batchSize) -> CppMLCTensor;


    /*! @abstract   Create a MLCTensor  object
        @discussion Create a tensor typically used by a recurrent layer
                    The tensor data type is MLCDataTypeFloat32.
        @param      sequenceLength                   The length of sequences stored in the tensor
        @param      featureChannelCount                 Number of feature channels
        @param      batchSize                              The tensor batch size
        @param      randomInitializerType   The random initializer type
        @return     A new MLCTensor object
     */
     static
     auto tensorWithSequenceLength(uint32_t sequenceLength,
                                   uint32_t featureChannelCount,
                                   uint32_t batchSize,
                                   eMLCRandomInitializerType randomInitializerType) -> CppMLCTensor;

    /*! @abstract   Create a MLCTensor  object
        @discussion Create a tensor typically used by a recurrent layer
                    The tensor data type is MLCDataTypeFloat32.
        @param      sequenceLength       The length of sequences stored in the tensor
        @param      featureChannelCount     Number of feature channels
        @param      batchSize                  The tensor batch size
        @param      data                             The tensor data
        @return     A new MLCTensor object
     */
     static
     auto tensorWithSequenceLength(uint32_t sequenceLength,
                                   uint32_t featureChannelCount,
                                   uint32_t batchSize,
                                   CppMLCTensorData& data) -> CppMLCTensor;


    /*! @abstract   Create a MLCTensor  object
        @discussion Create a tensor of variable length sequences typically used by a recurrent layer
                    The tensor data type is MLCDataTypeFloat32.
        @param      sequenceLengths                 An array of sequence lengths
        @param      sortedSequences                 A flag to indicate if the sequence lengths are sorted.  If yes, they must be sorted in descending order
        @param      featureChannelCount                 Number of feature channels
        @param      batchSize                              The tensor batch size
        @param      randomInitializerType   The random initializer type
        @return     A new MLCTensor object
     */
    static
    auto tensorWithSequenceLengths(std::vector<uint32_t> const& sequenceLengths,
                                   bool sortedSequences,
                                   uint32_t featureChannelCount,
                                   uint32_t batchSize,
                                   eMLCRandomInitializerType randomInitializerType) -> CppMLCTensor;

    /*! @abstract   Create a MLCTensor  object
        @discussion Create a tensor of variable length sequences typically used by a recurrent layer
                    The tensor data type is MLCDataTypeFloat32.
        @param      sequenceLengths     An array of sequence lengths
        @param      sortedSequences     A flag to indicate if the sequence lengths are sorted.  If yes, they must be sorted in descending order
        @param      featureChannelCount     Number of feature channels
        @param      batchSize                  The tensor batch size
        @param      data                             The tensor data
        @return     A new MLCTensor object
     */
    static
    auto tensorWithSequenceLengths(std::vector<uint32_t> const& sequenceLengths,
                                   bool sortedSequences,
                                   uint32_t featureChannelCount,
                                   uint32_t batchSize,
                                   CppMLCTensorData& data) -> CppMLCTensor;

    /*! @abstract   Returns a Boolean value indicating whether the underlying data has valid floating-point numerics, i.e. it
                    does not contain NaN or INF floating-point values.
     */
    bool hasValidNumerics();;

    /*! @abstract   Synchronize the data in host memory.
        @discussion Synchronize the data in host memory i.e. tensor.data with latest contents in device memory
                    This should only be called once the graph that this tensor is used with has finished execution;
                    Otherwise the results in device memory may not be up to date.
                    NOTE:  This method should not be called from a completion callback when device is the GPU.
        @return     Returns YES if success, NO if there is a failure to synchronize
     */
    bool synchronizeData();

    /*! @abstract   Synchronize the optimizer data in host memory.
        @discussion Synchronize the optimizer data in host memory with latest contents in device memory
                    This should only be called once the graph that this tensor is used with has finished execution;
                    Otherwise the results in device memory may not be up to date.
                    NOTE:  This method should not be called from a completion callback when device is the GPU.
        @return     Returns YES if success, NO if there is a failure to synchronize
    */
    bool synchronizeOptimizerData();

    /*! @abstract   Copy tensor data from device memory to user specified memory
        @discussion Before copying tensor data from device memory, one may need to synchronize the device memory for example
                    when device is the GPU.  The synchronizeWithDevice argumet can be set appropraitely to indicate this.
                    For CPU this is ignored.  If the tensor has been specified in outputs of a graph using addOutputs,
                    synchronizeWithDevice should be set to NO.
                    NOTE:  This method should only be called once the graph that this tensor is used with has finished execution;
                    Otherwise the results in device memory may not be up to date.  synchronizeWithDevice must be set to NO
                    when this method is called from a completion callback for GPU.
        @param bytes                                     The user specified data in which to copy
        @param length                                   The size in bytes to copy
        @param synchronizeWithDevice  Whether to synchronize device memory if device is GPU
        @return     Returns YES if success, NO if there is a failure to synchronize
     */
     bool copyDataFromDeviceMemoryToBytes(void* bytes, uint32_t length, bool synchronizeWithDevice) const;

    /*! @abstract   Associates the given data to the tensor. If the device is GPU, also copies the data to the device memory.
                    Returns true if the data is successfully associated with the tensor and copied to the device.
        @discussion The caller must guarantee the lifetime of the underlying memory of \p data for the entirety of the tensor's
                    lifetime.  For input tensors, we recommend that the bindAndwriteData method provided by MLCTrainingGraph
                    and MLCInferenceGraph be used.  This method should only be used to allocate and copy data to device memory
                    for tensors that are typically layer parameters such as weights, bias for convolution layers, beta, gamma for
                    normalization layers.
        @param      data             The data to associated with the tensor
        @param      device           The compute device
        @return     A Boolean value indicating whether the data is successfully associated with the tensor and copied to the device.
    */
    bool bindAndWriteData(CppMLCTensorData const& data, CppMLCDevice const& toDevice);

    /*! @abstract   Associates the given optimizer data and device data buffers to the tensor.
                    Returns true if the data is successfully associated with the tensor and copied to the device.
        @discussion The caller must guarantee the lifetime of the underlying memory of \p data for the entirety of the tensor's
                    lifetime.  The \p deviceData buffers are allocated by MLCompute.  This method must be called
                    before executeOptimizerUpdateWithOptions or executeWithInputsData is called for the training graph.
        @param      data                The optimizer data to be associated with the tensor
        @param      deviceData  The optimizer device data to be associated with the tensor
        @return     A Boolean value indicating whether the data is successfully associated with the tensor .
    */
    bool bindOptimizerData(std::vector<CppMLCTensorData> const& data, std::vector<CppMLCTensorOptimizerDeviceData> const& deviceData);

    ~CppMLCTensor();

public: // TODO: Opened only for completion handler
    CppMLCTensor(void* self = nullptr);

private:
    void* self;
    friend CppMLCBatchNormalizationLayer;
    friend CppMLCTensorParameter;
    friend CppMLCConvolutionLayer;
    friend CppMLCEmbeddingLayer;
    friend CppMLCFullyConnectedLayer;
    friend CppMLCGraph;
    friend CppMLCTypesPrivate;
    friend CppMLCGroupNormalizationLayer;
    friend CppMLCLossLayer;
    friend CppMLCLayerNormalizationLayer;
    friend CppMLCTrainingGraph;
};

std::ostream& operator<<(std::ostream& out, CppMLCTensor const&);