#pragma once

#include "CppMLCDevice.h"
#include "CppMLCTypesPrivate.h"
#include "CppMLCTensorData.h"

#include <vector>
#include <map>

class CppMLCTypesPrivate;

/*! @class      MLCGraph
    @discussion A graph of layers that can be used to build a training or inference graph
 */
class CppMLCGraph
{
public:
    /*! @abstract   The device to be used when compiling and executing a graph
     */
    auto device() -> CppMLCDevice;

    /*! @abstract   Layers in the graph
     */
    auto layers() -> std::vector<CppMLCLayer>;

    /*! @abstract   Creates a new graph.
        @return     A new graph.
     */
    static auto graph() -> CppMLCGraph;

    /*! @abstract A DOT representation of the graph.
        @discussion For more info on the DOT language, refer to https://en.wikipedia.org/wiki/DOT_(graph_description_language).
                    Edges that have a dashed lines are those that have stop gradients, while those with solid lines don't.
    */
    auto summarizedDOTDescription() -> std::string;

    /*! @abstract   Add a layer to the graph
        @param      layer        The layer
        @param      source      The source tensor
        @return     A result tensor
     */
    auto nodeWithLayer(CppMLCLayer const& layer,
                       CppMLCTensor const& source) -> CppMLCTensor;

    /*! @abstract   Add a layer to the graph
        @param      layer        The layer
        @param      sources    A list of source tensors
        @discussion For variable length sequences of LSTMs/RNNs layers, create an MLCTensor of sortedSequenceLengths and pass it as the last index (i.e. index 2 or 4) of sources. This tensor must of be type MLCDataTypeInt32.
        @return     A result tensor
     */
    auto nodeWithLayer(CppMLCLayer const& layer,
                       std::vector<CppMLCTensor> const& sources) -> CppMLCTensor;

    /*! @abstract   Add a layer to the graph
        @param      layer                       The layer
        @param      sources                   A list of source tensors
        @param      disableUpdate     A flag to indicate if optimizer update should be disabled for this layer
        @discussion For variable length sequences of LSTMs/RNNs layers, create an MLCTensor of sortedSequenceLengths and pass it as the last index (i.e. index 2 or 4) of sources. This tensor must of be type MLCDataTypeInt32.
        @return     A result tensor
     */
    auto nodeWithLayer(CppMLCLayer const& layer,
                       std::vector<CppMLCTensor> const& sources,
                       bool disableUpdate) -> CppMLCTensor;

    /*! @abstract   Add a loss layer to the graph
        @param      layer                      The loss layer
        @param      lossLabels           The loss labels tensor
        @discussion For variable length sequences of LSTMs/RNNs layers, create an MLCTensor of sortedSequenceLengths and pass it as the last index (i.e. index 2 or 4) of sources. This tensor must of be type MLCDataTypeInt32.
        @return     A result tensor
     */
    auto nodeWithLayer(CppMLCLayer const& layer,
                       std::vector<CppMLCTensor> const& sources,
                       std::vector<CppMLCTensor> const& lossLabels) -> CppMLCTensor;

    /*! @abstract   Add a split layer to the graph
        @param      source                         The source tensor
        @param      splitCount                The number of splits
        @param      dimension                  The dimension to split the source tensor
        @return     A result tensor
     */
    auto splitWithSource(CppMLCTensor const& source,
                         uint32_t splitCount,
                         uint32_t dimension) -> std::vector<CppMLCTensor>;

    /*! @abstract   Add a split layer to the graph
        @param      source                                     The source tensor
        @param      splitSectionLengths        The lengths of each split section
        @param      dimension                              The dimension to split the source tensor
        @return     A result tensor
     */
    auto splitWithSource(CppMLCTensor const& source,
                         std::vector<uint32_t> const& splitSectionLengths,
                         uint32_t dimension) -> std::vector<CppMLCTensor>;

    /*! @abstract   Add a concat layer to the graph
        @param      sources      The source tensors to concatenate
        @param      dimension  The concatenation dimension
        @return     A result tensor
     */
    auto concatenateWithSources(std::vector<CppMLCTensor> const& sources,
                                uint32_t dimension) -> CppMLCTensor;

    /*! @abstract   Add a reshape layer to the graph
        @param      shape                     An array representing the shape of result tensor
        @param      source                   The source tensor
        @return     A result tensor
     */
    auto reshapeWithShape(std::vector<uint32_t> const& shape,
                          CppMLCTensor const& source) -> CppMLCTensor;

    /*! @abstract   Add a transpose layer to the graph
        @param      dimensions NSArray<NSNumber *> representing the desired ordering of dimensions
                    The dimensions array specifies the input axis source for each output axis, such that the
                    K'th element in the dimensions array specifies the input axis source for the K'th axis in the
                    output.  The batch dimension which is typically axis 0 cannot be transposed.
        @return     A result tensor
     */
    auto transposeWithDimensions(std::vector<uint32_t> const& dimensions,
                                 CppMLCTensor const& source) -> CppMLCTensor;

    /*! @abstract   Associates data with input tensors. If the device is GPU, also copies the data to the device memory.
                    Returns true if the data is successfully associated with input tensors.
        @discussion This function should be used if you execute the forward, gradient and optimizer updates independently.
                    Before the forward pass is executed, the inputs should be written to device memory.  Similarly, before the
                    gradient pass is executed, the inputs (typically the initial gradient tensor) should be written to device
                    memory.  The caller must guarantee the lifetime of the underlying memory of each value of \p inputsData
                    for the entirety of each corresponding input tensor's lifetime.
        @param      inputsData        The input data to use to write to device memory
        @param      inputTensors    The list of tensors to perform writes on
        @param      device                 The device
        @param      batchSize          The batch size.  This should be set to the actual batch size that may be used when we execute
                                  the graph and can be a value less than or equal to the batch size specified in the tensor.
                                  If set to 0, we use batch size specified in the tensor.
        @param      synchronous     Whether to execute the copy to the device synchronously.  For performance, asynchronous
                                 execution is recommended.
        @return     A Boolean value indicating whether the data is successfully associated with the tensor.
     */
    bool bindAndWriteData(std::map<std::string, CppMLCTensorData> const& inputsData,
                          std::map<std::string, CppMLCTensor> const& inputTensors,
                          CppMLCDevice const& device,
                          uint32_t batchSize,
                          bool synchronous);

    /*! @abstract   Associates data with input tensors. If the device is GPU, also copies the data to the device memory.
                    Returns true if the data is successfully associated with input tensors.
        @discussion This function should be used if you execute the forward, gradient and optimizer updates independently.
                    Before the forward pass is executed, the inputs should be written to device memory.  Similarly, before the
                    gradient pass is executed, the inputs (typically the initial gradient tensor) should be written to device
                    memory.  The caller must guarantee the lifetime of the underlying memory of each value of \p inputsData
                    for the entirety of each corresponding input tensor's lifetime.
        @param      inputsData        The input data to use to write to device memory
        @param      inputTensors    The list of tensors to perform writes on
        @param      device                 The device
        @param      synchronous     Whether to execute the copy to the device synchronously.  For performance, asynchronous
                                 execution is recommended.
        @return     A Boolean value indicating whether the data is successfully associated with the tensor.
     */
    bool bindAndWriteData(std::map<std::string, CppMLCTensorData> const& inputsData,
                          std::map<std::string, CppMLCTensor> const& inputTensors,
                          CppMLCDevice const& device,
                          bool synchronous);

    /*! @abstract   Get the source tensors for a layer in the training graph
        @param      layer   A layer in the training graph
        @return     A list of tensors
     */
    auto sourceTensorsForLayer(CppMLCLayer const& layer) -> std::vector<CppMLCTensor>;

    /*! @abstract   Get the result tensors for a layer in the training graph
        @param      layer   A layer in the training graph
        @return     A list of tensors
     */
    auto resultTensorsForLayer(CppMLCLayer const& layer) -> std::vector<CppMLCTensor>;

protected:
    CppMLCGraph(void* self);

private:
    void* self;
    friend CppMLCTypesPrivate;
};
