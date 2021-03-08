#pragma once

#include "CppMLCGraph.h"
#include "CppMLCOptimizer.h"
#include "CppMLCTensorParameter.h"
#include "CppMLCTensorOptimizerDeviceData.h"

#include <vector>

/*! @class      MLCTrainingGraph
    @discussion A training graph created from one or more MLCGraph objects
                plus additional layers added directly to the training graph.
 */
class CppMLCTrainingGraph : public CppMLCGraph
{
public:
    /*! @property   optimizer
        @abstract   The optimizer to be used with the training graph
     */
    auto optimizer() -> CppMLCOptimizer;

    /*! @property   The device memory size used by the training graph
        @abstract   Returns the total size in bytes of device memory used for all intermediate tensors
                    for forward, gradient passes and optimizer update for all layers in the training graph.
                    We recommend executing an iteration before checking the device memory size as
                    the buffers needed get allocated when the corresponding pass such as gradient,
                    optimizer update is executed.
        @return     A NSUInteger value
     */
    auto deviceMemorySize() -> uint32_t;

/*! @abstract   Create a training graph
    @param      graphObjects    The layers from these graph objects will be added to the training graph
    @param      lossLayer           The loss layer to use.  The loss layer can also be added to the training graph
                              using nodeWithLayer:sources:lossLabels
    @param      optimizer           The optimizer to use
    @return     A new training graph object
 */
static
auto graphWithGraphObjects(std::vector<CppMLCGraph> const& graphObjects,
                           CppMLCLayer& lossLayer,
                           CppMLCOptimizer& optimizer) -> CppMLCTrainingGraph;

/*! @abstract   Add the list of inputs to the training graph
    @param      inputs           The inputs
    @param      lossLabels  The loss label inputs
    @return     A boolean indicating success or failure
 */
bool addInputs(std::map<std::string, CppMLCTensor> const& inputs,
               std::map<std::string, CppMLCTensor> const& lossLabels);

/*! @abstract   Add the list of inputs to the training graph
    @discussion Each input, loss label or label weights tensor is identified by a NSString.
                When the training graph is executed, this NSString is used to identify which data object
                should be as input data for each tensor whose device memory needs to be updated
                before the graph is executed.
    @param      inputs                        The inputs
    @param      lossLabels               The loss label inputs
    @param      lossLabelWeights  The loss label weights
    @return     A boolean indicating success or failure
 */
bool addInputs(std::map<std::string, CppMLCTensor> const& inputs,
               std::map<std::string, CppMLCTensor> const& lossLabels,
               std::map<std::string, CppMLCTensor> const& lossLabelWeights);

/*! @abstract   Add the list of outputs to the training graph
    @param      outputs           The outputs
    @return     A boolean indicating success or failure
 */
bool addOutputs(std::map<std::string, CppMLCTensor> const& outputs);

/*! @abstract   Add the list of tensors whose contributions are not to be taken when computing gradients during gradient pass
    @param      tensors           The list of tensors
    @return     A boolean indicating success or failure
 */
bool stopGradientForTensors(std::vector<CppMLCTensor> const& tensors);

/*! @abstract   Compile the training graph for a device.
    @param      options     The compiler options to use when compiling the training graph
    @param      device       The MLCDevice object
    @return     A boolean indicating success or failure
 */
bool compileWithOptions(eMLCGraphCompilationOptions options,
                        CppMLCDevice& device);

/*! @abstract   Compile the optimizer to be used with a training graph.
    @discussion Typically the optimizer to be used with a training graph is specifed when the training graph is created using
                graphWithGraphObjects:lossLayer:optimizer.  The optimizer will be compiled in when compileWithOptions:device
                is called if an optimizer is specified with the training graph.  In the case where the optimizer to be used is not known
                when the graph is created or compiled, this method can be used to associate and compile a training graph with an optimizer.
    @param      optimizer       The MLCOptimizer object
    @return     A boolean indicating success or failure
 */
bool compileOptimizer(CppMLCOptimizer& optimizer);

/*! @abstract   Link mutiple training graphs
    @discussion This is used to link subsequent training graphs with first training sub-graph.
                This method should be used when we have tensors shared by one or more layers in multiple sub-graphs
    @param      graphs       The list of training graphs to link
    @return     A boolean indicating success or failure
 */
bool linkWithGraphs(std::vector<CppMLCTrainingGraph> const& graphs);

/*! @abstract   Get the gradient tensor for an input tensor
    @param      input   The input tensor
    @return     The gradient tensor
 */
auto gradientTensorForInput(CppMLCTensor& input) -> CppMLCTensor;

/*! @abstract   Get the source gradient tensors for a layer in the training graph
    @param      layer   A layer in the training graph
    @return     A list of tensors
 */
auto sourceGradientTensorsForLayer(CppMLCLayer& layer) -> std::vector<CppMLCTensor>;

/*! @abstract   Get the result gradient tensors for a layer in the training graph
    @param      layer   A layer in the training graph
    @return     A list of tensors
 */
auto resultGradientTensorsForLayer(CppMLCLayer& layer) -> std::vector<CppMLCTensor>;

/*! @abstract   Get the gradient data for a trainable parameter associated with a layer
    @discussion This can be used to get the gradient data for weights or biases parameters associated with a convolution,
                fully connected or convolution transpose layer
    @param      parameter   The updatable parameter associated with the layer
    @param      layer   A layer in the training graph.  Must be one of the following:
                      - MLCConvolutionLayer
                      - MLCFullyConnectedLayer
                      - MLCBatchNormalizationLayer
                      - MLCInstanceNormalizationLayer
                      - MLCGroupNormalizationLayer
                      - MLCLayerNormalizationLayer
                      - MLCEmbeddingLayer
                      - MLCMultiheadAttentionLayer
    @return     The gradient data.  Will return nil if the layer is marked as not trainable or if
                training graph is not executed with separate calls to forward and gradient passes.
*/
auto gradientDataForParameter(CppMLCTensor& parameter,
                              CppMLCLayer& layer) -> std::vector<float>;

/*! @abstract   Allocate an entry for a user specified gradient for a tensor
    @param      tensor   A result tensor produced by a layer in the training graph
                       that is input to some user specified code and will need to
                       provide a user gradient during the gradient pass.
    @return     A gradient tensor
 */
auto allocateUserGradientForTensor(CppMLCTensor& tensor) -> CppMLCTensor;

/*! @abstract   Execute the training graph (forward, gradient and optimizer update) with given source and label data
    @discussion Execute the training graph with given source and label data.  If an optimizer is specified, the optimizer update is applied.
                If MLCExecutionOptionsSynchronous is specified in 'options', this method returns after the graph has been executed.
                Otherwise, this method returns after the graph has been queued for execution. The completion handler is called after the graph
                has finished execution.
    @param      inputsData                               The data objects to use for inputs
    @param      lossLabelsData                      The data objects to use for loss labels
    @param      lossLabelWeightsData         The data objects to use for loss label weights
    @param      batchSize                                 The batch size to use.  For a graph where batch size changes between layers this value must be 0.
    @param      options                                      The execution options
    @param      completionHandler                The completion handler
    @return     A boolean indicating success or failure
*/
bool executeWithInputsData(std::map<std::string, CppMLCTensorData> const& inputsData,
                           std::map<std::string, CppMLCTensorData> const& lossLabelsData,
                           std::map<std::string, CppMLCTensorData> const& lossLabelWeightsData,
                           uint32_t batchSize,
                           eMLCExecutionOptions options,
                           MLCGraphCompletionHandler completionHandler);


/*! @abstract   Execute the training graph (forward, gradient and optimizer update) with given source and label data
    @param      inputsData                               The data objects to use for inputs
    @param      lossLabelsData                      The data objects to use for loss labels
    @param      lossLabelWeightsData         The data objects to use for loss label weights
    @param      outputsData                             The data objects to use for outputs
    @param      batchSize                                 The batch size to use.  For a graph where batch size changes between layers this value must be 0.
    @param      options                                     The execution options
    @param      completionHandler               The completion handler
    @return     A boolean indicating success or failure
*/
bool executeWithInputsData(std::map<std::string, CppMLCTensorData> const& inputsData,
                           std::map<std::string, CppMLCTensorData> const& lossLabelsData,
                           std::map<std::string, CppMLCTensorData> const& lossLabelWeightsData,
                           std::map<std::string, CppMLCTensorData> const& outputsData,
                           uint32_t batchSize,
                           eMLCExecutionOptions options,
                           MLCGraphCompletionHandler completionHandler);

/*! @abstract   Execute the forward pass of the training graph
    @param      batchSize                         The batch size to use.  For a graph where batch size changes between layers this value must be 0.
    @param      options                             The execution options
    @param      completionHandler       The completion handler
    @return     A boolean indicating success or failure
 */
bool executeForwardWithBatchSize(uint32_t batchSize,
                                 eMLCExecutionOptions options,
                                 MLCGraphCompletionHandler completionHandler);

/*! @abstract   Execute the forward pass for the training graph
    @param      batchSize                         The batch size to use.  For a graph where batch size changes between layers this value must be 0.
    @param      options                             The execution options
    @param      outputsData                     The data objects to use for outputs
    @param      completionHandler       The completion handler
    @return     A boolean indicating success or failure
 */
bool executeForwardWithBatchSize(uint32_t batchSize,
                                 eMLCExecutionOptions options,
                                 std::map<std::string, CppMLCTensorData> const& outputsData,
                                 MLCGraphCompletionHandler completionHandler);

/*! @abstract   Execute the gradient pass of the training graph
    @param      batchSize                         The batch size to use.  For a graph where batch size changes between layers this value must be 0.
    @param      options                             The execution options
    @param      completionHandler       The completion handler
    @return     A boolean indicating success or failure
 */
bool executeGradientWithBatchSize(uint32_t batchSize,
                                  eMLCExecutionOptions options,
                                  MLCGraphCompletionHandler completionHandler);

/*! @abstract   Execute the gradient pass of the training graph
    @param      batchSize                         The batch size to use.  For a graph where batch size changes between layers this value must be 0.
    @param      options                             The execution options
    @param      outputsData                     The data objects to use for outputs
    @param      completionHandler       The completion handler
    @return     A boolean indicating success or failure
 */
bool executeGradientWithBatchSize(uint32_t batchSize,
                                  eMLCExecutionOptions options,
                                  std::map<std::string, CppMLCTensorData> outputsData,
                                  MLCGraphCompletionHandler completionHandler);

/*! @abstract   Execute the optimizer update pass of the training graph
    @param      options                             The execution options
    @param      completionHandler       The completion handler
    @return     A boolean indicating success or failure
 */
bool executeOptimizerUpdateWithOptions(eMLCExecutionOptions options,
                                       MLCGraphCompletionHandler completionHandler);


/*! @abstract   Synchronize updates (weights/biases from convolution, fully connected and LSTM layers, tensor parameters)
                from device memory to host memory.
 */
void synchronizeUpdates();

/*! @abstract   Set the input tensor parameters that also will be updated by the optimizer
    @discussion These represent the list of input tensors to be updated when we execute the optimizer update
                Weights, bias or beta, gamma tensors are not included in this list.  MLCompute automatically
                adds them to the parameter list based on whether the layer is marked as updatable or not.
    @param      parameters   The list of input tensors to be updated by the optimizer
    @return     A boolean indicating success or failure
 */
bool setTrainingTensorParameters(std::vector<CppMLCTensorParameter> const& parameters);

/*! @abstract   Associates the given optimizer data and device data buffers with the tensor.
                Returns true if the data is successfully associated with the tensor and copied to the device.
    @discussion The caller must guarantee the lifetime of the underlying memory of \p data for the entirety of the tensor's
                lifetime.  The \p deviceData buffers are allocated by MLCompute.  This method must be called
                before executeOptimizerUpdateWithOptions or executeWithInputsData is called for the training graph.
                We recommend using this method instead of using [MLCTensor bindOptimizerData] especially if the
                optimizer update is being called multiple times for each batch.
    @param      data                The optimizer data to be associated with the tensor
    @param      deviceData  The optimizer device data to be associated with the tensor
    @param      tensor           The tensor
    @return     A Boolean value indicating whether the data is successfully associated with the tensor .
*/
bool bindOptimizerData(std::vector<CppMLCTensorData> const& data,
                       std::vector<CppMLCTensorOptimizerDeviceData> const& deviceData,
                       CppMLCTensor& tensor);

private:
    CppMLCTrainingGraph(void* self);
};
