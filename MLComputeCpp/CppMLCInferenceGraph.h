#pragma once

#include "CppMLCGraph.h"

class CppMLCGraphCompilationOptions;
class CppMLCTypesPrivate;

/*! @class      MLCInferenceGraph
    @discussion An inference graph created from one or more MLCGraph objects
                plus additional layers added directly to the inference graph.
 */
class CppMLCInferenceGraph : public CppMLCGraph
{
public:
    /*! @property   The device memory size used by the inference graph
        @abstract   Returns the total size in bytes of device memory used by all intermediate tensors in the inference graph
        @return     A NSUInteger value
     */
    auto deviceMemorySize() -> uint32_t;

    /*! @abstract   Create an inference graph
        @param      graphObjects    The layers from these graph objects will be added to the training graph
        @return     A new inference graph object
     */
    static auto graphWithGraphObjects(std::vector<CppMLCGraph> const& graphObjects) -> CppMLCInferenceGraph;

    /*! @abstract   Add the list of inputs to the inference graph
        @param      inputs           The inputs
        @return     A boolean indicating success or failure
     */
    bool addInputs(std::map<std::string, CppMLCTensor> const& inputs);

    /*! @abstract   Add the list of inputs to the inference graph
        @discussion Each input, loss label or label weights tensor is identified by a NSString.
                    When the inference graph is executed, this NSString is used to identify which data object
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

    /*! @abstract   Add the list of outputs to the inference graph
        @param      outputs           The outputs
        @return     A boolean indicating success or failure
     */
    bool addOutputs(std::map<std::string, CppMLCTensor> const& outputs);

    /*! @abstract   Compile the training graph for a device.
        @param      options     The compiler options to use when compiling the training graph
        @param      device       The MLCDevice object
        @return     A boolean indicating success or failure
     */
    bool compileWithOptions(eMLCGraphCompilationOptions options, CppMLCDevice const& device);

    /*! @abstract   Link mutiple inference graphs
        @discussion This is used to link subsequent inference graphs with first inference sub-graph.
                    This method should be used when we have tensors shared by one or more layers in multiple sub-graphs
        @param      graphs     The list of inference graphs to link
        @return     A boolean indicating success or failure
     */
    bool linkWithGraphs(std::vector<CppMLCInferenceGraph> const& graphs);

    /*! @abstract   Execute the inference graph with given input data
        @discussion Execute the inference graph given input data.
                    If MLCExecutionOptionsSynchronous is specified in 'options', this method returns after the graph has been executed.
                    Otherwise, this method returns after the graph has been queued for execution.  The completion handler  is called after the graph has finished execution.
        @param      inputsData                       The data objects to use for inputs
        @param      batchSize                         The batch size to use.  For a graph where batch size changes between layers this value must be 0.
        @param      options                             The execution options
        @param      completionHandler       The completion handler
        @return     A boolean indicating success or failure
    */
    bool executeWithInputsData(std::map<std::string, CppMLCTensorData> const& inputsData,
                               uint32_t batchSize,
                               eMLCExecutionOptions options,
                               CppMLCGraphCompletionHandler completionHandler);
    bool executeWithInputsData(std::map<std::string, CppMLCTensorData*> const& inputsData,
                               uint32_t batchSize,
                               eMLCExecutionOptions options,
                               CppMLCGraphCompletionHandler completionHandler);

    /*! @abstract   Execute the inference graph with given input data
        @discussion Execute the inference graph given input data.
                    If MLCExecutionOptionsSynchronous is specified in 'options', this method returns after the graph has been executed.
                    Otherwise, this method returns after the graph has been queued for execution.  The completion handler  is called after the graph has finished execution.
        @param      inputsData                       The data objects to use for inputs
        @param      outputsData                     The data objects to use for outputs
        @param      batchSize                         The batch size to use.  For a graph where batch size changes between layers this value must be 0.
        @param      options                             The execution options
        @param      completionHandler       The completion handler
        @return     A boolean indicating success or failure
    */
    bool executeWithInputsData(std::map<std::string, CppMLCTensorData> const& inputsData,
                               std::map<std::string, CppMLCTensorData> const& outputsData,
                               uint32_t batchSize,
                               eMLCExecutionOptions options,
                               CppMLCGraphCompletionHandler completionHandler);


    /*! @abstract   Execute the inference graph with given input data
        @discussion Execute the inference graph given input data.
                    If MLCExecutionOptionsSynchronous is specified in 'options', this method returns after the graph has been executed.
                    Otherwise, this method returns after the graph has been queued for execution.  The completion handler  is called after the graph has finished execution.
        @param      inputsData                       The data objects to use for inputs
        @param      lossLabelsData              The data objects to use for loss labels
        @param      lossLabelWeightsData The data objects to use for loss label weights
        @param      batchSize                         The batch size to use.  For a graph where batch size changes between layers this value must be 0.
        @param      options                             The execution options
        @param      completionHandler       The completion handler
        @return     A boolean indicating success or failure
    */
    bool executeWithInputsData(std::map<std::string, CppMLCTensorData> const& inputsData,
                               std::map<std::string, CppMLCTensorData> const& lossLabelsData,
                               std::map<std::string, CppMLCTensorData> const& lossLabelWeightsData,
                               uint32_t batchSize,
                               eMLCExecutionOptions options,
                               CppMLCGraphCompletionHandler completionHandler);

    /*! @abstract   Execute the inference graph with given input data
        @discussion Execute the inference graph given input data.
                    If MLCExecutionOptionsSynchronous is specified in 'options', this method returns after the graph has been executed.
                    Otherwise, this method returns after the graph has been queued for execution.  The completion handler  is called after the graph has finished execution.
        @param      inputsData                       The data objects to use for inputs
        @param      lossLabelsData              The data objects to use for loss labels
        @param      lossLabelWeightsData The data objects to use for loss label weights
        @param      outputsData                     The data objects to use for outputs
        @param      batchSize                         The batch size to use.  For a graph where batch size changes between layers this value must be 0.
        @param      options                             The execution options
        @param      completionHandler       The completion handler
        @return     A boolean indicating success or failure
    */
    bool executeWithInputsData(std::map<std::string, CppMLCTensorData> const& inputsData,
                               std::map<std::string, CppMLCTensorData> const& lossLabelsData,
                               std::map<std::string, CppMLCTensorData> const& lossLabelWeightsData,
                               std::map<std::string, CppMLCTensorData> const& outputsData,
                               uint32_t batchSize,
                               eMLCExecutionOptions options,
                               CppMLCGraphCompletionHandler completionHandler);

private:
    CppMLCInferenceGraph(void* self = nullptr);

private:
    void* self;
    friend CppMLCTypesPrivate;
};
