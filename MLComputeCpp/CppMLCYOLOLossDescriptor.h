#pragma once

#include <cstdint>
#include <vector>

class CppMLCYOLOLossLayer;

/*! @class      CppMLCYOLOLossDescriptor
    @discussion The MLCYOLOLossDescriptor specifies a YOLO loss filter descriptor.
 */
class CppMLCYOLOLossDescriptor
{
public:
    /*! @property   anchorBoxCount
     *  @abstract   number of anchor boxes used to detect object per grid cell
     */
    auto anchorBoxCount() -> uint32_t;

    /*! @property   anchorBoxes
     *  @abstract   \p NSData containing the width and height for \p anchorBoxCount anchor boxes
     *              This \p NSData should have 2 floating-point values per anchor box which represent the width
     *              and height of the anchor box.
     */
    auto anchorBoxes() -> std::vector<float>;

    /*! @property   shouldRescore
     *  @abstract   Rescore pertains to multiplying the confidence groundTruth with IOU (intersection over union)
     *              of predicted bounding box and the groundTruth boundingBox.  The default is YES
     */
    bool shouldRescore();
    void shouldRescore(bool value);

    /*! @property   scaleSpatialPositionLoss
     *  @abstract   The scale factor for spatial position loss and loss gradient.  The default is 10.0
     */
    auto scaleSpatialPositionLoss() -> float;
    void scaleSpatialPositionLoss(float);

    /*! @property   scaleSpatialSizeLoss
     *  @abstract   The scale factor for spatial size loss and loss gradient.  The default is 10.0
     */
    auto scaleSpatialSizeLoss() -> float;
    void scaleSpatialSizeLoss(float);

    /*! @property   scaleNoObject
     *  @abstract   The scale factor for no object confidence loss and loss gradient.  The default is 5.0
     */
    auto scaleNoObjectConfidenceLoss() -> float;
    void scaleNoObjectConfidenceLoss(float);

    /*! @property   scaleObject
     *  @abstract   The scale factor for object confidence loss and loss gradient.  The default is 100.0
     */
    auto scaleObjectConfidenceLoss() -> float;
    void scaleObjectConfidenceLoss(float);

    /*! @property   scaleClass
     *  @abstract   The scale factor for no object classes loss and loss gradient.  The default is 2.0
     */
    auto scaleClassLoss() -> float;
    void scaleClassLoss(float);

    /*! @property   positive IOU
     *  @abstract   If the prediction IOU with groundTruth is higher than this
     *              value we consider it a confident object presence, The default is 0.7
     */
    auto minimumIOUForObjectPresence() -> float;
    void minimumIOUForObjectPresence(float);

    /*! @property   negative IOU
     *  @abstract   If the prediction IOU with groundTruth is lower than this
     *              value we consider it a confident object absence.  The default is 0.3
     */
    auto maximumIOUForObjectAbsence() -> float;
    void maximumIOUForObjectAbsence(float);

    /*! @abstract  Create a YOLO loss descriptor object
        @param     anchorBoxes       The anchor box data
        @param     anchorBoxCount    The number of anchor boxes
        @return    A new MLCYOLOLossDescriptor object.
     */
    auto descriptorWithAnchorBoxes(std::vector<float> const& anchorBoxes, uint32_t anchorBoxCount) -> CppMLCYOLOLossDescriptor;

private:
    CppMLCYOLOLossDescriptor(void* self);

private:
    void* self;
    friend CppMLCYOLOLossLayer;
};
