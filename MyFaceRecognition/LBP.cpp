#include <opencv2\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\objdetect\objdetect.hpp>
#include <iostream>
#include <cstring>

#include "LBP.h"

void LBP::train(cv::InputArrayOfArrays _in_src, cv::InputArray _in_labels, bool preserveData) {
	if (_in_src.kind() != cv::_InputArray::STD_VECTOR_MAT && _in_src.kind() != cv::_InputArray::STD_VECTOR_VECTOR) {
		std::string error_message = "The images are expected as InputArray::STD_VECTOR_MAT (a std::vector<Mat>) or _InputArray::STD_VECTOR_VECTOR (a std::vector< vector<...> >).";
		CV_Error(CV_StsBadArg, error_message);
	}
	if (_in_src.total() == 0) {
		std::string error_message = std::format("Empty training data was given. You'll need more than one sample to learn a model.");
		CV_Error(CV_StsUnsupportedFormat, error_message);
	}
	else if (_in_labels.getMat().type() != CV_32SC1) {
		std::string error_message = std::format("Labels must be given as integer (CV_32SC1). Expected %d, but was %d.", CV_32SC1, _in_labels.type());
		CV_Error(CV_StsUnsupportedFormat, error_message);
	}
	// get the vector of matrices
	vector<Mat> src;
	_in_src.getMatVector(src);
	// get the label matrix
	Mat labels = _in_labels.getMat();
	// check if data is well- aligned
	if (labels.total() != src.size()) {
		string error_message = format("The number of samples (src) must equal the number of labels (labels). Was len(samples)=%d, len(labels)=%d.", src.size(), _labels.total());
		CV_Error(CV_StsBadArg, error_message);
	}
	// if this model should be trained without preserving old data, delete old model data
	if (!preserveData) {
		_labels.release();
		_histograms.clear();
	}
	// append labels to _labels matrix
	for (size_t labelIdx = 0; labelIdx < labels.total(); labelIdx++) {
		_labels.push_back(labels.at<int>((int)labelIdx));
	}
	// store the spatial histograms of the original data
	for (size_t sampleIdx = 0; sampleIdx < src.size(); sampleIdx++) {
		// calculate lbp image
		Mat lbp_image = elbp(src[sampleIdx], _radius, _neighbors);
		// get spatial histogram from this lbp image
		Mat p = spatial_histogram(
			lbp_image, /* lbp_image */
			static_cast<int>(std::pow(2.0, static_cast<double>(_neighbors))), /* number of possible patterns */
			_grid_x, /* grid size x */
			_grid_y, /* grid size y */
			true);
		// add to templates
		_histograms.push_back(p);
	}
}