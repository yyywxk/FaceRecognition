#include "LBPH.h"
#include <iostream>
using namespace std;

void elbp(InputArray src, OutputArray dst, int radius, int neighbors);
static Mat spatial_histogram(InputArray _src, int numPatterns, int grid_x, int grid_y, bool /*normed*/);

void LBPH::train(InputArrayOfArrays _in_src, InputArray _in_labels) {
	this->train(_in_src, _in_labels, false);
}

void LBPH::update(InputArrayOfArrays _in_src, InputArray _in_labels) {
	// got no data, just return
	if (_in_src.total() == 0)
		return;

	this->train(_in_src, _in_labels, true);
}

void LBPH::train(InputArrayOfArrays _in_src, InputArray _in_labels, bool preserveData) {
	if (_in_src.kind() != _InputArray::STD_VECTOR_MAT && _in_src.kind() != _InputArray::STD_VECTOR_VECTOR) {
		string error_message = "The images are expected as InputArray::STD_VECTOR_MAT (a std::vector<Mat>) or _InputArray::STD_VECTOR_VECTOR (a std::vector< vector<...> >).";
		CV_Error(CV_StsBadArg, error_message);
	}
	if (_in_src.total() == 0) {
		string error_message = format("Empty training data was given. You'll need more than one sample to learn a model.");
		CV_Error(CV_StsUnsupportedFormat, error_message);
	}
	else if (_in_labels.getMat().type() != CV_32SC1) {
		string error_message = format("Labels must be given as integer (CV_32SC1). Expected %d, but was %d.", CV_32SC1, _in_labels.type());
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
	for (size_t sampleIdx = 0; sampleIdx < src.size(); ++sampleIdx) {
		// calculate lbp image
		Mat lbp_image;
		elbp(src[sampleIdx], lbp_image, _radius, _neighbors);
		// get spatial histogram from this lbp image
		Mat p = spatial_histogram(
			lbp_image, /* lbp_image */
			static_cast<int>(std::pow(2.0, static_cast<double>(_neighbors))), /* number of possible patterns */
			_grid_x, /* grid size x */
			_grid_y, /* grid size y */
			true);
		// add to templates
		//Mat tem_p = p.clone();
		_histograms.push_back(p);
	}
}

static Mat histc_(const Mat& src, int minVal = 0, int maxVal = 255, bool normed = false)
{
	Mat result;
	// Establish the number of bins.
	int histSize = maxVal - minVal + 1;
	// Set the ranges.
	float range[] = { static_cast<float>(minVal), static_cast<float>(maxVal + 1) };
	const float* histRange = { range };
	// calc histogram
	calcHist(&src, 1, 0, Mat(), result, 1, &histSize, &histRange, true, false);
	// normalize
	if (normed) {
		result /= (int)src.total();

	}
	return result.reshape(1, 1);
}

static Mat histc(InputArray _src, int minVal, int maxVal, bool normed)
{
	Mat src = _src.getMat();
	switch (src.type()) {
	case CV_8SC1:
		return histc_(Mat_<float>(src), minVal, maxVal, normed);
		break;
	case CV_8UC1:
		return histc_(src, minVal, maxVal, normed);
		break;
	case CV_16SC1:
		return histc_(Mat_<float>(src), minVal, maxVal, normed);
		break;
	case CV_16UC1:
		return histc_(src, minVal, maxVal, normed);
		break;
	case CV_32SC1:
		return histc_(Mat_<float>(src), minVal, maxVal, normed);
		break;
	case CV_32FC1:
		return histc_(src, minVal, maxVal, normed);
		break;
	default:
		CV_Error(CV_StsUnmatchedFormats, "This type is not implemented yet."); break;

	}
	return Mat();
}

static Mat spatial_histogram(InputArray _src, int numPatterns, int grid_x, int grid_y, bool /*normed*/)
{
	// ����LBPM�Ŀռ�ֱ��ͼ�ֲ����õ�һ��һά����
	// srcΪLBPM��ͨ��olbp����elbp����õ���
	// numPatternsΪ����LBP��ģʽ��Ŀ��һ��Ϊ2����
	// grid_x��grid_y�ֱ�Ϊÿ�л�ÿ�е�block����
	// normedΪ�Ƿ���й�һ������
	Mat src = _src.getMat();
	// allocate memory for the spatial histogramΪLBPH�����ڴ�ռ�
	Mat result = Mat::zeros(grid_x * grid_y, numPatterns, CV_32FC1);
	// return matrix with zeros if no data was given�����û���������ݣ����ص���0
	if (src.empty())
		return result.reshape(1, 1);
	// calculate LBP patch size block�ĳߴ�
	int width = src.cols / grid_x;
	int height = src.rows / grid_y;
	// initial result_row ��ʼ�������
	int resultRowIdx = 0;
	// iterate through grid
	for (int i = 0; i < grid_y; i++)
	{
		for (int j = 0; j < grid_x; j++)
		{
			// ��ȡָ������
			Mat src_cell = Mat(src, Range(i*height, (i + 1)*height), Range(j*width, (j + 1)*width));
			// ����ָ�������ֱ��ͼ
			Mat cell_hist = histc(src_cell, 0, (numPatterns - 1), true);
			// copy to the result matrix ������õ��Ľ��������ÿһ��
			Mat result_row = result.row(resultRowIdx);
			cell_hist.reshape(1, 1).convertTo(result_row, CV_32FC1);
			// increase row count in result matrix
			resultRowIdx++;
		}
	}
	// return result as reshaped feature vector
	return result.reshape(1, 1);
}


// srcΪ����ͼ��dstΪ���ͼ��radiusΪ�뾶��neighborΪ���㵱ǰ��LBP������������ص�����Ҳ�������������
template <typename _Tp> static // ģ�庯�������ݲ�ͬ��ԭʼ�������͵õ���ͬ�Ľ��
inline void elbp_(InputArray _src, OutputArray _dst, int radius, int neighbors)
{
	//get matrices
	Mat src = _src.getMat();
	// allocate memory for result��˲������ⲿ��_dst�����ڴ�ռ䣬����������Ͷ���int
	_dst.create(src.rows - 2 * radius, src.cols - 2 * radius, CV_32SC1);
	Mat dst = _dst.getMat();
	// zero
	dst.setTo(0);
	for (int n = 0; n<neighbors; n++)
	{
		// sample points ��ȡ��ǰ������
		float x = static_cast<float>(-radius * sin(2.0*CV_PI*n / static_cast<float>(neighbors)));
		float y = static_cast<float>(radius * cos(2.0*CV_PI*n / static_cast<float>(neighbors)));
		// relative indices ��ȡ������ȡ��
		int fx = static_cast<int>(floor(x)); // ����ȡ��
		int fy = static_cast<int>(floor(y));
		int cx = static_cast<int>(ceil(x));  // ����ȡ��
		int cy = static_cast<int>(ceil(y));
		// fractional part С������
		float tx = x - fx;
		float ty = y - fy;
		// set interpolation weights �����ĸ���Ĳ�ֵȨ��
		float w1 = (1 - tx) * (1 - ty);
		float w2 = tx  * (1 - ty);
		float w3 = (1 - tx) *      ty;
		float w4 = tx  *      ty;
		// iterate through your data ѭ������ͼ������
		for (int i = radius; i < src.rows - radius; i++)
		{
			for (int j = radius; j < src.cols - radius; j++)
			{
				// calculate interpolated value �����ֵ��t��ʾ�ĸ����Ȩ�غ�
				float t = static_cast<float> (w1*src.at<_Tp>(i + fy, j + fx) +
					w2*src.at<_Tp>(i + fy, j + cx) +
					w3*src.at<_Tp>(i + cy, j + fx) +
					w4*src.at<_Tp>(i + cy, j + cx));
				// floating point precision, so check some machine-dependent epsilon
				// std::numeric_limits<float>::epsilon()=1.192092896e-07F
				// ��t>=src(i,j)��ʱ��ȡ1����������Ӧ����λ
				dst.at<int>(i - radius, j - radius) += ((t > src.at<_Tp>(i, j)) ||
					(std::abs(t - src.at<_Tp>(i, j)) < std::numeric_limits<float>::epsilon())) << n;
			}
		}
	}
}

// �ⲿ�ӿڣ����ݲ�ͬ���������͵���ģ�庯��
static void elbp(InputArray src, OutputArray dst, int radius, int neighbors)
{
	int type = src.type();
	switch (type) {
	case CV_8SC1:   elbp_<char>(src, dst, radius, neighbors); break;
	case CV_8UC1:   elbp_<unsigned char>(src, dst, radius, neighbors); break;
	case CV_16SC1:  elbp_<short>(src, dst, radius, neighbors); break;
	case CV_16UC1:  elbp_<unsigned short>(src, dst, radius, neighbors); break;
	case CV_32SC1:  elbp_<int>(src, dst, radius, neighbors); break;
	case CV_32FC1:  elbp_<float>(src, dst, radius, neighbors); break;
	case CV_64FC1:  elbp_<double>(src, dst, radius, neighbors); break;
	default:
		string error_msg = format("Using Circle Local Binary Patterns for feature extraction only works                                     on single-channel images (given %d). Please pass the image data as a grayscale image!", type);
		CV_Error(CV_StsNotImplemented, error_msg);
		break;
	}
}


void LBPH:: predict(InputArray _src, int &minClass, double &minDist) const {
	int _grid_x = 8;
	int _grid_y = 8;
	int _radius = 1;
	int _neighbors = 8;
	double _threshold = 2100.0;
	if (_histograms.empty()) {
		// throw error if no data (or simply return -1?)
		string error_message = "This LBPH model is not computed yet. Did you call the train method?";
		CV_Error(CV_StsBadArg, error_message);

	}
	Mat src = _src.getMat();
	// get the spatial histogram from input image
	Mat lbp_image;
	elbp(src, lbp_image, _radius, _neighbors);
	Mat query = spatial_histogram(
		lbp_image, /* lbp_image */
		static_cast<int>(std::pow(2.0, static_cast<double>(_neighbors))), /* number of possible patterns */
		_grid_x, /* grid size x */
		_grid_y, /* grid size y */
		true /* normed histograms */);
	// find 1-nearest neighbor
	minDist = DBL_MAX;
	minClass = -1;
	for (size_t sampleIdx = 0; sampleIdx < _histograms.size(); sampleIdx++) {
		double dist = compareHist(_histograms[sampleIdx], query, CV_COMP_CHISQR);
		if ((dist < minDist) && (dist < _threshold)) {
			minDist = dist;
			minClass = _labels.at<int>((int)sampleIdx);

		}

	}
}

