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
	// 计算LBPM的空间直方图分布，得到一个一维向量
	// src为LBPM是通过olbp或者elbp计算得到的
	// numPatterns为计算LBP的模式数目，一般为2的幂
	// grid_x和grid_y分别为每行或每列的block个数
	// normed为是否进行归一化处理
	Mat src = _src.getMat();
	// allocate memory for the spatial histogram为LBPH分配内存空间
	Mat result = Mat::zeros(grid_x * grid_y, numPatterns, CV_32FC1);
	// return matrix with zeros if no data was given，如果没有输入数据，返回的是0
	if (src.empty())
		return result.reshape(1, 1);
	// calculate LBP patch size block的尺寸
	int width = src.cols / grid_x;
	int height = src.rows / grid_y;
	// initial result_row 初始化结果行
	int resultRowIdx = 0;
	// iterate through grid
	for (int i = 0; i < grid_y; i++)
	{
		for (int j = 0; j < grid_x; j++)
		{
			// 获取指定区域
			Mat src_cell = Mat(src, Range(i*height, (i + 1)*height), Range(j*width, (j + 1)*width));
			// 计算指定区域的直方图
			Mat cell_hist = histc(src_cell, 0, (numPatterns - 1), true);
			// copy to the result matrix 将计算得到的结果拷贝到每一行
			Mat result_row = result.row(resultRowIdx);
			cell_hist.reshape(1, 1).convertTo(result_row, CV_32FC1);
			// increase row count in result matrix
			resultRowIdx++;
		}
	}
	// return result as reshaped feature vector
	return result.reshape(1, 1);
}


// src为输入图像，dst为输出图像，radius为半径，neighbor为计算当前点LBP所需的邻域像素点数，也就是样本点个数
template <typename _Tp> static // 模板函数，根据不同的原始数据类型得到不同的结果
inline void elbp_(InputArray _src, OutputArray _dst, int radius, int neighbors)
{
	//get matrices
	Mat src = _src.getMat();
	// allocate memory for result因此不用在外部给_dst分配内存空间，输出数据类型都是int
	_dst.create(src.rows - 2 * radius, src.cols - 2 * radius, CV_32SC1);
	Mat dst = _dst.getMat();
	// zero
	dst.setTo(0);
	for (int n = 0; n<neighbors; n++)
	{
		// sample points 获取当前采样点
		float x = static_cast<float>(-radius * sin(2.0*CV_PI*n / static_cast<float>(neighbors)));
		float y = static_cast<float>(radius * cos(2.0*CV_PI*n / static_cast<float>(neighbors)));
		// relative indices 下取整和上取整
		int fx = static_cast<int>(floor(x)); // 向下取整
		int fy = static_cast<int>(floor(y));
		int cx = static_cast<int>(ceil(x));  // 向上取整
		int cy = static_cast<int>(ceil(y));
		// fractional part 小数部分
		float tx = x - fx;
		float ty = y - fy;
		// set interpolation weights 设置四个点的插值权重
		float w1 = (1 - tx) * (1 - ty);
		float w2 = tx  * (1 - ty);
		float w3 = (1 - tx) *      ty;
		float w4 = tx  *      ty;
		// iterate through your data 循环处理图像数据
		for (int i = radius; i < src.rows - radius; i++)
		{
			for (int j = radius; j < src.cols - radius; j++)
			{
				// calculate interpolated value 计算插值，t表示四个点的权重和
				float t = static_cast<float> (w1*src.at<_Tp>(i + fy, j + fx) +
					w2*src.at<_Tp>(i + fy, j + cx) +
					w3*src.at<_Tp>(i + cy, j + fx) +
					w4*src.at<_Tp>(i + cy, j + cx));
				// floating point precision, so check some machine-dependent epsilon
				// std::numeric_limits<float>::epsilon()=1.192092896e-07F
				// 当t>=src(i,j)的时候取1，并进行相应的移位
				dst.at<int>(i - radius, j - radius) += ((t > src.at<_Tp>(i, j)) ||
					(std::abs(t - src.at<_Tp>(i, j)) < std::numeric_limits<float>::epsilon())) << n;
			}
		}
	}
}

// 外部接口，根据不同的数据类型调用模板函数
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

