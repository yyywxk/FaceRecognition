#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cassert>
#include <vector>
#include <numeric>
#include <algorithm>
#include <functional>
#include <iterator>

using namespace cv;
using namespace std;


class LBPH
{
private:
	int _grid_x;
	int _grid_y;
	int _radius;
	int _neighbors;
	double _threshold;

	vector<Mat> _histograms;
	Mat _labels;
public:
	// Computes a LBPH model with images in src and
	// corresponding labels in labels, possibly preserving
	// old model data.
	void train(InputArrayOfArrays src, InputArray labels, bool preserveData);

	//Initializes this LBPH Model.The current implementation is rather fixed
	// as it uses the Extended Local Binary Patterns per default.
	//
	// radius, neighbors are used in the local binary patterns creation.
	// grid_x, grid_y control the grid size of the spatial histograms.
	LBPH(int radius_ = 1, int neighbors_ = 8,
			int gridx = 8, int gridy = 8,
			double threshold = DBL_MAX) :
		_grid_x(gridx),
		_grid_y(gridy),
		_radius(radius_),
		_neighbors(neighbors_),
		_threshold(threshold) {}

	// Initializes and computes this LBPH Model. The current implementation is
	// rather fixed as it uses the Extended Local Binary Patterns per default.
	//
	// (radius=1), (neighbors=8) are used in the local binary patterns creation.
	// (grid_x=8), (grid_y=8) controls the grid size of the spatial histograms.
	LBPH(InputArrayOfArrays src,
		InputArray labels,
		int radius_ = 1, int neighbors_ = 8,
		int gridx = 8, int gridy = 8,
		double threshold = DBL_MAX) :
		_grid_x(gridx),
		_grid_y(gridy),
		_radius(radius_),
		_neighbors(neighbors_),
		_threshold(threshold) {
		train(src, labels);
	}

	~LBPH() { }

	// Computes a LBPH model with images in src and
	// corresponding labels in labels.
	void train(InputArrayOfArrays src, InputArray labels);

	// Updates this LBPH model with images in src and
	// corresponding labels in labels.
	void update(InputArrayOfArrays src, InputArray labels);

	// Predicts the label of a query image in src.
	int predict(InputArray src) const;

	// Predicts the label and confidence for a given sample.
	void predict(InputArray _src, int &label, double &dist) const;

	// Getter functions.
	int neighbors() const { return _neighbors; }
	int radius() const { return _radius; }
	int grid_x() const { return _grid_x; }
	int grid_y() const { return _grid_y; }

};