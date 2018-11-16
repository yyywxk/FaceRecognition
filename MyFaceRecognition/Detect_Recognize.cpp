#include "Detect_Recognize.h"
#include <iostream>
#include <opencv2\imgproc.hpp>

const double Detect_Recognize::TICK_FREQUENCY = cv::getTickFrequency();

Detect_Recognize::Detect_Recognize(const std::string cascadeFilePath, cv::VideoCapture &videoCapture)
{
	setFaceCascade(cascadeFilePath);
	setVideoCapture(videoCapture);
}

Detect_Recognize::~Detect_Recognize()
{
	if (m_faceCascade != NULL) {
		delete m_faceCascade;
	}
}

void Detect_Recognize::setVideoCapture(cv::VideoCapture &videoCapture)
{
	m_videoCapture = &videoCapture;
}

cv::VideoCapture *Detect_Recognize::videoCapture() const
{
	return m_videoCapture;
}

void Detect_Recognize::setFaceCascade(const std::string cascadeFilePath)
{
	if (m_faceCascade == NULL) {
		m_faceCascade = new cv::CascadeClassifier(cascadeFilePath);
	}
	else {
		m_faceCascade->load(cascadeFilePath);
	}

	if (m_faceCascade->empty()) {
		std::cerr << "Error creating cascade classifier. Make sure the file" << std::endl
			<< cascadeFilePath << " exists." << std::endl;
	}
}

cv::CascadeClassifier *Detect_Recognize::faceCascade() const
{
	return m_faceCascade;
}

void Detect_Recognize::setResizedWidth(const int width)
{
	m_resizedWidth = std::max(width, 1);
}

int Detect_Recognize::resizedWidth() const
{
	return m_resizedWidth;
}

bool Detect_Recognize::isFaceFound() const
{
	return m_foundFace;
}

cv::Point Detect_Recognize::centerOfRect(const cv::Rect &rect) const
{
	return cv::Point(rect.x + rect.width / 2, rect.y + rect.height / 2);
}

std::vector<cv::Rect> Detect_Recognize::face() const
{
	std::vector<cv::Rect> faceRect = m_trackedFace;
	for (int i = 0; i < m_faceNum; i++)
	{
		faceRect[i].x = (int)(faceRect[i].x / m_scale);
		faceRect[i].y = (int)(faceRect[i].y / m_scale);
		faceRect[i].width = (int)(faceRect[i].width / m_scale);
		faceRect[i].height = (int)(faceRect[i].height / m_scale);
	}
	return faceRect;
}

std::vector<cv::Point> Detect_Recognize::facePosition() const
{
	std::vector<cv::Point> facePos;
	for (int i = 0; i < m_faceNum; i++)
	{
		facePos[i].x = (int)(m_facePosition[i].x / m_scale);
		facePos[i].y = (int)(m_facePosition[i].y / m_scale);
	}
	return facePos;
}

void Detect_Recognize::setTemplateMatchingMaxDuration(const double s)
{
	m_templateMatchingMaxDuration = s;
}

double Detect_Recognize::templateMatchingMaxDuration() const
{
	return m_templateMatchingMaxDuration;
}

cv::Rect Detect_Recognize::doubleRectSize(const cv::Rect &inputRect, const cv::Rect &frameSize) const
{
	cv::Rect outputRect;
	// Double rect size
	outputRect.width = inputRect.width * 2;
	outputRect.height = inputRect.height * 2;

	// Center rect around original center
	outputRect.x = inputRect.x - inputRect.width / 2;
	outputRect.y = inputRect.y - inputRect.height / 2;

	// Handle edge cases
	if (outputRect.x < frameSize.x) {
		outputRect.width += outputRect.x;
		outputRect.x = frameSize.x;
	}
	if (outputRect.y < frameSize.y) {
		outputRect.height += outputRect.y;
		outputRect.y = frameSize.y;
	}

	if (outputRect.x + outputRect.width > frameSize.width) {
		outputRect.width = frameSize.width - outputRect.x;
	}
	if (outputRect.y + outputRect.height > frameSize.height) {
		outputRect.height = frameSize.height - outputRect.y;
	}

	return outputRect;
}

cv::Rect Detect_Recognize::biggestFace(std::vector<cv::Rect> &faces) const
{
	assert(!faces.empty());

	cv::Rect *biggest = &faces[0];
	for (auto &face : faces) {
		if (face.area() < biggest->area())
			biggest = &face;
	}
	return *biggest;
}

/*
* Face template is small patch in the middle of detected face.
*/
cv::Mat Detect_Recognize::getFaceTemplate(const cv::Mat &frame, cv::Rect face)
{
	face.x += face.width / 4;
	face.y += face.height / 4;
	face.width /= 2;
	face.height /= 2;

	cv::Mat faceTemplate = frame(face).clone();
	return faceTemplate;
}

int Detect_Recognize::faceNum() const
{
	return m_faceNum;
}

void Detect_Recognize::detectFaceAllSizes(const cv::Mat &frame)
{
	// Minimum face size is 1/5th of screen height
	// Maximum face size is 2/3rds of screen height
	m_faceCascade->detectMultiScale(frame, m_allFaces, 1.1, 3, 0,
		cv::Size(frame.rows / 5, frame.rows / 5),
		cv::Size(frame.rows * 2 / 3, frame.rows * 2 / 3));

	m_faceNum = m_allFaces.size();

	if (m_allFaces.empty()) return;

	m_foundFace = true;

	m_templateMatchingRunning = false;

	m_trackedFace.clear();
	m_faceTemplate.clear();
	m_faceRoi.clear();
	m_facePosition.clear();
	m_templateMatchingCurrentTime.clear();
	m_templateMatchingStartTime.clear();
	// Locate biggest face
	//m_trackedFace = biggestFace(m_allFaces);
	m_trackedFace = m_allFaces;

	for (int i = 0; i < m_faceNum; i++)
	{	
		// Copy face template
		m_faceTemplate.push_back(getFaceTemplate(frame, m_trackedFace[i]));

		// Calculate roi
		m_faceRoi.push_back(doubleRectSize(m_trackedFace[i], cv::Rect(0, 0, frame.cols, frame.rows)));

		// Update face position
		m_facePosition.push_back(centerOfRect(m_trackedFace[i]));

		m_templateMatchingCurrentTime.push_back(0);
		m_templateMatchingStartTime.push_back(0);
	}
}

void Detect_Recognize::detectFaceAroundRoi(const cv::Mat &frame)
{
	std::vector<cv::Rect>   t_allFaces;
	for (int i = 0; i < m_faceNum; i++)
	{
		// Detect faces sized +/-20% off biggest face in previous search
		m_faceCascade->detectMultiScale(frame(m_faceRoi[i]), t_allFaces, 1.1, 3, 0,
			cv::Size(m_trackedFace[i].width * 8 / 10, m_trackedFace[i].height * 8 / 10),
			cv::Size(m_trackedFace[i].width * 12 / 10, m_trackedFace[i].width * 12 / 10));

		if (t_allFaces.empty())
		{
			// Activate template matching if not already started and start timer
			m_templateMatchingRunning = true;
			if (m_templateMatchingStartTime[i] == 0)
				m_templateMatchingStartTime[i] = cv::getTickCount();
			return;
		}

		// Turn off template matching if running and reset timer
		m_templateMatchingRunning = false;
		m_templateMatchingCurrentTime[i] = m_templateMatchingStartTime[i] = 0;

		// Get detected face
		m_trackedFace[i] = biggestFace(t_allFaces);

		// Add roi offset to face
		m_trackedFace[i].x += m_faceRoi[i].x;
		m_trackedFace[i].y += m_faceRoi[i].y;

		// Get face template
		m_faceTemplate[i] = getFaceTemplate(frame, m_trackedFace[i]);

		// Calculate roi
		m_faceRoi[i] = doubleRectSize(m_trackedFace[i], cv::Rect(0, 0, frame.cols, frame.rows));

		// Update face position
		m_facePosition[i] = centerOfRect(m_trackedFace[i]);
	}
}

void Detect_Recognize::detectFacesTemplateMatching(const cv::Mat &frame)
{
	for (int i = 0; i < m_faceNum; i++)
	{
		// Calculate duration of template matching
		m_templateMatchingCurrentTime[i] = cv::getTickCount();
		double duration = (double)(m_templateMatchingCurrentTime[i] - m_templateMatchingStartTime[i]) / TICK_FREQUENCY;

		// If template matching lasts for more than 2 seconds face is possibly lost
		// so disable it and redetect using cascades
		if (duration > m_templateMatchingMaxDuration) {
			m_foundFace = false;
			m_templateMatchingRunning = false;
			m_templateMatchingStartTime[i] = m_templateMatchingCurrentTime[i] = 0;
			m_facePosition[i].x = m_facePosition[i].y = 0;
			m_trackedFace[i].x = m_trackedFace[i].y = m_trackedFace[i].width = m_trackedFace[i].height = 0;
			return;
		}

		// Edge case when face exits frame while 
		if (m_faceTemplate[i].rows * m_faceTemplate[i].cols == 0 || m_faceTemplate[i].rows <= 1 || m_faceTemplate[i].cols <= 1) {
			m_foundFace = false;
			m_templateMatchingRunning = false;
			m_templateMatchingStartTime[i] = m_templateMatchingCurrentTime[i] = 0;
			m_facePosition[i].x = m_facePosition[i].y = 0;
			m_trackedFace[i].x = m_trackedFace[i].y = m_trackedFace[i].width = m_trackedFace[i].height = 0;
			return;
		}

		// Template matching with last known face 
		//cv::matchTemplate(frame(m_faceRoi), m_faceTemplate, m_matchingResult, CV_TM_CCOEFF);
		cv::matchTemplate(frame(m_faceRoi[i]), m_faceTemplate[i], m_matchingResult, CV_TM_SQDIFF_NORMED);
		cv::normalize(m_matchingResult, m_matchingResult, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
		double min, max;
		cv::Point minLoc, maxLoc;
		cv::minMaxLoc(m_matchingResult, &min, &max, &minLoc, &maxLoc);

		// Add roi offset to face position
		minLoc.x += m_faceRoi[i].x;
		minLoc.y += m_faceRoi[i].y;

		// Get detected face
		//m_trackedFace = cv::Rect(maxLoc.x, maxLoc.y, m_trackedFace.width, m_trackedFace.height);
		m_trackedFace[i] = cv::Rect(minLoc.x, minLoc.y, m_faceTemplate[i].cols, m_faceTemplate[i].rows);
		m_trackedFace[i] = doubleRectSize(m_trackedFace[i], cv::Rect(0, 0, frame.cols, frame.rows));

		// Get new face template
		m_faceTemplate[i] = getFaceTemplate(frame, m_trackedFace[i]);

		// Calculate face roi
		m_faceRoi[i] = doubleRectSize(m_trackedFace[i], cv::Rect(0, 0, frame.cols, frame.rows));

		// Update face position
		m_facePosition[i] = centerOfRect(m_trackedFace[i]);
	}
}

bool Detect_Recognize::isAround(const cv::Rect &rect) const
{
	for (int i = 0; i < m_faceNum; i++)
	{
		cv::Point centerFace, centerRect;
		centerFace = centerOfRect(m_trackedFace[i]);
		centerRect = centerOfRect(rect);
		if (abs(centerFace.x - centerRect.x) < ((m_trackedFace[i].width + rect.width)/ 2))
		{
			if (abs(centerFace.y - centerRect.y) < ((m_trackedFace[i].height + rect.height) / 2))
			{
				return true;
			}
		}
	}
	return false;
}

void Detect_Recognize::detectFacesOther(const cv::Mat &frame)
{
	
	// Minimum face size is 1/5th of screen height
	// Maximum face size is 2/3rds of screen height
	std::vector<cv::Rect>   allFaces;

	m_faceCascade->detectMultiScale(frame, allFaces, 1.1, 3, 0,
		cv::Size(frame.rows / 5, frame.rows / 5),
		cv::Size(frame.rows * 2 / 3, frame.rows * 2 / 3));

	size_t faceNum = allFaces.size();

	if (allFaces.empty()) return;

	m_foundFace = true;

	m_templateMatchingRunning = false;

	// Locate biggest face
	//m_trackedFace = biggestFace(m_allFaces);

	for (int i = 0; i < faceNum; i++)
	{
		if (!isAround(allFaces[i]))
		{
			m_trackedFace.push_back(allFaces[i]);
			// Copy face template
			m_faceTemplate.push_back(getFaceTemplate(frame, allFaces[i]));

			// Calculate roi
			m_faceRoi.push_back(doubleRectSize(allFaces[i], cv::Rect(0, 0, frame.cols, frame.rows)));

			// Update face position
			m_facePosition.push_back(centerOfRect(allFaces[i]));

			m_templateMatchingCurrentTime.push_back(0);
			m_templateMatchingStartTime.push_back(0);
		}
	}
	m_faceNum = m_trackedFace.size();
}

void Detect_Recognize::getFrameAndDetect(cv::Mat &frame)
{
	*m_videoCapture >> frame;

	// Downscale frame to m_resizedWidth width - keep aspect ratio
	m_scale = (double)std::min(m_resizedWidth, frame.cols) / frame.cols;
	cv::Size resizedFrameSize = cv::Size((int)(m_scale*frame.cols), (int)(m_scale*frame.rows));

	cv::resize(frame, resizedFrame, resizedFrameSize);

	if (!m_foundFace) {
		detectFaceAllSizes(resizedFrame); // Detect using cascades over whole image
		flag = false;
	}		
	else {
		if (flag == false)
		{
			m_StartTime = cv::getTickCount();
			flag = true;
		}
		detectFaceAroundRoi(resizedFrame); // Detect using cascades only in ROI
		if (m_templateMatchingRunning) {
			detectFacesTemplateMatching(resizedFrame); // Detect using template matching
		}
		m_CurrentTime = cv::getTickCount();
		if ((m_CurrentTime - m_StartTime) > 10)
		{
			//m_foundFace = false;
			detectFacesOther(resizedFrame);
			flag = false;
		}
	}
}

void Detect_Recognize::operator >> (cv::Mat &frame)
{
	return this->Detect_and_Recognize(frame);
}

void Detect_Recognize::Detect_and_Recognize(cv::Mat &frame)
{
	getFrameAndDetect(frame);

	cv::Mat gray, m_res;
	cv::Size ResImgSiz = cv::Size(100, 100);
	Test.clear();
	for (int i = 0; i < m_trackedFace.size(); i++)
	{
		cv::cvtColor(resizedFrame(m_trackedFace[i]), gray, cv::COLOR_BGR2GRAY);
		if (gray.rows == 0 || gray.cols == 0)
		{
			m_trackedFace.erase(m_trackedFace.begin()+i);
			i = i - 1;
			continue;
		}
		cv::resize(gray, m_res, ResImgSiz, CV_INTER_NN);
		//cv::imshow("result", m_res);
		//cv::imwrite("../15.jpg", m_res);
		Test.push_back(m_res);
	}
	m_faceNum = Test.size();
}

std::vector<cv::Mat> Detect_Recognize::TestFaces() const
{
	return Test;
}