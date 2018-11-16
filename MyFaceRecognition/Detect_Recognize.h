#pragma once

#include <opencv2\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\objdetect\objdetect.hpp>

class Detect_Recognize 
{
public:
	Detect_Recognize(const std::string cascadeFilePath, cv::VideoCapture &videoCapture);
	~Detect_Recognize();

	void					Detect_and_Recognize(cv::Mat &frame);

	void					getFrameAndDetect(cv::Mat &frame);
	//std::vector<cv::Point>	operator >> (cv::Mat &frame);
	void					operator >> (cv::Mat &frame);
	void                    setVideoCapture(cv::VideoCapture &videoCapture);
	cv::VideoCapture*       videoCapture() const;
	void                    setFaceCascade(const std::string cascadeFilePath);
	cv::CascadeClassifier*  faceCascade() const;
	void                    setResizedWidth(const int width);
	int                     resizedWidth() const;
	bool					isFaceFound() const;
	//cv::Rect                face() const;
	std::vector<cv::Rect>   face() const;
	//cv::Point               facePosition() const;
	std::vector<cv::Point>  facePosition() const;
	void                    setTemplateMatchingMaxDuration(const double s);
	double                  templateMatchingMaxDuration() const;
	int						faceNum() const;
	std::vector<cv::Mat>	TestFaces() const;

private:
	static const double     TICK_FREQUENCY;

	cv::VideoCapture*       m_videoCapture = NULL;
	cv::CascadeClassifier*  m_faceCascade = NULL;
	std::vector<cv::Rect>   m_allFaces;
	//cv::Rect                m_trackedFace;
	std::vector<cv::Rect>   m_trackedFace;
	//cv::Rect                m_faceRoi;
	std::vector<cv::Rect>   m_faceRoi;
	//cv::Mat                 m_faceTemplate;
	std::vector<cv::Mat>	m_faceTemplate;
	cv::Mat                 m_matchingResult;
	bool                    m_templateMatchingRunning = false;
	std::vector<int64>      m_templateMatchingStartTime;
	std::vector<int64>      m_templateMatchingCurrentTime;
	bool                    m_foundFace = false;
	double                  m_scale;
	int                     m_resizedWidth = 320;
	std::vector<cv::Point>  m_facePosition;
	double                  m_templateMatchingMaxDuration = 3;
	size_t					m_faceNum = 0;
	int64                   m_StartTime = 0;
	int64                   m_CurrentTime = 0;
	bool					flag = false;
	cv::Mat                 resizedFrame;
	std::vector<cv::Mat>	Test;

	cv::Rect    doubleRectSize(const cv::Rect &inputRect, const cv::Rect &frameSize) const;
	cv::Rect    biggestFace(std::vector<cv::Rect> &faces) const;
	cv::Point   centerOfRect(const cv::Rect &rect) const;
	cv::Mat     getFaceTemplate(const cv::Mat &frame, cv::Rect face);
	void        detectFaceAllSizes(const cv::Mat &frame);
	void        detectFaceAroundRoi(const cv::Mat &frame);
	void        detectFacesTemplateMatching(const cv::Mat &frame);
	void        detectFacesOther(const cv::Mat &frame);
	bool		isAround(const cv::Rect &rect) const;
};