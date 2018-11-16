#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <iostream>
#include <fstream>

#include "Detect_Recognize.h"
#include "LBPH.h"
#include "Cv310Text.h"

const cv::String    WINDOW_NAME("Camera video");
const cv::String    CASCADE_FILE("haarcascade_frontalface_default.xml");
std::string CSVFN = std::string("TrainSample.txt");

std::vector<cv::Mat> images;//两个容器images,labels来存放图像数据和对应的标签
std::vector<int> labels;

void read_csv(const std::string &filename, std::vector<cv::Mat> &images, std::vector<int> &labels);

int main(int argc, char** argv)
{
	//读取CSV文件	
	try
	{
		//读取训练图像和类别标签
		read_csv(CSVFN, images, labels);
	}
	catch (cv::Exception &e)
	{
		std::cerr << "Error opening file " << CSVFN << ". Reason: " << e.msg << std::endl;
		exit(1);
	}

	//如果没有读到足够的图片，就退出
	if (images.size() <= 2)
	{
		std::string error_message = "This demo needs at least 2 images to work.";
		CV_Error(CV_StsError, error_message);
	}

	LBPH model;
	model.train(images, labels);

	// 打开摄像头
	cv::VideoCapture camera(0);
	if (!camera.isOpened()) {
		fprintf(stderr, "Error getting camera...\n");
		exit(1);
	}
	//调整窗口大小
	cv::namedWindow(WINDOW_NAME, cv::WINDOW_KEEPRATIO | cv::WINDOW_AUTOSIZE);
	//视频帧处理实现人脸检测与识别
	Detect_Recognize detector(CASCADE_FILE, camera);
	cv::Mat frame;
	double fps = 0, time_per_frame;
	std::vector<cv::Rect> tface;
	while (true)
	{
		auto start = cv::getCPUTickCount();
		detector >> frame;
		auto end = cv::getCPUTickCount();

		time_per_frame = (end - start) / cv::getTickFrequency();
		fps = (15 * fps + (1 / time_per_frame)) / 16;

		printf("Time per frame: %3.3f\tFPS: %3.3f\n", time_per_frame, fps);

		//cv::Mat gray, m_res;
		//cv::Size ResImgSiz = cv::Size(100, 100);
		/*for (int i = 0; i < m_trackedFace.size(); i++)
		{
			cv::cvtColor(resizedFrame(m_trackedFace[i]), gray, cv::COLOR_BGR2GRAY);
			if (gray.rows == 0 || gray.cols == 0)
			{
				m_trackedFace.erase(m_trackedFace.begin() + i);
				i = i - 1;
				continue;
			}
			cv::resize(gray, m_res, ResImgSiz, CV_INTER_NN);
			cv::imshow("result", m_res);
			//cv::imwrite("../15.jpg", m_res);
			Test.push_back(m_res);
		}*/

		int predictedLabel = -1;
		double predicted_confidence = 0.0;
		if (detector.isFaceFound())
		{
			//std::vector<cv::Mat> testface = detector.TestFaces();
			cv::Mat gray, img;
			cv::Size ResImgSiz = cv::Size(100, 100);
			tface = detector.face();
			for (int i = 0; i < detector.faceNum(); i++)
			{
				//cv::Mat img = testface[i];
				cv::cvtColor(frame(tface[i]), gray, cv::COLOR_BGR2GRAY);
				cv::resize(gray, img, ResImgSiz, CV_INTER_NN);

				model.predict(img, predictedLabel, predicted_confidence);
				if (predicted_confidence > 100)
					putText(frame, "NO FOUND", Point(tface[i].x + tface[i].width / 2, tface[i].y), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(255, 0, 0));
				else if (predictedLabel == 1)
					putText(frame, "zaoyifan", Point(tface[i].x+ tface[i].width/2, tface[i].y), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(255, 0, 0));
				else if (predictedLabel == 2)
					putText(frame, "qiulinwei", Point(tface[i].x + tface[i].width / 2, tface[i].y), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255));
				
				cv::rectangle(frame, tface[i], cv::Scalar(0, 255, 0), 2);
				//cv::circle(frame, detector.facePosition(), 30, cv::Scalar(0, 255, 0));
			}
			tface.clear();
		}
		
		cv::imshow(WINDOW_NAME, frame);
		if (cv::waitKey(25) == 27) break;
	}

	return 0;
}


//读取文件中的图像数据和类别，存入images和labels这两个容器
void read_csv(const std::string &filename, std::vector<cv::Mat> &images, std::vector<int> &labels)
{
	std::ifstream file(filename.c_str(), std::ifstream::in);
	if (!file)
	{
		cv::String error_message("No valid input file was given.");
		CV_Error(CV_StsBadArg, error_message);
	}

	std::string line, path, classlabel;
	while (getline(file, line))
	{
		std::stringstream liness(line);
		getline(liness, path, ';');  //遇到分号就结束
		getline(liness, classlabel);     //继续从分号后面开始，遇到换行结束
		if (!path.empty() && !classlabel.empty())
		{
			images.push_back(cv::imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}