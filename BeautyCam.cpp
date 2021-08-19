#include "BeautyCam.h"
BeautyCam *BeautyCam::m_pIntance=nullptr;

BeautyCam::BeautyCam()
{
	//��ʼ����̬����
	m_pIntance = this;
	//��ʼ��������
	initMainImgUI();
}
void BeautyCam::initMainImgUI()
{
	string path;
	cout << "������ͼƬ�ľ���·����" << endl;
	cin>> path;
	
	namedWindow("BeautyCam", WINDOW_AUTOSIZE);
	//string path = "33.jpg";


	m_MainImg =imread(path);
	imshow("src", m_MainImg);

	//�����������68��
	m_vecFaceData = dectectFace68(path);

	int max_value = 100;
	int con_value = 100;
	int lignhtnesss = 50;
	int contrast = 50;
	int bigeyeval = 0;
	int faceval = 0;
	int beautyval = 0;
	createTrackbar("����", "BeautyCam", &lignhtnesss, max_value, on_lightness, (void*)(&m_MainImg));
	createTrackbar("�Աȶ�", "BeautyCam", &contrast, max_value, on_contrast, (void*)(&m_MainImg));
	createTrackbar("����", "BeautyCam", &bigeyeval, 60, on_BigEye, (void*)(&m_MainImg));
	createTrackbar("����", "BeautyCam", &faceval, 70, on_thinFace, (void*)(&m_MainImg));
	createTrackbar("����", "BeautyCam", &beautyval, 200, on_beautyFace, (void*)(&m_MainImg));

	on_lightness(50, (void*)(&m_MainImg));
	//imshow("BeautyCam", m_MainImg);
}
void BeautyCam::browersBut_callback(int state, void *data)
{
	cout << "state" << state << endl;
}
void BeautyCam::on_lightness(int b, void*userdata)
{
	Mat img = *((Mat *)userdata);
	Mat m = Mat::zeros(img.size(), img.type());
	Mat dst = Mat::zeros(img.size(), img.type());
	m = Scalar(b, b, b);
	//add(img, m, dst);
	addWeighted(img, 1.0, m, 0, b, dst);
	imshow("BeautyCam", dst);
}
void BeautyCam::on_contrast(int b, void*userdata)
{
	Mat img = *((Mat *)userdata);
	Mat m = Mat::zeros(img.size(), img.type());
	Mat dst = Mat::zeros(img.size(), img.type());
	m = Scalar(b, b, b);
	double con = b / 100.0;
	addWeighted(img, con, m, 0, 0, dst);
	imshow("BeautyCam", dst);
}
void BeautyCam::on_BigEye(int b, void*userdata)
{
	Mat src = *((Mat *)userdata);
	Mat dst = src.clone();
	for (auto points_vec : m_pIntance->m_vecFaceData)
	{
		Point2f left_landmark = points_vec[38];
		Point2f	left_landmark_down = points_vec[27];

		Point2f	right_landmark = points_vec[44];
		Point2f	right_landmark_down = points_vec[27];

		Point2f	endPt = points_vec[30];

		//# �����4���㵽��6����ľ�����Ϊ����
		/*float r_left = sqrt(
			(left_landmark.x - left_landmark_down.x) * (left_landmark.x - left_landmark_down.x) +
			(left_landmark.y - left_landmark_down.y) * (left_landmark.y - left_landmark_down.y));
		cout << "���۾���:" << r_left;*/
		float r_left = b;

		//	# �����14���㵽��16����ľ�����Ϊ����
		//float	r_right = sqrt(
		//	(right_landmark.x - right_landmark_down.x) * (right_landmark.x - right_landmark_down.x) +
		//	(right_landmark.y - right_landmark_down.y) * (right_landmark.y - right_landmark_down.y));
		//cout << "���۾���:" << r_right;
		float r_right = b;
		//	# ����                     
		m_pIntance->LocalTranslationWarp_Eye(src, dst, left_landmark.x, left_landmark.y, endPt.x, endPt.y, r_left);
		//	# ����
		m_pIntance->LocalTranslationWarp_Eye(src, dst, right_landmark.x, right_landmark.y, endPt.x, endPt.y, r_right);

	}
	imshow("BeautyCam", dst);
}
std::vector<std::vector<Point2f>> BeautyCam::dectectFace68(const string &path)
{
	std::vector<std::vector<Point2f>>  rets;
	//����ͼƬ·��
	array2d<rgb_pixel> img;
	load_image(img, path.c_str());
	//�������������
	frontal_face_detector detector = get_frontal_face_detector();
	std::vector<dlib::rectangle> dets = detector(img);

	for (auto var : dets)
	{
		//�ؼ�������
		shape_predictor sp;
		deserialize("shape_predictor_68_face_landmarks.dat") >> sp;
		//����shape���󱣴����68���ؼ���
		full_object_detection shape = sp(img, var);
		//�洢�ļ�
		ofstream out("face_detector.txt");
		//��ȡ�ؼ��㵽������
		std::vector<Point2f> points_vec;
		for (int i = 0; i < shape.num_parts(); ++i)
		{
			auto a = shape.part(i);
			out << a.x() << " " << a.y() << " ";
			Point2f ff(a.x(), a.y());
			points_vec.push_back(ff);
		}
		rets.push_back(points_vec);
	}
	cout << "����������:" <<dets.size()<<"����������"<< endl;
	return rets;
}
void BeautyCam::LocalTranslationWarp_Eye(Mat &img, Mat &dst, int warpX, int warpY, int endX, int endY, float radius)
{
	//ƽ�ƾ��� 
	float ddradius = radius * radius;
	//����|m-c|^2
	size_t mc = (endX - warpX)*(endX - warpX) + (endY - warpY)*(endY - warpY);
	//���� ͼ��ĸ�  �� ͨ������
	int height = img.rows;
	int width = img.cols;
	int chan = img.channels();

	auto Abs = [&](float f) {
		return f > 0 ? f : -f;
	};

	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			// # ����õ��Ƿ����α�Բ�ķ�Χ֮��
			//# �Ż�����һ����ֱ���ж��ǻ��ڣ�startX, startY)�ľ������
			if ((Abs(i - warpX) > radius) && (Abs(j - warpY) > radius))
				continue;

			float distance = (i - warpX)*(i - warpX) + (j - warpY)*(j - warpY);
			if (distance < ddradius)
			{
				float rnorm = sqrt(distance) / radius;
				float ratio = 1 - (rnorm - 1)*(rnorm - 1)*0.5;
				//ӳ��ԭλ��
				float UX = warpX + ratio * (i - warpX);
				float UY = warpY + ratio * (j - warpY);

				//����˫���Բ�ֵ�õ�UX UY��ֵ
				BilinearInsert(img, dst, UX, UY, i, j);
			}
		}
	}

}
void BeautyCam::BilinearInsert(Mat &src, Mat &dst, float ux, float uy, int i, int j)
{
	auto Abs = [&](float f) {
		return f > 0 ? f : -f;
	};

	int c = src.channels();
	if (c == 3)
	{
		//�洢ͼ��ø�������
		CvPoint2D32f uv;
		CvPoint3D32f f1;
		CvPoint3D32f f2;

		//ȡ����
		int iu = (int)ux;
		int iv = (int)uy;
		uv.x = iu + 1;
		uv.y = iv + 1;

		//stepͼ�������е�ʵ�ʿ��  ����ͨ�����м���(0 , 1 2  ��ͨ��)
		f1.x = ((uchar*)(src.data + src.step*iv))[iu * 3] * (1 - Abs(uv.x - iu)) + \
			((uchar*)(src.data + src.step*iv))[(iu + 1) * 3] * (uv.x - iu);
		f1.y = ((uchar*)(src.data + src.step*iv))[iu * 3 + 1] * (1 - Abs(uv.x - iu)) + \
			((uchar*)(src.data + src.step*iv))[(iu + 1) * 3 + 1] * (uv.x - iu);
		f1.z = ((uchar*)(src.data + src.step*iv))[iu * 3 + 2] * (1 - Abs(uv.x - iu)) + \
			((uchar*)(src.data + src.step*iv))[(iu + 1) * 3 + 2] * (uv.x - iu);


		f2.x = ((uchar*)(src.data + src.step*(iv + 1)))[iu * 3] * (1 - Abs(uv.x - iu)) + \
			((uchar*)(src.data + src.step*(iv + 1)))[(iu + 1) * 3] * (uv.x - iu);
		f2.y = ((uchar*)(src.data + src.step*(iv + 1)))[iu * 3 + 1] * (1 - Abs(uv.x - iu)) + \
			((uchar*)(src.data + src.step*(iv + 1)))[(iu + 1) * 3 + 1] * (uv.x - iu);
		f2.z = ((uchar*)(src.data + src.step*(iv + 1)))[iu * 3 + 2] * (1 - Abs(uv.x - iu)) + \
			((uchar*)(src.data + src.step*(iv + 1)))[(iu + 1) * 3 + 2] * (uv.x - iu);

		((uchar*)(dst.data + dst.step*j))[i * 3] = f1.x*(1 - Abs(uv.y - iv)) + f2.x*(Abs(uv.y - iv));  //����ͨ�����и�ֵ
		((uchar*)(dst.data + dst.step*j))[i * 3 + 1] = f1.y*(1 - Abs(uv.y - iv)) + f2.y*(Abs(uv.y - iv));
		((uchar*)(dst.data + dst.step*j))[i * 3 + 2] = f1.z*(1 - Abs(uv.y - iv)) + f2.z*(Abs(uv.y - iv));

	}
}
void BeautyCam::on_thinFace(int b, void*userdata)
{
	Mat src = *((Mat *)userdata);
	Mat dst = src.clone();
	for (auto points_vec : m_pIntance->m_vecFaceData)
	{
		Point2f endPt = points_vec[34];
		for (int i = 3; i < 15; i = i + 2)
		{
			Point2f start_landmark = points_vec[i];
			Point2f end_landmark = points_vec[i + 2];

			//������������
			/*float dis = sqrt(
				(start_landmark.x - end_landmark.x) * (start_landmark.x - end_landmark.x) +
				(start_landmark.y - end_landmark.y) * (start_landmark.y - end_landmark.y));*/
			float dis = b;
			dst = m_pIntance->LocalTranslationWarp_Face(dst, start_landmark.x, start_landmark.y, endPt.x, endPt.y, dis);

			/*
			//ָ��λ��
			Point2f left_landmark = points_vec[2];
			Point2f	left_landmark_down = points_vec[5];

			Point2f	right_landmark = points_vec[13];
			Point2f	right_landmark_down = points_vec[15];

			Point2f	endPt = points_vec[30];

			//# �����4���㵽��6����ľ�����Ϊ��������
			float r_left = sqrt(
				(left_landmark.x - left_landmark_down.x) * (left_landmark.x - left_landmark_down.x) +
				(left_landmark.y - left_landmark_down.y) * (left_landmark.y - left_landmark_down.y));
			cout << "�����������:" << r_left;


			//	# �����14���㵽��16����ľ�����Ϊ��������
			float	r_right = sqrt(
				(right_landmark.x - right_landmark_down.x) * (right_landmark.x - right_landmark_down.x) +
				(right_landmark.y - right_landmark_down.y) * (right_landmark.y - right_landmark_down.y));
			cout << "�ұ���������:" << r_right;
				//	# �������                         Դͼ��   �����ƶ��� x           y                  ������x       y     ��������
			LocalTranslationWarp(src, dst, left_landmark.x, left_landmark.y, endPt.x, endPt.y, r_left);
			//	# ���ұ���
			LocalTranslationWarp(src, dst, right_landmark.x, right_landmark.y, endPt.x, endPt.y, r_right);
			*/
		}
	}
	imshow("BeautyCam", dst);
}
Mat BeautyCam::LocalTranslationWarp_Face(Mat &img, int warpX, int warpY, int endX, int endY, float radius)
{
	Mat dst = img.clone();
	//ƽ�ƾ��� 
	float ddradius = radius * radius;
	//����|m-c|^2
	size_t mc = (endX - warpX)*(endX - warpX) + (endY - warpY)*(endY - warpY);
	//���� ͼ��ĸ�  �� ͨ������
	int height = img.rows;
	int width = img.cols;
	int chan = img.channels();

	auto Abs = [&](float f) {
		return f > 0 ? f : -f;
	};

	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			// # ����õ��Ƿ����α�Բ�ķ�Χ֮��
			//# �Ż�����һ����ֱ���ж��ǻ��ڣ�startX, startY)�ľ������
			if ((Abs(i - warpX) > radius) && (Abs(j - warpY) > radius))
				continue;

			float distance = (i - warpX)*(i - warpX) + (j - warpY)*(j - warpY);
			if (distance < ddradius)
			{
				//# �������i, j�������ԭ����
				//# ���㹫ʽ���ұ�ƽ������Ĳ���
				float ratio = (ddradius - distance) / (ddradius - distance + mc);
				ratio *= ratio;

				//ӳ��ԭλ��
				float UX = i - ratio * (endX - warpX);
				float UY = j - ratio * (endY - warpY);

				//����˫���Բ�ֵ�õ�UX UY��ֵ
				BilinearInsert(img, dst, UX, UY, i, j);
				//�ı䵱ǰ��ֵ
			}
		}
	}

	return dst;

}
void BeautyCam::on_beautyFace(int b, void*userdata)
{
	Mat src = *((Mat *)userdata);
	Mat img = src.clone();
	double scale = 1.3;
	
	CascadeClassifier cascade = m_pIntance->loadCascadeClassifier("./haarcascade_frontalface_alt.xml");//������ѵ������
	CascadeClassifier netcascade = m_pIntance->loadCascadeClassifier("./haarcascade_eye_tree_eyeglasses.xml");//���۵�ѵ������
	if (cascade.empty() || netcascade.empty())
		return;
	m_pIntance->detectAndDraw(img, cascade, scale,b);
	if (m_pIntance->isDetected == false)
	{
		cout << "enter" << endl;
		Mat dst;

		int value1 = 3, value2 = 1;

		int dx = value1 * 5;    //˫���˲�����֮һ  
		//double fc = value1 * 12.5; //˫���˲�����֮һ  
		double fc = b;
		int p = 50;//͸����  
		Mat temp1, temp2, temp3, temp4;

		//��ԭͼ��image����˫���˲����������temp1ͼ����
		bilateralFilter(img, temp1, dx, fc, fc);

		//��temp1ͼ���ȥԭͼ��image�����������temp2ͼ����
		temp2 = (temp1 - img + 128);

		//��˹ģ��  
		GaussianBlur(temp2, temp3, Size(2 * value2 - 1, 2 * value2 - 1), 0, 0);

		//��ԭͼ��imageΪ��ɫ����temp3ͼ��Ϊ���ɫ��������ͼ��������Թ��ϵõ�ͼ��temp4
		temp4 = img + 2 * temp3 - 255;

		//���ǲ�͸���ȣ�������һ���Ľ�����õ�����ͼ��dst
		dst = (img*(100 - p) + temp4 * p) / 100;
		dst.copyTo(img);
	}
	imshow("BeautyCam", img);
}
CascadeClassifier BeautyCam::loadCascadeClassifier(const string cascadePath)
{
	CascadeClassifier cascade;
	if (!cascadePath.empty())
	{
		if(!cascade.load(cascadePath))//��ָ�����ļ�Ŀ¼�м��ؼ���������
		{
			cerr << "ERROR: Could not load classifier cascade" << endl;
		}
	}
	return cascade;
}
void BeautyCam::detectAndDraw(Mat& img, CascadeClassifier& cascade,  double scale, int val)
{
	std::vector<Rect> faces;
	const static Scalar colors[] = { CV_RGB(0,0,255),
		CV_RGB(0,128,255),
		CV_RGB(0,255,255),
		CV_RGB(0,255,0),
		CV_RGB(255,128,0),
		CV_RGB(255,255,0),
		CV_RGB(255,0,0),
		CV_RGB(255,0,255) };//�ò�ͬ����ɫ��ʾ��ͬ������
	//��ͼƬ��С���ӿ����ٶ�
	Mat gray, smallImg(cvRound(img.rows / scale), cvRound(img.cols / scale), CV_8UC1);
	//��Ϊ�õ�����haar���������Զ��ǻ��ڻҶ�ͼ��ģ�����Ҫת���ɻҶ�ͼ��
	cvtColor(img, gray, CV_BGR2GRAY);
	resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);//���ߴ���С��1/scale,�����Բ�ֵ
	equalizeHist(smallImg, smallImg);//ֱ��ͼ����
	cascade.detectMultiScale(smallImg, //image��ʾ����Ҫ��������ͼ��
		faces,//objects��ʾ��⵽������Ŀ������
		1.1, //caleFactor��ʾÿ��ͼ��ߴ��С�ı���
		2, //minNeighbors��ʾÿһ��Ŀ������Ҫ����⵽3�β��������Ŀ��(��Ϊ��Χ�����غͲ�ͬ�Ĵ��ڴ�С�����Լ�⵽����),
		0 | CASCADE_SCALE_IMAGE ,//minSizeΪĿ�����С�ߴ�
		Size(30, 30)); //minSizeΪĿ������ߴ�
	int i = 0;
	//�������ľ��ο�
	for (std::vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++)
	{
		isDetected = true;
		Mat smallImgROI;
		std::vector<Rect> nestedObjects;
		Point center, left, right;
		Scalar color = colors[i % 8];
		int radius;
		center.x = cvRound((r->x + r->width*0.5)*scale);//��ԭ��ԭ���Ĵ�С
		center.y = cvRound((r->y + r->height*0.5)*scale);
		radius = cvRound((r->width + r->height)*0.25*scale);

		left.x = center.x - radius;
		left.y = cvRound(center.y - radius * 1.3);

		if (left.y < 0)
		{
			left.y = 0;
		}
		right.x = center.x + radius;
		right.y = cvRound(center.y + radius * 1.3);

		if (right.y > img.rows)
		{
			right.y = img.rows;
		}
		/*ԭ���㷨
		����-ĥƤ�㷨
		Dest =(Src * (100 - Opacity) + (Src + 2 * GuassBlur(EPFFilter(Src) - Src + 128) - 256) * Opacity) /100 ;
		*/
		//�滭ʶ���������
		//rectangle(img, left, right, Scalar(255, 0, 0));
		Mat roi = img(Range(left.y, right.y), Range(left.x, right.x));
		
		Mat dst;
		int value1 = 3, value2 = 1;

		int dx = value1 * 5;    //˫���˲�����֮һ  
		//double fc = value1 * 12.5; //˫���˲�����֮һ 
		double fc = val;//�仯ֵ
		int p = 50;//͸����  
		Mat temp1, temp2, temp3, temp4;

		//˫���˲�    ����ͼ�� ���ͼ�� ÿ���������ֱ����Χ��ɫ�ռ��������sigma  ����ռ��˲�����sigma 
		bilateralFilter(roi, temp1, dx, fc, fc);
		temp2 = (temp1 - roi + 128);
		//��˹ģ��  
		GaussianBlur(temp2, temp3, Size(2 * value2 - 1, 2 * value2 - 1), 0, 0);
		temp4 = roi + 2 * temp3 - 255;
		dst = (roi*(100 - p) + temp4 * p) / 100;
		dst.copyTo(roi);
	}
}