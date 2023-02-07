#include <openvino/openvino.hpp>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>


int main() {
	std::string img_path = "E:/project/VS2017/dog_cat/img/cat.jpg";	// Ԥ��ͼƬ
	std::string onnx_path = "E:/project/VS2017/dog_cat/model/Cat_dog.onnx"; // Ԥ��ģ��
	size_t input_batch_size = 1;	// ����ͼƬ��batch_size
	size_t num_channels = 3;		// ����ͨ��
	size_t h = 224;					// ����ͼƬ�ĸ�
	size_t w = 224;					// ����ͼƬ�Ŀ�
	clock_t startTime, endTime;		// ����ʱ���¼����

	// 0������IE�������ѯ֧��Ӳ���豸
	ov::Core core;
	//��ȡ��ǰ֧�ֵ����е�AIӲ�������豸
	std::vector<std::string> devices = core.get_available_devices();
	for (int i = 0; i < devices.size(); i++) {
		std::cout << devices[i] << std::endl;
	}
	// 1�����ؼ��ģ��
	// ģ�ͼ��ز�����
	ov::CompiledModel compiled_model = core.compile_model(onnx_path, "AUTO");
	// ���������ƶ��ѱ���ģ�͵������������  ���������������������������
	ov::InferRequest infer_request = compiled_model.create_infer_request();	
	
	// 2��������������
	auto input_tensor = infer_request.get_input_tensor(0);

	// 3��ָ��shape�Ĵ�С
	input_tensor.set_shape({ input_batch_size, num_channels, w, h }); 
	// 4����ȡ����ĵ�ַ�������ݸ�ָ��input_data_host
	float* input_data_host = input_tensor.data<float>();  

	// ��Ӧ��pytorch�Ĵ��벿��
	// ����ʼʱ��
	startTime = clock();
	// opencv��ȡͼƬ
	cv::Mat src = cv::imread(img_path);
	int image_height = src.rows;
	int image_width = src.cols;
	// �޸�ͼƬ��С
	cv::Mat image;
	cv::resize(src, image, cv::Size(w, h));
	int image_area = image.cols * image.rows;
	unsigned char* pimage = image.data;
	float* phost_b = input_data_host + image_area * 0;   // input_data_host��phost_*���е�ַ����
	float* phost_g = input_data_host + image_area * 1;
	float* phost_r = input_data_host + image_area * 2;
	// BGR->RGB
	float mean[] = { 0.406, 0.456, 0.485 };
	float std[] = { 0.225, 0.224, 0.229 };
	for (int i = 0; i < image_area; ++i, pimage += 3) {
		// ע�������˳��rgb������
		*phost_r++ = pimage[0] / 255.;  // ��ͼƬ�е����ص���м�ȥ��ֵ���������ֵ��input
		*phost_g++ = pimage[1] / 255.;
		*phost_b++ = pimage[2] / 255.;
	}
	
	
	// 5��ִ��Ԥ��
	infer_request.infer();
	// 6��������
	auto output = infer_request.get_output_tensor(0);
	// ������������
	float* prob = output.data<float>();
	const int num_classes = 2; // ����
	int predict_label = std::max_element(prob, prob + num_classes) - prob;  // ȷ��Ԥ�������±�
	std::string label;
	if (predict_label == 0)
		label = "cat";
	else
		label = "dog";
	float confidence = prob[predict_label];    // ���Ԥ��ֵ�����Ŷ�
	printf("confidence = %f, label = %s\n", confidence, label);
	endTime = clock();//��ʱ����
	std::cout << "total����ʱ��: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;

	return 0;
}