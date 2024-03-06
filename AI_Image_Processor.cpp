#include "AI_Image_Processor.h"
#include "itkGDCMImageIO.h"
#include <itkRescaleIntensityImageFilter.h>
#include <itkImageSeriesReader.h>
#include <itkResampleImageFilter.h>
#include <itkLinearInterpolateImageFunction.h>
#include "itkImageSeriesWriter.h"
#include <filesystem>
#include <experimental/filesystem>


// 构造函数初始化
ImageProcessor::ImageProcessor(int mode) : mode(mode), engine(nullptr), logger() {
	// Load the model
	string engine_model_path;
	string onnx_model_path;
	switch (mode) {
	case 0:
		engine_model_path = "./engines/enhance.engine";
		onnx_model_path = "./engines/enhance.onnx";
		engine = buildEngine(engine_model_path, onnx_model_path, logger);
		break;
	case 1:
		engine_model_path = "./engines/denoise_para2.engine";
		onnx_model_path = "./engines/denoise_para2.onnx";
		engine = buildEngine(engine_model_path, onnx_model_path, logger);
		break;
	case 2:
		engine_model_path = "./engines/denoise_para4.engine";
		onnx_model_path = "./engines/denoise_para4.onnx";
		engine = buildEngine(engine_model_path, onnx_model_path, logger);
		break;
	case 3:
		engine_model_path = "./engines/denoise_para10.engine";
		onnx_model_path = "./engines/denoise_para10.onnx";
		engine = buildEngine(engine_model_path, onnx_model_path, logger);
		break;
	default:
		cout << "ImageProcessor mode is illegal!" << endl;
		break;
	}
}

ImageProcessor::~ImageProcessor() {
	if (engine) {
		engine->destroy();
	}
}

ItkData ImageProcessor::readImage(string dcmPath) {
	typedef itk::GDCMImageIO ImageIOType; 
	typedef itk::ImageFileReader<ImageType2F> ReaderType; 

	ImageIOType::Pointer dicomIO = ImageIOType::New(); // 创建 DICOM IO
	ReaderType::Pointer reader = ReaderType::New(); // 创建图像文件读取器

	// 设置 DICOM 文件路径
	reader->SetFileName(dcmPath);
	reader->SetImageIO(dicomIO);

	try {
		reader->Update();
	}
	catch (itk::ExceptionObject& ex) {
		cerr << "Exception caught: " << ex << endl;
	}

	// 获取图像数据
	ImageType2F::Pointer img = reader->GetOutput();
	itk::MetaDataDictionary meta_dict = dicomIO->GetMetaDataDictionary();
	ImageType2F::RegionType region = img->GetLargestPossibleRegion();
	ImageType2F::SizeType size = region.GetSize();

	ItkData itkData;
	itkData.dcmPath = dcmPath;
	itkData.img = img;
	itkData.size = size;
	itkData.meta_dict = meta_dict;

	return itkData;
}

void ImageProcessor::printMetaData(const itk::MetaDataDictionary& meta_dict)
{
	typedef itk::MetaDataDictionary::ConstIterator MetaDataIterator;
	MetaDataIterator metaIt = meta_dict.Begin();
	MetaDataIterator metaEnd = meta_dict.End();

	while (metaIt != metaEnd)
	{
		string key = metaIt->first;
		string value;
		itk::ExposeMetaData<string>(meta_dict, key, value);
		cout << "Key: " << key << ", Value: " << value << endl;
		++metaIt;
	}
}

// 图像预处理函数实现
ItkData ImageProcessor::preprocessImage(ItkData inputData) {
	cout << "Preprocessing the image..." << endl;

	// 图像归一化
	typedef itk::RescaleIntensityImageFilter<ImageType2F, ImageType2F> RescalerType;
	RescalerType::Pointer rescaler = RescalerType::New();
	ImageType2F::Pointer inputImg = inputData.img;
	ImageType2F::SizeType inputSize = inputData.size;

	rescaler->SetInput(inputImg);
	rescaler->SetOutputMinimum(0.0);
	rescaler->SetOutputMaximum(1.0);
	rescaler->Update();

	// 图像resize，因为神经网络包含5次下采样，图像大小必须是32的整数倍
	ImageType2F::SizeType outSize;
	float upsample_scale = (mode == 0) ? 2.0 : 1.0;

	outSize[0] = static_cast<unsigned int>(ceil(inputSize[0] / 32.0)) * 32 * upsample_scale;
	outSize[1] = static_cast<unsigned int>(ceil(inputSize[1] / 32.0)) * 32 * upsample_scale;

	ImageType2F::Pointer outputImg;
	resizeImg(rescaler->GetOutput(), outputImg, outSize);
	ItkData outputData(inputData);
	outputData.img = outputImg;
	outputData.size = outSize;

	return outputData;
}


void ImageProcessor::resizeImg(ImageType2F::Pointer oriImg, ImageType2F::Pointer& outImg, ImageType2F::SizeType targetSize) {
	typedef itk::ResampleImageFilter<ImageType2F, ImageType2F> ResampleFilterType;
	ResampleFilterType::Pointer resampleFilter = ResampleFilterType::New();

	resampleFilter->SetInput(oriImg);
	resampleFilter->SetSize(targetSize);

	// 计算 resize 后的 spacing
	ImageType2F::SizeType origintSize = oriImg->GetLargestPossibleRegion().GetSize();
	ImageType2F::SpacingType targetSpacing;
	targetSpacing[0] = oriImg->GetSpacing()[0] * origintSize[0] / targetSize[0];
	targetSpacing[1] = oriImg->GetSpacing()[1] * origintSize[1] / targetSize[1];
	resampleFilter->SetOutputSpacing(targetSpacing);

	// 设置输出原点与输入一致
	resampleFilter->SetOutputOrigin(oriImg->GetOrigin());
	resampleFilter->SetOutputDirection(oriImg->GetDirection());

	// 使用线性插值
	typedef itk::LinearInterpolateImageFunction<ImageType2F, double> InterpolatorType;
	InterpolatorType::Pointer interpolator = InterpolatorType::New();
	resampleFilter->SetInterpolator(interpolator);

	// 执行重采样
	resampleFilter->Update();

	// 返回重采样后的图像
	outImg = resampleFilter->GetOutput();
}


// engine加载
ICudaEngine* ImageProcessor::buildEngine(
	string engine_model_path,
	string onnx_model_path,
	ILogger& logger) 
{
	ICudaEngine *engine = nullptr;
	// 判断是否存在序列化文件
	ifstream engineFile(engine_model_path, ios_base::in | ios::binary);
	if (!engineFile) {
		engineFile.close();
		// 如果不存在.engine文件则读取.onnx文件并启动序列化过程，生成.engine文件，并反序列化得到engine
		ifstream onnxFile(onnx_model_path);
		if (!onnxFile.good()){
			throw runtime_error("Failed to open engine file and onnx file!");
		}
		IBuilder *builder = createInferBuilder(logger);
		const uint32_t explicit_batch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
		INetworkDefinition *network = builder->createNetworkV2(explicit_batch);

		IParser *parser = createParser(*network, logger);
		parser->parseFromFile(onnx_model_path.c_str(), static_cast<int>(ILogger::Severity::kERROR));
		for (int32_t i = 0; i < parser->getNbErrors(); ++i) {
			cout << parser->getError(i)->desc() << endl;
		}

		IBuilderConfig *config = builder->createBuilderConfig();
		config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1U << 32);

		// 设置FP16推理
		if (builder->platformHasFastFp16()) {
			config->setFlag(BuilderFlag::kFP16);
		}

		IOptimizationProfile* profile = builder->createOptimizationProfile();
		profile->setDimensions("input", OptProfileSelector::kMIN, Dims4(1, 1, 64, 64));	
		profile->setDimensions("input", OptProfileSelector::kOPT, Dims4(1, 1, 512, 512));	
		profile->setDimensions("input", OptProfileSelector::kMAX, Dims4(1, 1, 1024, 1024));	
		config->addOptimizationProfile(profile);


		IHostMemory *serialized_model = builder->buildSerializedNetwork(*network, *config);

		// 将模型序列化到engine文件中
		stringstream engine_file_stream;
		engine_file_stream.seekg(0, engine_file_stream.beg);
		engine_file_stream.write(static_cast<const char *>(serialized_model->data()), serialized_model->size());
		ofstream out_file(engine_model_path, ios_base::out | ios::binary);
		assert(out_file.is_open());
		out_file << engine_file_stream.rdbuf();
		out_file.close();

		// 反序列化
		IRuntime *runtime = createInferRuntime(logger);
		assert(runtime != nullptr);
		engine = runtime->deserializeCudaEngine(serialized_model->data(), serialized_model->size());
		assert(engine != nullptr);

		delete config;
		delete parser;
		delete network;
		delete builder;
		delete serialized_model;
		delete runtime;
	}
	else {
		// 如果有.engine文件，则直接读取文件，反序列化生成engine
		engineFile.seekg(0, ios::end);
		size_t engineSize = engineFile.tellg();
		engineFile.seekg(0, ios::beg);
		vector<char> engineData(engineSize);
		engineFile.read(engineData.data(), engineSize);
		engineFile.close();

		IRuntime *runtime = createInferRuntime(logger);
		assert(runtime != nullptr);
		engine = runtime->deserializeCudaEngine(engineData.data(), engineSize, nullptr);
		assert(engine != nullptr);

		delete runtime;
	}
	return engine;
}

// engine预测
void ImageProcessor::enginePredict(ICudaEngine *engine, ItkData& inputData)
{
	ImageType2F::Pointer imageData = inputData.img;
	float* imageData_buffer = imageData->GetBufferPointer();
	void *buffers[2];
	// 获取模型的输入维度并分配GPU内存
	Dims inputDims = engine->getBindingDimensions(0);
	inputDims.d[2] = inputData.size[0];
	inputDims.d[3] = inputData.size[1];

	int insize = 1;
	for (int j = 0; j < inputDims.nbDims; ++j) {
		insize *= inputDims.d[j];
	}
	cudaMalloc(&buffers[0], insize * sizeof(float));

	// 获取模型输出尺寸并分配GPU内存
	Dims outputDims = engine->getBindingDimensions(1);
	outputDims.d[2] = inputData.size[0];
	outputDims.d[3] = inputData.size[1];
	
	int osize = 1;
	for (int j = 0; j < outputDims.nbDims; ++j) {
		osize *= outputDims.d[j];
	}
	cudaMalloc(&buffers[0], osize * sizeof(float));
	cudaMalloc(&buffers[1], osize * sizeof(float));

	// 给模型输出数据分配相应的CPU内存
	float *output_buffer = new float[osize]();

	// 创建cuda流
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	// 拷贝输入数据
	cudaMemcpyAsync(buffers[0], imageData_buffer, insize * sizeof(float), cudaMemcpyHostToDevice, stream);

	// 执行推理
	IExecutionContext *context = engine->createExecutionContext();

	// 设置输入维度
	context->setBindingDimensions(0, inputDims);
	context->setOptimizationProfileAsync(0, stream);
	context->enqueueV2(buffers, stream, nullptr);
	// 拷贝输出数据
	cudaMemcpyAsync(output_buffer, buffers[1], osize * sizeof(float), cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);

	cudaFree(buffers[0]);
	cudaFree(buffers[1]);
	cudaStreamDestroy(stream);
	context->destroy();

	// 输出数据截断到0-1之间，并传递给ItkImage
	for (int i = 0; i < osize; ++i) {
		if (output_buffer[i] < 0.0f) {
			output_buffer[i] = 0.0f;
		}
		else if (output_buffer[i] > 1.0f) {
			output_buffer[i] = 1.0f;
		}
	}
	memcpy(inputData.img->GetBufferPointer(), output_buffer, inputData.size[0]* inputData.size[1] * sizeof(float));
}

void ImageProcessor::postProcess(ItkData& inputData, ImageType2F::SizeType oShape) {

	// 图像灰阶调整
	typedef itk::RescaleIntensityImageFilter<ImageType2F, ImageType2F> RescalerType;
	RescalerType::Pointer rescaler = RescalerType::New();

	rescaler->SetInput(inputData.img);
	rescaler->SetOutputMinimum(0.0);
	rescaler->SetOutputMaximum(4095.0);
	rescaler->Update();

	// 图像缩放
	resizeImg(rescaler->GetOutput(), inputData.img, oShape);
	inputData.size = oShape;
}

void ImageProcessor::saveAsDICOM(ItkData inputData) {
	string dcmPath = inputData.dcmPath;
	size_t lastSlashPos = dcmPath.find_last_of('/');
	if (lastSlashPos == string::npos) {
		lastSlashPos = 0;
	}
	else {
		lastSlashPos++;
	}

	// 构造新文件名
	size_t lastDotPos = dcmPath.find_last_of('.');
	string fileName = dcmPath.substr(lastSlashPos, lastDotPos - lastSlashPos);
	string dcmPath_save = dcmPath.substr(0, lastSlashPos) + fileName + "_Res" + dcmPath.substr(lastDotPos);

	// 创建Dcm图像写入器
	using WriterType = itk::ImageFileWriter<ImageType2F>;
	WriterType::Pointer writer_dcm = WriterType::New();
	itk::GDCMImageIO::Pointer dcmIO = itk::GDCMImageIO::New();

	// 设置输出路径
	writer_dcm->SetFileName(dcmPath_save);
	dcmIO->SetMetaDataDictionary(inputData.meta_dict);
	writer_dcm->SetImageIO(dcmIO);
	writer_dcm->SetInput(inputData.img);
	writer_dcm->SetUseInputMetaDataDictionary(0);
	writer_dcm->Update();
}

void ImageProcessor::pushToStack(string dataPath) {
	dataStack.push(dataPath);
}

string ImageProcessor::popFromStack() {
	string dataPath = dataStack.top();
	dataStack.pop();
	return dataPath;
}

void ImageProcessor::processStart() {
	cout << "Image processing pipeline initiated!..." << endl;
	while (!dataStack.empty()) {
		// 从栈中取出数据
		string dataPath = popFromStack();
		ItkData inputData = readImage(dataPath);
		// 图像预处理
		ItkData processData = preprocessImage(inputData);
		// engine 预测
		enginePredict(engine, processData);
		float upsample_scale = (mode == 0) ? 2.0 : 1.0;
		ImageType2F::SizeType Osize;
		Osize[0] = inputData.size[0] * upsample_scale;
		Osize[1] = inputData.size[1] * upsample_scale;
		postProcess(processData, Osize);
		saveAsDICOM(processData);
	}
}


int main() {
	auto startTime = chrono::high_resolution_clock::now(); // 开始计时

	// 创建 ImageProcessor 对象
	ImageProcessor processor(2);
	string directory = "./test_data";
	for (const auto& entry : experimental::filesystem::directory_iterator(directory)) {
		if (entry.path().extension() == ".dcm") {
			processor.pushToStack(entry.path().string());
		}
	}
	processor.processStart();

	auto endTime = chrono::high_resolution_clock::now(); // 结束计时
	chrono::duration<double> TotalTime = endTime - startTime; // 计算耗时
	cout << "Total Time: " << TotalTime.count() << " seconds" << endl;

	return 0;

}