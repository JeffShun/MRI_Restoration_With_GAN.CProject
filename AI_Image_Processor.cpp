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
ImageProcessor::ImageProcessor(string engine_model_path, string onnx_model_path) :
	engine_model_path(engine_model_path),
	onnx_model_path(onnx_model_path),
	engine(nullptr),
	logger() {
	// Load the model
	engine = buildEngine(engine_model_path, onnx_model_path, logger);
}

ImageProcessor::~ImageProcessor() {
	if (engine) {
		engine->destroy();
	}
}

void ImageProcessor::readImage(mData& dataStream) {
	typedef itk::GDCMImageIO ImageIOType; 
	typedef itk::ImageFileReader<ImageType2F> ReaderType; 

	ImageIOType::Pointer dicomIO = ImageIOType::New(); 
	ReaderType::Pointer reader = ReaderType::New();

	reader->SetFileName(dataStream.inputPath);
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

	dataStream.inImg = img;
	dataStream.inSize = size;
	dataStream.inMetaDict = meta_dict;
}


// 图像预处理函数实现
void ImageProcessor::preprocessImage(mData& dataStream) {
	cout << "Preprocessing the image..." << endl;

	// 图像归一化
	typedef itk::RescaleIntensityImageFilter<ImageType2F, ImageType2F> RescalerType;
	RescalerType::Pointer rescaler = RescalerType::New();

	rescaler->SetInput(dataStream.inImg);
	rescaler->SetOutputMinimum(0.0);
	rescaler->SetOutputMaximum(1.0);
	rescaler->Update();

	// 图像resize，因为神经网络包含5次下采样，图像大小必须是32的整数倍
	float upsample_scale = dataStream.sr_ratio * 2 + 1;
	dataStream.outSize[0] = static_cast<unsigned int>(ceil(dataStream.inSize[0] * upsample_scale / 32.0)) * 32 ;
	dataStream.outSize[1] = static_cast<unsigned int>(ceil(dataStream.inSize[1] * upsample_scale / 32.0)) * 32 ;

	ImageType2F::Pointer Img;
	resizeImg(rescaler->GetOutput(), Img, dataStream.outSize);
	dataStream.outImg = Img;
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
ICudaEngine* ImageProcessor::buildEngine(string engine_model_path, string onnx_model_path, ILogger& logger) 
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
		profile->setDimensions("input", OptProfileSelector::kMIN, Dims4(1, 4, 64, 64));	
		profile->setDimensions("input", OptProfileSelector::kOPT, Dims4(1, 4, 512, 512));	
		profile->setDimensions("input", OptProfileSelector::kMAX, Dims4(1, 4, 1024, 1024));	
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
void ImageProcessor::enginePredict(mData& dataStream)
{
	ImageType2F::Pointer inputImage = dataStream.outImg;
	float* imageBuffer = inputImage->GetBufferPointer();

	// 定义输入、输出尺寸
	size_t insize = 4 * dataStream.outSize[0] * dataStream.outSize[1];
	size_t osize = dataStream.outSize[0] * dataStream.outSize[1];

	// 可变维度赋值
	Dims inputDims = engine->getBindingDimensions(0);
	Dims outputDims = engine->getBindingDimensions(1);
	inputDims.d[2] = dataStream.outSize[0];
	inputDims.d[3] = dataStream.outSize[1];
	outputDims.d[2] = dataStream.outSize[0];
	outputDims.d[3] = dataStream.outSize[1];

	// 将Image和SR Ratio、Deblur Ratio、Denoise Ratio拼接为engine输入
	vector<float> engineBuffer(insize);
	copy(imageBuffer, imageBuffer + osize, engineBuffer.begin());
	fill_n(engineBuffer.begin() + osize * 1, osize, dataStream.sr_ratio);
	fill_n(engineBuffer.begin() + osize * 2, osize, dataStream.deblur_ratio);
	fill_n(engineBuffer.begin() + osize * 3, osize, dataStream.denoise_ratio);
	float* engineBufferPtr = engineBuffer.data();
	
	// 分配输入、输出显存
	void *buffers[2];
	cudaMalloc(&buffers[0], insize * sizeof(float));
	cudaMalloc(&buffers[1], osize * sizeof(float));

	// 给模型输出数据分配相应的CPU内存
	float *output_buffer = new float[osize]();

	// 创建cuda流
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	// 拷贝输入数据
	cudaMemcpyAsync(buffers[0], engineBufferPtr, insize * sizeof(float), cudaMemcpyHostToDevice, stream);

	// 创建Context
	IExecutionContext *context = engine->createExecutionContext();

	// 设置Context参数，执行推理
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

	// 输出数据截断到0-1之间
	for (int i = 0; i < osize; ++i) {
		if (output_buffer[i] < 0.0f) {
			output_buffer[i] = 0.0f;
		}
		else if (output_buffer[i] > 1.0f) {
			output_buffer[i] = 1.0f;
		}
	}
	memcpy(dataStream.outImg->GetBufferPointer(), output_buffer, osize * sizeof(float));
}

void ImageProcessor::postProcess(mData& dataStream) {

	// 图像灰阶调整
	typedef itk::RescaleIntensityImageFilter<ImageType2F, ImageType2F> RescalerType;
	RescalerType::Pointer rescaler = RescalerType::New();

	rescaler->SetInput(dataStream.outImg);
	rescaler->SetOutputMinimum(0.0);
	rescaler->SetOutputMaximum(4095.0);
	rescaler->Update();
	dataStream.outImg = rescaler->GetOutput();
}

void ImageProcessor::saveDICOM(mData& dataStream) {
	string dcmPath = dataStream.inputPath;
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

	// 设置metaDict
	dataStream.outMetaDict = dataStream.inMetaDict;
	dcmIO->SetMetaDataDictionary(dataStream.outMetaDict);

	writer_dcm->SetFileName(dcmPath_save);
	writer_dcm->SetImageIO(dcmIO);
	writer_dcm->SetInput(dataStream.outImg);
	writer_dcm->SetUseInputMetaDataDictionary(0);
	writer_dcm->Update();
}

void ImageProcessor::pushToQueue(mData& dataStream) {
	dataQueue.push(dataStream);
}


size_t ImageProcessor::queueLen() {
	size_t length = dataQueue.size();
	return length;
}


void ImageProcessor::processStart() {
	cout << "Image processing pipeline initiated!..." << endl;
	while (!dataQueue.empty()) {
		// 从队列中取出数据
		mData dataStream = dataQueue.front();
		dataQueue.pop();
		// 读取到数据流
		readImage(dataStream);
		// 图像预处理
		preprocessImage(dataStream);
		// engine 预测
		enginePredict(dataStream);
		// 后处理
		postProcess(dataStream);
		// 保存输出为dcm
		saveDICOM(dataStream);
	}
}


int main() {
	
	// 创建 ImageProcessor 对象
	string engine_model_path = "./model/model.engine";
	string onnx_model_path = "./model/model.onnx";
	ImageProcessor processor(engine_model_path, onnx_model_path);

	auto startTime = chrono::high_resolution_clock::now(); // 开始计时
	// 初始化传入参数
	string directory = "./test_mini";
	float sr_ratio = 1.0;
	float deblur_ratio = 0.7;
	float denoise_ratio = 0.3;
	// Push 到处理管道
	for (const auto& entry : experimental::filesystem::directory_iterator(directory)) {
		if (entry.path().extension() == ".dcm") {
			mData dataStream;
			dataStream.inputPath = entry.path().string();
			dataStream.sr_ratio = sr_ratio;
			dataStream.deblur_ratio = deblur_ratio;
			dataStream.denoise_ratio = denoise_ratio;
			processor.pushToQueue(dataStream);
		};
	}
	processor.processStart();

	auto endTime = chrono::high_resolution_clock::now(); // 结束计时
	chrono::duration<double> TotalTime = endTime - startTime; // 计算耗时
	cout << "Total Time: " << TotalTime.count() << " seconds" << endl;

	return 0;

}