# Distributed-Image-Processing-Pipeline

This project implements a distributed image processing pipeline using the Hadoop ecosystem and TensorFlow. The pipeline performs image classification on the CIFAR-10 dataset using distributed computing frameworks like HDFS, Spark, and TensorFlow, enabling scalable and efficient processing of large-scale image datasets.

---

## Features
- Distributed data storage using HDFS.
- Parallel image preprocessing and feature extraction with Apache Spark.
- TensorFlow-based image classification into categories like "dog," "car," and "airplane."
- Scalable architecture that integrates big data tools with machine learning.

---

## Tech Stack
- **Hadoop (HDFS, YARN):** Distributed file storage and resource management.
- **Apache Spark (PySpark):** Parallelized image preprocessing and data processing.
- **TensorFlow:** Machine learning for image classification.
- **Python Libraries:** NumPy, Pandas, Matplotlib, PIL, Pickle, OpenCV.
- **Jupyter Notebook:** Interactive environment for development and visualization.

---

## Dataset
- **CIFAR-10 Dataset**
  - Source: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
  - Format: Binary files with 60,000 images across 10 categories.
  - Images: 32x32 pixels, RGB.

---

## Prerequisites
- **Hadoop Ecosystem**: HDFS and YARN services running.
- **Python 3.6+**: Virtual environment recommended.
- **Libraries**: TensorFlow, PySpark, NumPy, Pandas, Matplotlib, PIL, OpenCV.
- **Jupyter Notebook**: For interactive execution.

---

## Installation and Setup

### 1. Install Python Libraries
Install the required Python libraries in a virtual environment:

pip install tensorflow pyspark numpy pandas matplotlib pillow opencv-python


### 2. Set Up Hadoop and HDFS
-Start Hadoop services:
start-dfs.sh
start-yarn.sh

-Upload CIFAR-10 dataset to HDFS:
hdfs dfs -mkdir -p /images/cifar10/

### 3. Run Jupyter Notebook
Launch Jupyter Notebook and open `Image_Classification_Final.ipynb`:

---

## Execution Flow
1. **Data Ingestion**:
   - CIFAR-10 dataset uploaded to HDFS for distributed access.
2. **Preprocessing**:
   - Images are resized and normalized using Spark and OpenCV.
3. **Image Classification**:
   - TensorFlow model classifies images into categories.
4. **Storage and Visualization**:
   - Classification results stored in HDFS.
   - Results visualized using Matplotlib in Jupyter Notebook.

---

## Results
- Successfully classified images into categories from the CIFAR-10 dataset.
- Demonstrated the integration of TensorFlow with Spark for distributed image processing.
- Highlighted the scalability and performance of the pipeline using big data tools.

---

## Authors
- **Indrayani Vijaysinh Bhosale**

---

## References
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Apache Spark Documentation](https://spark.apache.org/docs/latest/)
- [Hadoop Ecosystem Documentation](https://hadoop.apache.org/)

