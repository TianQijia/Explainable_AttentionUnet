# Explainable Machine Learning Algorithm and Hardware for Surgical Robotics

## Overview
This project aims to develop an AI-driven segmentation approach tailored for breast tumor delineation, focusing on efficiency and interpretability. The primary objective is to design an Attention UNet model capable of achieving high segmentation accuracy while addressing computational constraints commonly encountered in medical imaging tasks.

## Features
- **High Segmentation Accuracy**: Utilizes Attention UNet model to achieve precise segmentation of breast tumor images.
- **Interpretability**: Implements various interpretability methods like GradCAM to provide insights into model decisions.
- **Efficient Computation**: Designed to work with limited computational resources, such as those available in environments like Google Colab.
- **Hardware Integration**: Proposes a high-performance, low-power hardware solution to complement the algorithm.

## Requirements
- Python 3.x
- TensorFlow
- NumPy
- OpenCV
- tf_explain (for interpretability methods)
- Jupyter Notebook

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/TianQijia/Explainable_AttentionUnet.git
    cd Explainable_AttentionUnet
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
### Running the Jupyter Notebook
1. Open the Jupyter Notebook:
    ```sh
    jupyter notebook
    ```

2. In the Jupyter Notebook interface, navigate to the `finalcode.ipynb` file and open it.

3. Run all the cells in the notebook to execute the code. This will include loading the dataset, training the model, evaluating the model, and visualizing the interpretability results.

### Training the Model
1. Prepare your dataset and place it in the `data` directory.
2. Open and run the `finalcode.ipynb` notebook. Follow the instructions provided in the notebook to train the model.

### Evaluating the Model
1. After training, evaluation results will be generated as part of the `finalcode.ipynb` notebook. Follow the instructions provided in the notebook to evaluate the model.

### Visualizing Interpretability Results
1. Use the interpretability sections in the `finalcode.ipynb` notebook to generate visualizations. This includes visualizing GradCAM and other interpretability methods to understand the model's decision-making process.

## Methodology
### Data Input
The dataset used for training is the Breast Ultrasound Images Dataset from Kaggle, containing 780 images of breast ultrasounds with annotations for tumor regions.

### Model Design
The model architecture is based on the UNet framework, enhanced with attention mechanisms to improve interpretability and segmentation accuracy. The model includes Encoder Blocks, Decoder Blocks, and Attention Gates.

### Evaluation Metrics
The model's performance is evaluated using metrics such as Dice coefficient, Intersection over Union (IoU), accuracy, sensitivity, and specificity.

## Results
The model demonstrates high segmentation accuracy, with extensive experimentation revealing stable performance metrics. The use of interpretability methods like GradCAM provides valuable insights into the model's decision-making process.

## Limitations
Despite its promise, the model has limitations such as a small dataset size and instability in training performance. Future work involves expanding the dataset, improving evaluation metrics, and integrating advanced architectures like Transformers.

## Future Work
- Acquiring larger, more diverse datasets.
- Enhancing evaluation metrics for better performance representation.
- Improving hardware efficiency to support the computational demands.
- Integrating the model with large language models for better interpretability.

## Contributing
Contributions are welcome! Please fork the repository and submit pull requests.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
Special thanks to the project supervisors Dr.Can Li and contributors for their guidance and support.

---

For more details, please refer to the [technical paper](link_to_technical_paper.pdf).
