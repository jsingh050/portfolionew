## portfolionew ##

# All Projects

## 02: MNIST digit Classification with MLP ##

In this project, I developed a Multi-Layer Perceptron (MLP) to classify images of digits (1-4) from the MNIST dataset, achieving an accuracy of around 95%. The objective was to explore the impact of different activation functions (ReLU vs. Sigmoid) and variations in the number of hidden layers and nodes.
          
          Model      Activation Loss	          Accuracy
          32 Nodes	ReLU	0.0299	          93.06%
          32 Nodes	Sigmoid	0.2963	          90.98%
          16 Nodes	Sigmoid	0.4215	          87.43%
          16 Nodes	ReLU	0.6434	          78.47%
          64 Nodes	ReLU	0.2790	          94.88%
          64 Nodes	Sigmoid	0.2790	          91.51%
          64 Nodes * Sigmoid	0.2568	        92.37%
          * 2-hidden-layers
          
Through experimentation, I gained insights into balancing model complexity to avoid underfitting and   overfitting. I also observed the trade-offs between activation functions, understanding that ReLU tends to perform better in deeper networks but can lead to overfitting without regularization techniques.
