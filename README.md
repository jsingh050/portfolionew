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

## 07: Neural Data Analysis in Matlab ##

I have completed several projects in Matlab to perform data analysis. Some of these projects are represented here in my portfolio: 

# 1. Brain-Computer Interface (BCI) Movement Data Analysis (MATLAB)
Project Focus:

Neural data analysis using a linear regression model to decode hand movements in monkeys based on brain signals.

Technical Elements:

Comet Plot Visualization: Movement trajectories of the monkey’s hand are plotted dynamically to study how it reaches toward targets during a movement task​(

Decoder Analysis: Linear regression models were used to predict hand movement based on neural signals. The accuracy of the decoder was around 94%, but there is scope for improvement using cross-validation or more advanced machine learning methods such as neural networks​Time-Lagged Analysis: The impact of various time lags on decoding accuracy was explored by applying delays between brain signal input and hand movement output​(BCI systems v.2).

Spike Count Tuning Curves: Neural spike counts were analyzed across different neurons to study target-specific firing activity​

Engineering Relevance:

This project involves signal processing, linear modeling, and decoder optimization for real-time applications in brain-computer interfaces. Methods for improving accuracy include adjusting hyperparameters and incorporating neural network-based decoders.

# 2. Neural Visual Perception Data Analysis (MATLAB)
Project Focus:

Simulating visual processing in the retina and V1 cortex using neural data and computational models in MATLAB.

Technical Elements:

Mach Bands Illusion: The phenomenon of edge detection is modeled computationally to mimic how retinal ganglion cells detect brightness and contrast changes​(Singh, J Computer Visio…)​(Singh, J Computer Visio…).

Convolution for Visual Processing: Receptive fields of retinal ganglion cells were convolved with visual stimuli to simulate neural response to light intensity and edge detection​(Singh, J Computer Visio…)​(Singh, J Computer Visio…).

Gabor Function for V1 Simulation: Gabor filters were used to model neurons in the visual cortex (V1), which are sensitive to specific orientations and spatial frequencies. Convolution with images helped detect edges and textures, simulating V1 neuron activity​(Singh, J Computer Visio…).

Engineering Relevance:

This analysis integrates image processing, convolution operations, and Gabor filters to simulate biological vision systems. It emphasizes how artificial systems can replicate biological processes for applications like computer vision and neural prosthetics.

# 3. Reaching Task Data Analysis (MATLAB)
Project Focus:

Behavioral and neural analysis of monkeys performing a reaching task with data collected from the premotor cortex.

Technical Elements:

Movement and Eye Position Tracking: Analysis of hand and eye movements during a delayed center-out task, with visualizations comparing actual and predicted movement paths​(reaching hw disclaim).

Reaction Time and Velocity Analysis: Velocity and reaction time of hand movements were studied to calculate how quickly the monkey responded to stimuli. An ANOVA test was used to assess statistical significance between reaction times across different trials​(reaching hw disclaim).

Peristimulus Time Histogram (PSTH): Neural spike data were plotted to show the firing rates of neurons before and after stimulus presentation. Directional selectivity was detected in certain neurons​(reaching hw disclaim).

Engineering Relevance:

This project combines biomechanics and neuroscience data analysis to understand motor responses. The statistical methods used, such as PSTH and ANOVA, are critical in quantifying neuron response times and performance in motor tasks.

# 4. Spike Sorting for Neural Data (MATLAB)
Project Focus:

The uploaded .m file likely contains code for spike sorting, a method used to identify and classify action potentials from multi-neuron recordings in neural data.

Technical Elements:

Spike sorting involves filtering raw data, detecting spikes, and clustering them based on waveform characteristics. This process is essential for interpreting neural recordings from brain-computer interfaces or neuroscience experiments.

Engineering Relevance:

Spike sorting is central to signal processing in neural engineering, where separating signals from different neurons is crucial for accurate data interpretation. This method supports advanced applications in neural prosthetics and neurofeedback systems.

Overall Engineering Themes:
Signal Processing: Across all projects, signal processing is a core element, whether decoding movement from neural activity or processing visual stimuli to simulate perception.

Statistical Modeling and Machine Learning: Linear regression, decoder models, ANOVA, and potential neural networks are used to model relationships between neural inputs and behavioral outputs.

Neuroprosthetics Applications: The projects focus on translating neural activity into actionable outputs like hand movements or visual processing, which are key areas in brain-computer interfaces and prosthetics.

Computational Neuroscience Tools: MATLAB is heavily used for modeling neural data and analyzing physiological responses, combining methods from both biomedical engineering and computer vision.


