
# All Projects

## 01: Motion-Edge Detector ##


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

# 07: Neural Data Analysis in Matlab ##

I have completed several projects in Matlab to perform data analysis. Some of these projects are represented here in my portfolio: 

## 1. Brain-Computer Interface (BCI) Movement Data Analysis (MATLAB)
Project Focus:

Neural data analysis using a linear regression model to decode hand movements in monkeys based on brain signals.

Technical Elements:

Comet Plot Visualization: Movement trajectories of the monkey’s hand are plotted dynamically to study how it reaches toward targets during a movement task​(

Decoder Analysis: Linear regression models were used to predict hand movement based on neural signals. The accuracy of the decoder was around 94%, but there is scope for improvement using cross-validation or more advanced machine learning methods such as neural networks​Time-Lagged Analysis: The impact of various time lags on decoding accuracy was explored by applying delays between brain signal input and hand movement output​(BCI systems v.2).

Spike Count Tuning Curves: Neural spike counts were analyzed across different neurons to study target-specific firing activity​

Engineering Relevance:

This project involves signal processing, linear modeling, and decoder optimization for real-time applications in brain-computer interfaces. Methods for improving accuracy include adjusting hyperparameters and incorporating neural network-based decoders.

## 2. Neural Visual Perception Data Analysis (MATLAB)
Project Focus:

Simulating visual processing in the retina and V1 cortex using neural data and computational models in MATLAB.

Technical Elements:

Mach Bands Illusion: The phenomenon of edge detection is modeled computationally to mimic how retinal ganglion cells detect brightness and contrast changes​(Singh, J Computer Visio…)​(Singh, J Computer Visio…).

Convolution for Visual Processing: Receptive fields of retinal ganglion cells were convolved with visual stimuli to simulate neural response to light intensity and edge detection​(Singh, J Computer Visio…)​(Singh, J Computer Visio…).

Gabor Function for V1 Simulation: Gabor filters were used to model neurons in the visual cortex (V1), which are sensitive to specific orientations and spatial frequencies. Convolution with images helped detect edges and textures, simulating V1 neuron activity​(Singh, J Computer Visio…).

Engineering Relevance:

This analysis integrates image processing, convolution operations, and Gabor filters to simulate biological vision systems. It emphasizes how artificial systems can replicate biological processes for applications like computer vision and neural prosthetics.

## 3. Reaching Task Data Analysis (MATLAB)
Project Focus:

Behavioral and neural analysis of monkeys performing a reaching task with data collected from the premotor cortex.

Technical Elements:

Movement and Eye Position Tracking: Analysis of hand and eye movements during a delayed center-out task, with visualizations comparing actual and predicted movement paths​(reaching hw disclaim).

Reaction Time and Velocity Analysis: Velocity and reaction time of hand movements were studied to calculate how quickly the monkey responded to stimuli. An ANOVA test was used to assess statistical significance between reaction times across different trials​(reaching hw disclaim).

Peristimulus Time Histogram (PSTH): Neural spike data were plotted to show the firing rates of neurons before and after stimulus presentation. Directional selectivity was detected in certain neurons​(reaching hw disclaim).

Engineering Relevance:

This project combines biomechanics and neuroscience data analysis to understand motor responses. The statistical methods used, such as PSTH and ANOVA, are critical in quantifying neuron response times and performance in motor tasks.

## 4. Spike Sorting for Neural Data (MATLAB)
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

# 03: 1-dimensional cochlear model

In this project, I created a neuromorphic event-based audio sensor that models a 1D cochlea using a microphone. When audio is spoken into the microphone, the sensor processes the audio signals (in this case, spoken digits) by applying Fast Fourier Transform (FFT), breaking down the audio into frequency bands, and generating spike-based events based on power thresholds in each band.

The project mimics the human cochlea, which converts sound into electrical signals by detecting frequency components.

Building a neuromorphic 1D cochlea is important because it simulates how the human ear processes sound in real-time, enabling efficient, low-power signal processing similar to biological systems. This approach is especially valuable for developing auditory prosthetics (e.g., cochlear implants) and low-power audio processing systems for speech recognition and other applications, as it captures key auditory features while minimizing computational and power demands.


## 04: Keyword Spotting Using Spiking Neural Network ##
Keyword Spotting, or using A.I to detect spoken words with increasing accuracy has a multitude of uses in today's day and age - beginning with Alexa, to translation, to accessibility. Spilking Neural Networks may offer an alternative approach with lower latency and higher computational power. 

In this project, using Loihi's neuromorphic chip platform Lava, I created a Spiking Neural Network to perform keyword spotting for spoken digits (0-9)  from a validated dataset. Additionally, Pytorch and keras were used to implement a LIF neuron model to perform keyword spotting.

​I have not made my code or paper publicly available for privacy reasons. Please reach out to view this code, paper, or presentation and I would be happy to share more information. 
