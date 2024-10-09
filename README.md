
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

## 09: Neuron Models ##

I explored implementation of a bio-inspired neuron model in Python. This model draws inspiration from biological neurons, which are complex and incredibly energy efficient and have low-latency. First, I implemented the famous Izhikevick neuron model. This model simplifies the the Hodgkin-Huxley model to offer computational efficiency. The Izhikevich model uses two differential equations to simulate membrane potential (voltage) and recovery (adaptation) variables in neurons. By adjusting the parameters, I could simulate different types of neuronal behaviors such as regular spiking (RS), fast spiking (FS), intrinsically bursting (IB), and chattering (CH). I used python to perform differential equation solving, tune the parameters, and finally to plot the data (voltage-time and adaptation-time graphs) for each neuron type. Spike frequency was analyzed as a function of input current for the regular spiking neuron. By plotting spike frequency versus the input current, the minimum spike frequency was determined, providing insight into the responsiveness of neurons to external stimuli. This exercise is critical for designing neuromorphic circuits that respond accurately to varying inputs, a key feature in sensory systems like vision or audition.

Then, I went on to model associative learning in neural networks. In neuroscience, it is known that neurons that "fire together, wire together". Since neurons can form strong connections to one another, it allows learning to take place. Through an LIF neuron model with Spike-Timing Dependent Plasticity (STDP), it can be seen how synaptic weights can evolve in response to repeated neural firing, leading to the learning of associations between stimuli. The neural network consists of two input neurons (N0, N1) and one output neuron (N2). Initially, N2 fires only in response to N0 due to a strong synaptic weight between N0 and N2. By applying simultaneous stimulation to both N0 and N1, the synaptic weight between N1 and N2 is strengthened through the STDP rule, demonstrating associative learning when, after training, N2 begins to fire in response to N1 alone. Implementing the STDP rule to simulate how synaptic strengths evolve based on the timing of pre- and post-synaptic spikes.
This project showcases a key principle in neural plasticity: the ability of synaptic connections to change based on the activity patterns of neurons. In real biological systems, this allows for learning and memory formation. Here, the successful training of the neural network demonstrates how synaptic plasticity can be computationally modeled, providing a framework for further exploration in machine learning and neuromorphic computing.

Both the Izhikevich neuron model and the LIF neural network with STDP contribute to the broader field of neuromorphic engineering, where hardware and software systems are designed to emulate the functionality of the brain. By simulating neuron firing patterns and synaptic plasticity, these projects lay the foundation for building adaptive neural networks that can learn from stimuli in real-time.



## 09b: Neuron Model ##
Neuromorphic Learning: Izhikevich Models, SNNs, and Bootstrap Training
The recent assignments provide a comprehensive exploration of neuromorphic computing systems, with a focus on Spike-Timing-Dependent Plasticity (STDP), Leaky-Integrate-and-Fire (LIF) neuron models, and Spiking Neural Networks (SNNs) trained using both conventional methods and advanced bootstrap training techniques. These tasks demonstrate key technical skills such as differential equation solving, neural simulation, and training SNNs with biologically inspired learning rules.

1. Spike-Timing-Dependent Plasticity (STDP) in LIF Neurons
The first assignment centered around the implementation of STDP in a neural network of Leaky-Integrate-and-Fire (LIF) neurons. STDP is a form of Hebbian learning where the synaptic strength is adjusted based on the timing of pre- and post-synaptic spikes. The technical work involved the following steps:

Neuron Dynamics Simulation: Simulating the behavior of pre- and post-synaptic neurons using LIF models.
STDP Rule Implementation: Applying the STDP rule to update synaptic weights based on the timing difference between spikes. This biologically inspired rule is crucial for tasks involving associative learning and memory formation in artificial neural networks.
Visualization of Spike Trains and Weight Changes: Monitoring the evolution of spikes and synaptic weights in response to STDP, which allowed for the observation of learning behavior.
The Lava framework was employed for creating LIF neurons and integrating STDP, showcasing the technical expertise in working with neuromorphic platforms.

2. MNIST Digit Classification Using LIF Networks
The second part of the assignment extended the use of LIF neurons to a classic problem in neural networks: MNIST digit classification. This task demonstrated how to build a feed-forward neural network using LIF neurons, where the neurons' spiking behavior approximates that of traditional artificial neurons. The process involved:

Converting MNIST Images to Spike Trains: Input images from the MNIST dataset were converted into spikes proportional to pixel intensity, a technique commonly used in neuromorphic vision systems.
Feed-Forward Network with Dense Layers: A multi-layer LIF network with fully connected layers was used to perform classification. Pre-trained weights were employed to initialize the network, which was then simulated in the Lava framework.
Accuracy Measurement: The model’s accuracy was evaluated by comparing its predictions to the ground truth, achieving promising results for a spiking network without convolutional layers.
This task highlights the potential of spiking neural networks (SNNs) in standard classification tasks while illustrating how biologically plausible models can be adapted to tasks traditionally solved by artificial neural networks (ANNs).

3. Bootstrap SNN Training for MNIST Classification
The final assignment introduced an advanced technique known as bootstrap training for SNNs. The underlying principle of this method is the approximation of a spiking neuron’s firing rate using the ReLU activation function, enabling the seamless conversion between ANN and SNN modes. The following steps were key to this work:

SNN-ANN Conversion: By collecting data from an SNN operating in sampling mode, the activations were used to approximate the corresponding ANN activations, improving training efficiency.
Hybrid ANN-SNN Training: Alternating between ANN and SNN modes accelerated the training process, resulting in faster convergence and higher accuracy for MNIST classification.
Visualization and Export: The trained SNN was visualized through spike-based animations, and the final model was exported for further use in neuromorphic hardware.
This bootstrap method represents an important step in neuromorphic deep learning, as it bridges the gap between traditional ANNs and energy-efficient SNNs, which are well-suited for deployment on neuromorphic hardware platforms like Intel’s Loihi.

Importance and Broader Applications
These assignments are significant for several reasons:

Bio-Inspired Learning: The use of STDP and LIF neuron models in the first two tasks is a direct application of biologically inspired principles to artificial intelligence. STDP, for example, is critical for temporal learning and spike-based computation, both of which are essential for developing intelligent systems that mimic the brain's efficiency.

Spiking Neural Networks (SNNs): SNNs are gaining traction in the neuromorphic community due to their low power consumption and real-time processing capabilities. By approximating neuronal spiking behavior, SNNs provide a more biologically realistic model of computation compared to ANNs, making them suitable for edge devices and applications requiring energy efficiency.

Advanced Training Techniques: The introduction of bootstrap training for SNNs demonstrates how traditional deep learning methods can be adapted to neuromorphic systems, improving their scalability and performance on tasks such as image classification. This hybrid approach is a promising step towards making neuromorphic computing more practical for mainstream applications.

Cross-Disciplinary Applications: The techniques developed here can be applied in various domains, including robotics, autonomous systems, and prosthetics, where real-time processing and energy efficiency are paramount. Furthermore, neuromorphic systems are essential for next-generation AI, offering solutions that combine the power of deep learning with the brain's natural computational efficiency.

Conclusion
The assignments successfully demonstrate the technical capability to work with cutting-edge neuromorphic platforms, implementing advanced models such as STDP and SNNs while leveraging hybrid training techniques like bootstrap learning. The broader significance of this work lies in its potential to revolutionize fields that require low-latency and low-power computation, paving the way for practical neuromorphic applications in real-world AI systems.


