# QM Active Learning Diffusion for Molecular Design

This model uses the Diffusion model and DFT through active learning to navigate chemical space and improve our generative model.

The software system is organized into a modular architecture that follows a three-phase workflow, guiding the process from molecular trajectory design to data analysis. Each phase encapsulates a well-defined set of responsibilities and interacts through standardized data interfaces, ensuring a clear separation of concerns and reproducible experimentation.

• Phase I – Trajectory Planning
This phase provides an interactive environment for defining molecular trajectories within a property space (e.g., HLgap, eMBD, atNUM). The resulting trajectory files trajectory_ ∗ .csv serve as the starting point for all subsequent diffusion experiments.

<img width="200" height="200" alt="trajectory_planer" src="https://github.com/user-attachments/assets/a827726d-cfd1-411e-80ce-79d87f5f7dbe" />

• Phase II – Active Diffusion Execution
It represents the core operational layer of the system, responsible for orchestrating the iterative molecular generation, evaluation, and feedback process.
The architectural elements are organized as follows:
  • Top Layer – Experiment Orchestration Layer (blue box): Responsible for workflow coordination, configuration management, and resource allocation across HPC environments. It defines the experiment logic and interacts with lower components throughstandardized interfaces.
  • Bottom Layer – Functional Core Layer (red box): Contains the domain-specific components that implement the scientific and computational functionality of the diffusion process, including molecule relaxation, property calculations, filtering, and data reinjection.
  • Utilities Subsystem (green box): Provides cross-cutting services that support both layers by handling data management, task parallelization, and trajectory synchronization.
  
<img width="280" height="240" alt="active_diffusion_execution" src="https://github.com/user-attachments/assets/bb0c4d90-376a-4cf1-ad2c-f767dece0c78" />

• Phase III – Output Analysis
The final phase focuses on analyzing and visualizing the output. Jupyter notebooks (accuracy_analysis.ipynb, iteration_analysis.ipynb, hdf 5_analysis.ipynb) allow users to inspect the Active diffusion performance and compare results across experiments.

<img width="400" height="400" alt="embd_model_eval" src="https://github.com/user-attachments/assets/c23ebf26-e553-4d12-bcee-e8302bc121f0" />

<img width="400" height="400" alt="eval_embd_exp" src="https://github.com/user-attachments/assets/16936e0f-e4e8-4dcc-82b7-91b445ec5684" />
