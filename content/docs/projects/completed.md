# Completed Projects

## In spare time

### LLM Processor for Obsidian Notes

[[GitHub](https://github.com/deepbuzin/note_processor)]

## At TensorSense

### Data Agent

In this unreleased project I built a data generation agent using LangGraph, with a user interface and 4 machine annotation tools. 
It was meant to automate a 5 step workflow from scraping data to generating labels. 

### Training Toolkit and other devtools

- [[GitHub](https://github.com/tensorsense/trash_demo)] [[Substack]()] Assembled a demo app for trash sorting that ran two LoRAs on top of a vanilla PaliGemma to perform segmentation and RAG.
- [[GitHub](https://github.com/tensorsense/training_toolkit)] Made a framework to simplify VLM training, supporting 2 modalities and 4 tasks.
- [[GitHub](https://github.com/tensorsense/training_toolkit)] Also built a system for parallel evaluation of language models. It achieved 10 times faster processing when using 4 GPUs, compared to our previous sequential approach.

### Gemamba

[[GitHub](https://github.com/tensorsense/trash_demo)]

Built a multimodal LLM using the Mamba architecture. 
Achieved a 1% improvement over the equivalent-size PaliGemma model on the MSRVTT-QA.

### ML Platform

[[GitHub](https://github.com/tensorsense/faceflow)] [[Substack]()]

Built a platform for training action unit classifiers with multitask and semi-supervised learning features. 

- Ran 96 experiments across 13 iterations, improving accuracy by 15% compared to off-the-shelf solutions on the in-house benchmark. 
- Integrated PyTorch Lightning, WandB and S3 to track of code, experiments, parameters, dependencies and artifacts.
- Increased experiment throughput from 1 to 3 per day.

Created a data processing pipeline using Airflow, CVAT, and MediaPipe. 

- Automated and parallelized 30 tasks. 
- Brought data processing from 12 to 3 hours per 10000 samples. 
- Integrated cold storage and artifact tracking, ensuring reproducible artifact production and automatic data registry updates. 

### RAG-related stuff

[[GitHub](https://github.com/tensorsense/edgedb_rag)]

Created a RAG system using LangChain for EdgeDB, reducing hallucinations from 75% to 0% on the expert-curated benchmark.

## At Speech Technology Center

### Video conferencing system

Developed cross-camera human tracking system for a 100 square meter conference room using 9 cameras.

- Overcame challenges of obscured floor views and intermittent facial recognition by implementing robust modeling and tracking heuristics.
- Reduced error rate by 20% through system overhaul and process improvements. 
- Managed collection and annotation of 100 hours of footage, collaborating with actors and annotation team to iterate on data quality.

### Anti-spoofing

Implemented quantization for anti-spoofing models, achieving 60% faster inference speed and 4x memory footprint reduction.

- Conducted 120 experiments with quantization-aware training on ConvNext and EfficientNet architectures.
- Enabled use of models with twice the parameter count while maintaining 90% of their original quality.

Led domain generalization research for anti-spoofing, improving upon market-leading model by additional 1-2% error rate.

- Supervised two master's thesis projects, guiding experiment design and result interpretation. 
- Explored self-supervised techniques like PatchNet and SSDG, tailored for anti-spoofing applications.

## At CROC

### Industrial safety

Engineered industrial safety framework deployed at four major facilities including oil drilling sites and cargo ships.

- Developed system to monitor 10 safety scenarios using 300 hours of annotated footage.
- Identified 500 safety violations in first two months of deployment at oil drilling facility, enhancing workplace safety.

Developed drone-based aerial photography solution for 20,000 m² orange farm, processing 500MB GeoTIFF files.

- Implemented efficient cutting and stitching process to enable 224x224 CNN object detection without data duplication.
- Successful client demo led to project adoption for production by separate team.


